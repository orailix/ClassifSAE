import os
import re
import gc
import torch
import json
import joblib
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from functools import partial
import torch.nn.functional as F
from loguru import logger
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
from .baseline_method import ConceptNet, ConceptNetEncoder, HIConcept, HIConceptEncoder
import ctypes, ctypes.util
from sae_implementation import TrainingSAE
from torch.utils.data import Dataset
from typing import List, Tuple, Union
from ..llm_classifier_tuning import process_dataset


class ActivationDataset(Dataset):
    """
    Stores five‑tuple samples:
        0. input_ids
        1. cache
        2. output
        3. label
        4. attention_mask
    """
    _NUM_FIELDS = 5  # expected tensors per sample

    def __init__(self) -> None:
        self.data_block: List[Tuple[torch.Tensor, ...]] = []

    def append(self, *tensors: torch.Tensor) -> None:
        """Add one sample to the dataset (expects exactly five tensors)."""
        if len(tensors) != self._NUM_FIELDS:
            raise ValueError(
                f"Expected {self._NUM_FIELDS} tensors, got {len(tensors)}"
            )

        self.data_block.append(tuple(t.detach().cpu() for t in tensors))

    def __len__(self) -> int:
        return len(self.data_block)

    def __getitem__(
        self, idx: Union[int, List[int], torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        idx = self._normalize_index(idx)
        inp_ids, cache, out, label, attn_mask = self.data_block[idx]

        return {
            "input_ids": inp_ids,
            "cache": cache,
            "output": out,
            "label": label,
            "attention_mask": attn_mask,
        }

   
    @staticmethod
    def _normalize_index(idx: Union[int, List[int], torch.Tensor]) -> int:
        """Accepts int, single‑element list, or single‑element tensor."""
        if isinstance(idx, torch.Tensor):
            if idx.numel() != 1:
                raise ValueError("Index tensor must have exactly one element")
            idx = idx.item()

        elif isinstance(idx, list):
            if len(idx) != 1:
                raise ValueError("Index list must have exactly one element")
            idx = idx[0]

        if not isinstance(idx, int):
            raise TypeError(f"Index must resolve to int, got {type(idx)}")

        return idx

def collate_single_sample(batch):
    # batch is a list of samples; with batch_size=1 it's [sample]
    return batch[0]

def is_empty_dir(path):
    """
    Returns True if `path` is an existing directory
    containing no non-hidden files or subdirectories.
    Hidden entries are those whose names start with '.'.
    """
    try:
        # list all entries…
        entries = os.listdir(path)
        # …but filter out hidden ones
        visible = [name for name in entries if not name.startswith('.')]
        return len(visible) == 0
    except FileNotFoundError:
        raise ValueError(f"No such directory: {path!r}")
    except NotADirectoryError:
        raise ValueError(f"Not a directory: {path!r}")
    except PermissionError:
        raise ValueError(f"Permission denied: {path!r}")

def get_sae_path(cfg_sae):

    d_sae = cfg_sae.methods_args['d_sae']
    feature_activation_rate = cfg_sae.methods_args['feature_activation_rate']
    supervised_classification = cfg_sae.methods_args['supervised_classification']
    sae_name = cfg_sae.sae_name
    sae_checkpoint = cfg_sae.sae_checkpoint
    latest_version = cfg_sae.latest_version
    checkpoint_version = cfg_sae.checkpoint_version

    if feature_activation_rate == 1.:
        feature_activation_rate='1.0'

    if supervised_classification:
        sae_name = sae_name.replace("sae", "ClassifSAE")

    full_sae_name = f"{sae_name}_d_sae_{d_sae}_activation_rate_{feature_activation_rate}"
    sae_load_dir = os.path.join(sae_checkpoint, full_sae_name)

    # The different versions of the trained sae are saved in folders named 'final_X'
    if not (os.path.exists(sae_load_dir) and os.path.isdir(sae_load_dir)):
        raise ValueError(f"The sae name {full_sae_name} is not present in {sae_checkpoint}")
    
    numbers_steps = []
    version_names = {}
    for version in os.listdir(sae_load_dir):
        if os.path.isdir(os.path.join(sae_load_dir, version)) and re.match(r'final_\d+', version):

            number_steps = int(version.split('_')[-1])
            numbers_steps.append(number_steps)
            version_names[number_steps] = version
    if numbers_steps==[]:
        raise ValueError(f"No checkpoints available in {sae_load_dir}. Verify that the checkpoints folders are named according to the template 'final_X' with 'X' an integer.")
    
    if latest_version:
        numbers_steps.sort(reverse=True)
        max_number_steps = numbers_steps[0]
        selected_version = version_names[max_number_steps]
        checkpoint_version = max_number_steps
    else:
        #Find the model tuned the with closest number of steps to the one provided in checkpoint_version 
        closest_number_steps = min(numbers_steps, key=lambda x: abs(x - checkpoint_version))
        selected_version = version_names[closest_number_steps]
    
    sae_path = os.path.join(sae_load_dir,selected_version)
    full_sae_name = f"{full_sae_name}_{selected_version}"
    logger.info(f"Loading SAE from {full_sae_name}")

    return sae_path, full_sae_name


# In evaluation inference, in the forward pass we ablate every features which are not in z_class, to ensure an upper bound on the number of investigated concepts imposed by the user. 
def forward_on_selected_features(feature_acts,selected_features):

    if isinstance(selected_features,list):
        selected_features_tensor = torch.tensor(selected_features.copy())
    else:
        selected_features_tensor = selected_features
    selected_features_tensor = selected_features_tensor.to(feature_acts.device)

    feature_acts_mask = torch.zeros_like(feature_acts,dtype=torch.bool)
    if feature_acts.dim() == 2:         # [B, F]
        feature_acts_mask[:,selected_features_tensor] = True
    elif feature_acts.dim() == 3: ## [B, L, F]
        feature_acts_mask[:,:,selected_features_tensor] = True
    else:
        raise ValueError("Unsupported feature_acts shape")
    
    feature_acts_masked = feature_acts * feature_acts_mask

    return feature_acts_masked


# To avoid leakage of information across the latest layers, when we reconstruct the classification hidden state, we ablate all the other hidden states of the previous tokens
def reconstr_hook_classification_token_single_element(activation, hook, replacement):
    return replacement


# Used to compute the Recovery Accuracy
def count_same_match(
    logits_original : torch.Tensor,
    logits_reconstruction : torch.Tensor
):
 
    y_pred_original = logits_original.argmax(dim=1)
    y_pred_reconstruction = logits_reconstruction.argmax(dim=1)

    return (y_pred_original==y_pred_reconstruction).sum()


# Retrieve text from tokens ids
def decode_dataset(tokenized_dataset,tokenizer):
    
    original_texts = []
    
    for input_ids in tokenized_dataset['input_ids']:
        # Decoding the token IDs back to text
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        original_texts.append(text)

    return original_texts


def flush_shard(ds, shard_id: int, save_dir: str):
    if len(ds) == 0:
        return

    fname = os.path.join(save_dir, f"shard_{shard_id:03d}.pt")
    torch.save(ds.data_block, fname)
    ds.data_block.clear()

    # run GC + trim here
    gc.collect()
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        # Not available on non-glibc platforms (e.g., Windows); ignore.
        pass
    

def load_all_shards(save_dir: str):
    """
    Re‑assemble every shard back into a single ActivationDataset.
    """
    from glob import glob
    files = sorted(glob(os.path.join(save_dir, "shard_*.pt")))

    ds = ActivationDataset()
    for f in files:
        for item in torch.load(f, map_location="cpu",weights_only=True):
            ds.data_block.append(item)
    return ds



# Create a dataset with the activations from the given split that we save to speed up the causality metrics computation
def create_activations_dataset(
            cfg_model,
            cfg_concept,
            model,
            tokenizer,
            data_collator,
            decoder,
            add_template,
            max_ctx,
            device,
            cache_activations_dir,
            labels_tokens_id=None,
            eos=False
            ):

    # Load the corresponding tokenized dataset
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer,add_template=add_template,decoder=decoder,max_len=max_ctx) 

    # Potential prompt-tuning if specified
    if cfg_model.prompt_tuning and decoder:

        vtok = cfg_model.prompt_tuning_params['vtok']
        total_steps = cfg_model.prompt_tuning_params['total_steps']

        # Check if prompt embeddings have already been generated for the specified combination (model,dataset)
        dir_corresponding_prompt_embeddings=os.path.join(cfg_model.prompt_embeddings_dir,cfg_model.model_name,cfg_model.dataset_name)
        file_name = f"embeddings_prompt_{vtok}_{total_steps}.pt"
        prompt_embeddings_file = os.path.join(dir_corresponding_prompt_embeddings,file_name)
        
        assert (
            os.path.isdir(dir_corresponding_prompt_embeddings)
            and os.path.isfile(prompt_embeddings_file)
        ), (
            f"Unable to find the prompt embeddings specific to the pair model-dataset: ({cfg_model.model_name},{cfg_model.dataset_name}) with the corresponding parameters: vtok: {vtok} and total_steps: {total_steps}. Either specify prompt_tuning to False in the configuration file to continue or realize the prompt tuning first with `evaluation.py` in module `llm_classifier_tuning`"
        )

        logger.info(f"Prompt tuning is selected, we append the trained prompt embeddings for evaluation. The prompt embeddings have been trained on the pair model-dataset: ({cfg_model.model_name},{cfg_model.dataset_name}) with the parameters: vtok: {vtok} and total_steps: {total_steps}")

        logger.info(f"file loaded : {prompt_embeddings_file}")

        prompt_embeddings = torch.load(prompt_embeddings_file, map_location=device)   
    else:
        prompt_embeddings = None
    
    
    original_text_used = cache_activations_with_labels(model,
                                                        dataset_tokenized,
                                                        data_collator,
                                                        tokenizer,
                                                        path_directory=cache_activations_dir,
                                                        hook_name = cfg_concept.hook_name,
                                                        hook_layer = cfg_concept.hook_layer,
                                                        device=device,
                                                        decoder=decoder,
                                                        labels_tokens_id=labels_tokens_id,
                                                        eos = eos,
                                                        prompt_embeddings=prompt_embeddings)

    original_text_file_path = os.path.join(cache_activations_dir,f'original_text.json')
    with open(original_text_file_path, 'w') as f:
        json.dump(original_text_used, f)


# Cache LLM classifier activations on the split dataset to speed up the subsequent multiple inferences for the computation of the causality metrics
def cache_activations_with_labels( model,
                                    dataset, #expected to be tokenized
                                    data_collator,
                                    tokenizer,
                                    path_directory,
                                    hook_name,
                                    hook_layer,
                                    device,
                                    decoder,
                                    labels_tokens_id=None,
                                    eos=None,
                                    prompt_embeddings=None):

    FLUSH_EVERY_BATCH = 100

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator,shuffle=False, num_workers=0)
    
    # Save the different activations to speed the causality calculations
    activations_dataset = ActivationDataset()


    if labels_tokens_id is not None:
        # Filter predictions based on the logits corresponding to an accepted answer
        # Useful if we do classification prediction with a Decoder-only LLM + template
        ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                                key=lambda kv: kv[1])]

    # Evaluation loop
    model.eval()
    with torch.no_grad():
       
        step = 0
        message = f"Caching of the hook {hook_name} activations" if decoder else f"Caching of the activations exiting the {hook_layer}-th layer"
        for batch in tqdm(dataloader, desc=message, unit="batch"):
            step += 1
            input_ids = batch['input_ids'].to(device)
            labels_key = next(k for k in ("label", "labels") if k in batch)
            labels = batch[labels_key]
            attention_mask = batch['attention_mask'].to(dtype=int).to(device)
           
            if prompt_embeddings is not None and decoder:

                emb = model.embed(input_ids)
                bs = input_ids.size(0)
                max_ctx = model.cfg.n_ctx
                vtok = prompt_embeddings.size(0)

                if (emb.size(1)+vtok) > max_ctx:
                    emb = emb[:, -(max_ctx-vtok):]         # keep the right-most tokens
                    attention_mask = attention_mask[:, -(max_ctx-vtok):]


                P = prompt_embeddings.unsqueeze(0).expand(bs, -1, -1)
                new_emb = torch.cat([P, emb], dim=1) 

                prefix_mask = torch.ones(bs, vtok, device=model.cfg.device).long()
                attention_mask   = torch.cat([prefix_mask, attention_mask], dim=1) 

                def override_embed(resid_pre, hook):        # resid_pre is (B, L, d_model)
                    return new_emb                       

                B, L, d_model = new_emb.shape         # (batch, seq_len, d_model)
                dummy_tokens  = torch.zeros(B, L, dtype=torch.long, device=model.cfg.device)
                
                # Manually add at the beginning of the prompt, the computed Prompt embedding to align the LLM with the classification task (alternative to the fine-tuning)
                with model.hooks(
                        fwd_hooks=[("hook_embed", override_embed)]
                ):
                    outputs, cache = model.run_with_cache(
                        dummy_tokens,
                        attention_mask=attention_mask,
                        names_filter=hook_name,
                        prepend_bos=False
                    )
                
                # Decoder-inly setup, we cache the activations of the token preceding the classification token in the sentence
                cache = cache[hook_name][:,(-2-int(eos)),:]
                outputs_logits = outputs[:,(-2-int(eos))].contiguous().view(-1, outputs.shape[-1])
                outputs_logits = outputs_logits[:,ordered_old_idxs]  #shape : (bs,nb_classes)
                

            else:    

                # Decoder only LLM
                if decoder:

                    # Output logits + internal activations of the layer of interest
                    outputs, cache = model.run_with_cache(input_ids,
                                                        attention_mask=attention_mask,
                                                        names_filter=hook_name,
                                                        prepend_bos=False)
                
                    # Decoder-inly setup, we cache the activations of the token preceding the classification token in the sentence
                    outputs_logits = outputs[:,(-2-int(eos))].contiguous().view(-1, outputs.shape[-1])
                    outputs_logits = outputs_logits[:,ordered_old_idxs]  #shape : (bs,nb_classes)
                    cache = cache[hook_name][:,(-2-int(eos)),:]
                
                # Encoder only LLM
                else:
                    
                    out = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,        
                        return_dict=True
                    )
                    cache = out.hidden_states[hook_layer][:,0,:]
                    outputs_logits = out.logits #shape : (bs,nb_classes)
            
            # Save the data so that it can be reused later for causality metrics computation 
            activations_dataset.append(
                input_ids.cpu(),
                cache.cpu(),     
                outputs_logits.cpu(),
                labels,                 
                attention_mask.cpu(),
            )

            # flush every N batches
            if step % FLUSH_EVERY_BATCH == 0:
                flush_shard(activations_dataset, shard_id=step // FLUSH_EVERY_BATCH, save_dir=path_directory)
        

    # after the loop flush whatever is left
    flush_shard(
        activations_dataset,
        shard_id=(step // FLUSH_EVERY_BATCH) + 1,
        save_dir=path_directory,
    )

    # Return the decoded sentences used to generate the activations (it ensures aligned correspondence)
    original_text_used = decode_dataset(dataset, tokenizer)

    return original_text_used


# # Load interpretation model
def load_interp_model(
    cfg_concept,
    activations_dataset,
    decoder,
    device
):

    if cfg_concept.method_name=='concept_shap':

        method_name = cfg_concept.method_name
        
        n_concepts = cfg_concept.methods_args['n_concepts']
        hidden_dim = cfg_concept.methods_args['hidden_dim']
        thres = cfg_concept.methods_args['thres']
        activations_dataloader = DataLoader(activations_dataset,batch_size=1,collate_fn=collate_single_sample)
       
        try:
            first_batch = next(iter(activations_dataloader))
        except StopIteration:
            raise ValueError("Dataloader is empty.")
            
        cache_tensor = first_batch["cache"].to(device)
        embed_dim = cache_tensor.shape[-1]
            
        if decoder: 
            interp_model = ConceptNet(n_concepts, embed_dim, hidden_dim, thres).to(device)
        else:
            interp_model = ConceptNetEncoder(n_concepts, embed_dim, hidden_dim, thres).to(device)

        conceptnet_weights_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}_{thres:.1f}',f'conceptshap_weights.pth')

        try:
            interp_model.load_state_dict(torch.load(conceptnet_weights_path,weights_only=True))
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: '{conceptnet_weights_path}' not found. Ensure you have trained the corresponding ConceptNet before evaluation of its concepts with 'train-baseline'")

        matrix_concepts = interp_model.concept #(embedding_dim,n_components)
        mixing_matrix=None
        # Name of the specifc concept-based method used
        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_concept.hook_layer}'

    elif cfg_concept.method_name=='hi_concept':

        method_name = cfg_concept.method_name
        n_concepts = cfg_concept.methods_args['n_concepts']
        hidden_dim = cfg_concept.methods_args['hidden_dim']
        activations_dataloader = DataLoader(activations_dataset,batch_size=1,collate_fn=collate_single_sample)

        try:
            first_batch = next(iter(activations_dataloader))
        except StopIteration:
            raise ValueError("Dataloader is empty.")

        cache_tensor = first_batch["cache"].to(device)
        embed_dim = cache_tensor.shape[-1]

        if decoder:
            interp_model = HIConcept(n_concepts, embed_dim, hidden_dim).to(device)
        else:
            interp_model = HIConceptEncoder(n_concepts, embed_dim, hidden_dim).to(device)            
        hi_concept_weights_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}',f'hiconcept_weights.pth')

        try:
            interp_model.load_state_dict(torch.load(hi_concept_weights_path,weights_only=True))
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: '{hi_concept_weights_path}' not found. Ensure you have trained the corresponding HIConcept before evaluation of its concepts with 'train-baseline'")

        matrix_concepts = interp_model.concept #(embedding_dim,n_components)
        mixing_matrix=None
        # Name of the specifc concept-based method used
        concept_model_name=f'hi_concept_{n_concepts}_layer_{cfg_concept.hook_layer}'

    elif cfg_concept.method_name=='ica':

        method_name = cfg_concept.method_name
        ica_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}',f'ica.pkl')
        mixing_matrix_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}',f'mixing_matrix.npy')

        try : 
            interp_model = joblib.load(ica_path)
            mixing_matrix = np.load(mixing_matrix_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Either '{ica_path}' or '{mixing_matrix_path}' not found. Ensure you have fitted the corresponding ICA before evaluation of its concepts with 'train-baseline'")
    
        device = "cpu"

        matrix_concepts = interp_model.components_.T #(embedding_dim,n_components)
        matrix_concepts = torch.from_numpy(matrix_concepts)

        # Name of the specifc concept-based method used
        concept_model_name=f'ica_{interp_model.components_.shape[0]}_layer_{cfg_concept.hook_layer}'

    elif cfg_concept.method_name=='sae':

        # For the SAE, we implement additional post-processing steps for the selection of features since the hidden layer dimension may be larger than the number of desired concepts
        assert  cfg_concept.methods_args['features_selection'] in ['truncation', 'logistic_regression'], f"Error: For the post-selection of the SAE features, the options supported in methods_args['features_selection'] are either 'truncation' or 'logistic_regression' "
    
        # Method used to extract z_class from z_sae in post-processing
        if cfg_concept.methods_args['features_selection'] == 'truncation':
            method_name = "sae_truncation"
        elif cfg_concept.methods_args['features_selection'] == 'logistic_regression':
            method_name = "sae_logistic_regression"

        # Retrieve the trained sae path
        sae_path, concept_model_name = get_sae_path(cfg_concept)

        #Load the local trained SAE
        interp_model = TrainingSAE.load_from_pretrained(sae_path,cfg_concept.methods_args['device'])
        
        mixing_matrix=None
        matrix_concepts = interp_model.W_dec.T #(embedding_dim,n_components)

    return interp_model, matrix_concepts, mixing_matrix, method_name, concept_model_name, device


def compute_loss_classif(
    true_labels: torch.Tensor,
    outputs_logits: torch.Tensor,
    reduction: str = "mean"
):
  """Computes the loss that focuses on the classification."""

  loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)

  return loss_ce(outputs_logits,true_labels)


# Accuracy, F1-score metrics on the classification tasks when evaluating the models on the test split
def update_metrics(
    true_labels : torch.Tensor,
    logits : torch.Tensor,
    dict_metrics : dict,
):
    
    logits_prediction = logits.argmax(dim=1)

    exact_matches = (logits_prediction==true_labels)
    count_exact_matches = exact_matches.sum() #tensor

    for label in range(logits.shape[1]):

      # In case the values do not already exist (typically the first time we call this function)
      dict_metrics.setdefault(f'number real samples_{label}',0)
      dict_metrics.setdefault(f'true matches_{label}',0)
      dict_metrics.setdefault(f'number predicted samples_{label}',0)

      position_label = (true_labels==label)
      number_samples_labels = (true_labels==label).sum().item()
      dict_metrics[f'number real samples_{label}'] += number_samples_labels

      exact_matches_label = position_label & exact_matches
      count_exact_matches_label = exact_matches_label.sum().item()
      dict_metrics[f'true matches_{label}']+= count_exact_matches_label

      count_predicted_label = (logits_prediction==label).sum().item()
      dict_metrics[f'number predicted samples_{label}'] += count_predicted_label
      
      dict_metrics[f'recall_{label}'] = 0 if dict_metrics[f'number real samples_{label}']==0 else dict_metrics[f'true matches_{label}'] / dict_metrics[f'number real samples_{label}']
      dict_metrics[f'precision_{label}'] = 0 if  dict_metrics[f'number predicted samples_{label}']==0  else dict_metrics[f'true matches_{label}'] / dict_metrics[f'number predicted samples_{label}']
      dict_metrics[f'f1-score_{label}'] = 0 if (dict_metrics[f'recall_{label}'] + dict_metrics[f'precision_{label}'])==0  else 2 * dict_metrics[f'recall_{label}'] * dict_metrics[f'precision_{label}'] / (dict_metrics[f'recall_{label}'] + dict_metrics[f'precision_{label}'])
        
    return count_exact_matches



# Compute the concepts from the internal activations given the interpretability model provided
def compute_concepts_from_internal_activations(interp_model,cache,method_name,selected_features,device):

    if method_name=='concept_shap':
    
        #concept_score_thres_prob
        cache = cache.to(device)
        cache = cache.squeeze(1)
        feature_acts, _, _ = interp_model.concepts_activations(cache)
        feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
    
    elif method_name=='hi_concept':

        cache = cache.to(device)
        #concept_score_thres_prob
        cache = cache.squeeze(1)
        feature_acts, _, _, _,_ = interp_model.concepts_activations(cache)
        feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)   
    
    elif method_name=='sae':

        cache = cache.to(device)
        # Use the SAE
        feature_acts, acts_without_process = interp_model.encode_with_hidden_pre(cache)
        feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
        feature_acts_masked = feature_acts_masked.squeeze(1) 
    
    elif method_name=='ica':
        cache = cache.squeeze(1)
        embeddings_sentences_numpy = cache.detach().cpu().numpy()
        #ica_activations
        feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)).to(device=device, dtype=cache.dtype)
        feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
        feature_acts_masked = feature_acts_masked.to(device)
        
    
    return feature_acts_masked


# Run the inference by enforcing the forward pass to go through the concepts bottleneck layer (z_class) 
# Re-compute the new logits outputed by the model given the new embedding reconstruction obtained from the computed concepts
def inference_through_z_class(decoder,model,cache_sentence,reconstruct_act,hook_layer,hook_name,ordered_old_idxs=None):

    device = getattr(model, "cfg.device", next(model.parameters()).device)

    # B, T, H = reconstruct_act.shape
    # Since z_ctx can be masked, we rescale the reconstructed vector based on the norm of the original embedding
    hidden = reconstruct_act
    hidden_norm = hidden.norm(p=2,dim=-1)
    cache_norm = cache_sentence.norm(p=2,dim=-1)
    factor = cache_norm.squeeze(1) / (hidden_norm.squeeze(1)+1e-9)
    hidden = hidden * factor[:,None,None]
    hidden = hidden.to(device)

    # Inference on Decoder-only LLM
    if decoder:
               
        # To avoid leakage of information across the latest layers, when we reconstruct the classification hidden state from the concepts activations, we ablate all the other hidden states of the previous tokens
        # Run with dummy input; override residual via hook to inject reconstructed activations
        dummy_input = torch.zeros_like(cache_sentence, device=device)
        
        logits_reconstruction = model.run_with_hooks(
            dummy_input,
            start_at_layer=hook_layer,
            fwd_hooks=[
                (
                    hook_name,
                    partial(reconstr_hook_classification_token_single_element, replacement=hidden),
                ) ],
            return_type="logits",
        )
    
        logits_reconstruction = logits_reconstruction.contiguous().view(-1, logits_reconstruction.shape[-1])

        # Filter predictions based on the logits corresponding to an accepted answer
        logits_reconstruction = logits_reconstruction[:,ordered_old_idxs] #shape : (bs,nb_classes)

    # Inference on Encoder-only LLM
    else:
    
        attn_mask = torch.ones(reconstruct_act.size()[:2], dtype=torch.long).to(device)
        extended = model.get_extended_attention_mask(
            attention_mask=attn_mask,
            input_shape=attn_mask.shape
        )   

        base = getattr(model, model.base_model_prefix)
        encoder = base.encoder                         # BERT->bert.encoder, debrta->deberta.encoder

        for idx in range(hook_layer, model.config.num_hidden_layers):
            hidden = encoder.layer[idx](hidden, attention_mask=extended, output_attentions=False)[0]
        
        pooler = getattr(model, "pooler", None) or getattr(base, "pooler", None)
        pooled = pooler(hidden) if pooler is not None else hidden[:, 0]

        logits_reconstruction = model.classifier(model.dropout(pooled))

    return logits_reconstruction


# Compute the reconstructed embeddings and z_class activations from the original embedding
def reconstruct_from_method(method_name, interp_model, cache_sentence, selected_features, device, mixing_matrix=None):
    if method_name in ("concept_shap", "hi_concept"):
        cache_flat = cache_sentence.squeeze(1).to(device)
        feature_acts = interp_model.concepts_activations(cache_flat)[0]
        feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
        reconstruct_act = interp_model.reconstruct_from_concepts_activations(feature_acts_masked).unsqueeze(1).to(device)
        return reconstruct_act, feature_acts_masked

    if method_name == "sae":
        feature_acts, _ = interp_model.encode_with_hidden_pre(cache_sentence.to(device))
        feature_acts_masked = forward_on_selected_features(feature_acts, selected_features)
        reconstruct_act = interp_model.decode(feature_acts_masked).to(device)
        return reconstruct_act, feature_acts_masked.squeeze(1) 

    if method_name == "ica":
        cache_flat = cache_sentence.squeeze(1)
        embeddings_sentences_numpy = cache_flat.detach().cpu().numpy()
        feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)).to(device=device, dtype=cache_flat.dtype)
        feature_acts_masked = forward_on_selected_features(feature_acts, selected_features)
        ica_mean = torch.from_numpy(interp_model.mean_).to(device=device, dtype=cache_flat.dtype)
        mixing_matrix = torch.from_numpy(mixing_matrix).to(device=device, dtype=cache_flat.dtype)
        
        reconstruct_act = feature_acts_masked @ mixing_matrix.T + ica_mean
        reconstruct_act = reconstruct_act.unsqueeze(1).to(device)
        
        return reconstruct_act, feature_acts_masked
    
    raise ValueError(f"Unknown method: {method_name}")


# Run full inference of the LLM classifier on the test sentence split when plugged with the investigated concepts
# We measure the recovery accuracy of the model's predictions when the original activations goes through the concepts bottleneck layer z_class
def eval_hook_loss(
    decoder:bool,
    model,
    interp_model,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    device,
    labels_tokens_id,
    mixing_matrix=None):

    dict_metrics_original = {}
    dict_metrics_reconstruction = {}
    total_matches_original = 0
    total_matches_reconstruction = 0
    total_same_predictions = 0
    original_total_loss = 0
    reconstruction_total_loss = 0
    total_sentences = 0
    
    activations_dataloader = DataLoader(activations_dataset,batch_size=1,shuffle=False,collate_fn=collate_single_sample)
    
    # # Each sample of 'activations_dataloader' contains activations of multiple sentences
    # nb_sentences_per_sample = next(iter(activations_dataloader))["input_ids"].shape[0]
    # nb_sentences = len(activations_dataset)*nb_sentences_per_sample


    if decoder:
        # Filter predictions based on the logits corresponding to an accepted answer
        # Useful when using a Decoder-only LLM to make the classification prediction so that it get rids of the other vocabulary tokens when inspecting the logits
        ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                                key=lambda kv: kv[1])]
    else:
        ordered_old_idxs=None
    

    # Evaluation loop
    model.eval()
    if method_name in ["sae","concept_shap","hi_concept"]:
        interp_model.eval()
    with torch.no_grad():
        
        feature_activations_list = []
        original_activation_list = []
        true_labels_list = []
        logits_original_list = []
        logits_reconstruction_list = []
        
        for batch in tqdm(activations_dataloader, desc="Forward Passes on the classifier LLM model", unit="batch"):


            cache = batch["cache"].to(device) #(bs,d_in)
            logits_original = batch["output"].to(device) # shape : [bs, nb classes]
            true_labels = batch["label"].to(device) # Ground truth labels
            
            bs = true_labels.shape[0]
            total_sentences += bs            
            
            cache_sentence = cache.unsqueeze(1) #(bs,1,d_in)

            reconstruct_act, feature_acts_masked = reconstruct_from_method(method_name, interp_model, cache_sentence,
                                                                           selected_features, device, mixing_matrix)

            # In addition to the SAE metrics, we want to store feature activations and model predictions
            feature_activations_list.append(feature_acts_masked.cpu())
            original_activation_list.append(cache.cpu())
            true_labels_list.append(true_labels.cpu())

            reconstruct_act = reconstruct_act.to(dtype=next(model.parameters()).dtype)

            # Run the inference by enforcing the forward pass to go through the concepts bottleneck layer. 
            logits_reconstruction = inference_through_z_class(decoder,model,cache_sentence,reconstruct_act,hook_layer,hook_name,ordered_old_idxs)
            logits_original_list.append(logits_original.cpu())
            logits_reconstruction_list.append(logits_reconstruction.cpu())

            logits_reconstruction = logits_reconstruction.to(device)
            # Compute the original classification cross-entropy loss and the same loss obtained by only using the reconstructed hidden state of the token preceding the class' answer.
            original_loss = compute_loss_classif(true_labels,logits_original)
            original_total_loss  += original_loss
            reconstruction_loss = compute_loss_classif(true_labels,logits_reconstruction)
            reconstruction_total_loss += reconstruction_loss

            # We compute the variation in true accuracy
            acc_original = update_metrics(true_labels,logits_original,dict_metrics_original)
            acc_reconstruction = update_metrics(true_labels,logits_reconstruction,dict_metrics_reconstruction)
            total_matches_original += acc_original.item()
            total_matches_reconstruction += acc_reconstruction.item()
            # We compute the recovery accuracy metric
            count_same_predictions = count_same_match(logits_original,logits_reconstruction)
            total_same_predictions += count_same_predictions.item()
            
        del cache
        
        accuracy_original = total_matches_original / total_sentences
        accuracy_reconstruction = total_matches_reconstruction / total_sentences
        recovery_accuracy = total_same_predictions / total_sentences
        reconstruction_mean_loss = reconstruction_total_loss / len(activations_dataloader)
        original_mean_loss = original_total_loss / len(activations_dataloader)

        print(f"\n Method name : {method_name}")
        print(f"\nOriginal classification crossentropy mean loss : {original_mean_loss:.3f} ({total_matches_original} matches) - Reconstruction classification crossentropy mean loss : {reconstruction_mean_loss:.3f} (Computed over {total_sentences} sentences) ")
        print(f'\nRecovery accuracy : {recovery_accuracy:.3f}')
        print(f"\nOriginal accuracy of the model : {accuracy_original:.3f} - Accuracy of the model when plugging the reconstruction hidden states : {accuracy_reconstruction:.3f}")

        # One hidden state for each sentence (decoder : hidden state of the token preceding the class-generating token / encoder : hidden state of the [CLS] token). 
        feature_activations_tensor = torch.cat(feature_activations_list,dim=0)
        original_activation_tensor = torch.cat(original_activation_list,dim=0)

        logits_reconstruction_tensor = torch.cat(logits_reconstruction_list,dim=0)
        logits_original_tensor = torch.cat(logits_original_list,dim=0)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mean_activations = feature_activations_tensor.mean(dim=0)

        max_activity = 0
        mean_activity = 0
        dead_threshold = 1e-6
        n = feature_activations_tensor.shape[0]
        # Compute statistics on the distribution of activations of the features
        for feature_number in selected_features:
            active_samples = torch.sum(feature_activations_tensor[:,feature_number].abs() > dead_threshold)
            percentage_activity = ((active_samples / n) * 100).item()
            mean_activity += percentage_activity
            if percentage_activity > max_activity:
                max_activity = percentage_activity
        mean_activity /= len(selected_features)


        print(f"\nAveraged frequency of activation of the selected features across the evaluated dataset : {mean_activity:.3f}% - Highest frequency of activation : {max_activity:.3f}%")

        performance = {'Original Mean Loss': original_mean_loss.item(), 'Reconstruction Mean loss': reconstruction_mean_loss.item(), 'Number sentences':n, 'Recovery accuracy':recovery_accuracy, 'Original accuracy':accuracy_original, 'Reconstruction accuracy' : accuracy_reconstruction, 'Averaged activation frequency' : mean_activity, 'Highest activation frequency' : max_activity}
        # We merge the macro information of the concepts with the dictionary of the variation metrics
        performance = performance |  dict_metrics_original | dict_metrics_reconstruction
        
        # Return the ground truth labels of the prompts
        true_labels_tensor = torch.cat(true_labels_list,dim=0)
        
        return performance , mean_activations, {'concepts activations' : feature_activations_tensor, 'original activations' : original_activation_tensor, 'true labels' : true_labels_tensor, 'original logits' : logits_original_tensor ,'logits from reconstruction' : logits_reconstruction_tensor}
        

def design_figure(W_dec_pca, sizes, np_prototypes_pca,
                  feature_colors, normalized_class_scores,
                  labels_names):
   

    if feature_colors is None:
        feature_colors = np.array(['#636EFA', '#EF553B', '#00CC96', '#3D2B1F'])

    all_x = np.concatenate((W_dec_pca[:, 0], np_prototypes_pca[:, 0]))
    all_y = np.concatenate((W_dec_pca[:, 1], np_prototypes_pca[:, 1]))
    span  = max(all_x.ptp(), all_y.ptp())

    base_r = 0.03 * span          # ≈ 3 % of the plot span

    fig, ax = plt.subplots(figsize=(8, 8))

    # ── prototype triangles ───────────────────────────────────────────
    tri_h = 1.8 * base_r
    for (cx, cy) in np_prototypes_pca:
        verts = np.array([[cx,             cy + tri_h],
                          [cx - 0.577*tri_h, cy - tri_h/2],
                          [cx + 0.577*tri_h, cy - tri_h/2]])
        ax.fill(verts[:, 0], verts[:, 1],
                facecolor='orange', edgecolor='k',
                alpha=.9, zorder=6)

    for (cx, cy), lab in zip(np_prototypes_pca, labels_names.values()):
        ax.text(cx - 1.1*base_r, cy + 0.1*base_r, lab,
                fontsize=13, weight='bold',
                bbox=dict(facecolor='white', alpha=.75, boxstyle='round,pad=.25'),
                zorder=10)

    # ── pie charts ────────────────────────────────────────────────────
    def draw_pie(center, parts, radius):
        start = 0
        for frac, col in zip(parts, feature_colors):
            if frac == 0:
                continue
            end = start + frac * 360
            ax.add_patch(Wedge(center, radius, start, end,
                               facecolor=col, edgecolor='white',
                               lw=.5, alpha=.85, zorder=5))
            start = end

    radius_scale = base_r * (sizes / sizes.max())   # <── SCALE LINE
    for centre, fractions, r in zip(W_dec_pca[:, :2],
                                    normalized_class_scores.T,
                                    radius_scale):
        if r == 0:
            continue                                # invisible by design
        draw_pie(centre, fractions, r)

    # ── cosmetics ─────────────────────────────────────────────────────
    ax.axhline(0, color='grey', lw=1, zorder=1)
    ax.axvline(0, color='grey', lw=1, zorder=1)
    ax.grid(True, ls='--', alpha=.4, zorder=0)

    pad = 0.1 * span
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('PCA component 1')
    ax.set_ylabel('PCA component 2')

    return ax, fig



# For SAE, 2D PCA representation of the learned concepts in z_class projected into the activations space of the investigated LLM layer 
def pca_activations_projection(W_dec_numpy,mean_activations,dict_analysis_features,label_names):

    #2D projection of the decoder features/rows on PCA axis computed with original activations 
    logger.info(f"Compute the PCA plan of the original activations and projection the features/decoder rows on it")
    '''PCA in the activation space'''
    N = W_dec_numpy.shape[0]
    
    labels = dict_analysis_features['true labels'].numpy()
    original_activations = dict_analysis_features['original activations'].numpy()
    sae_activations = dict_analysis_features['concepts activations'].numpy()
    unique_labels = np.unique(labels) #sorted unique elements of the array

    #Put each vector to norm 1
    norms = np.linalg.norm(original_activations, axis=1, keepdims=True)
    original_activations_norm = original_activations / norms

    #Compute class prototype activations by using test data
    hidden_size_prototype_class = {}
    for label in unique_labels:
        indices = np.where(labels==label)[0]
        if indices.size > 0: #Is at least one sample is associated to this label
            hidden_size_prototype_class[label] = np.mean(original_activations_norm[indices],axis=0)

    #Assign a score to each feature with regard to each class
    feature_score_class = {}
    for label in unique_labels:
        indices = np.where(labels==label)[0]
        if indices.size > 0: #Is at least one sample is associated to this label
            feature_score_class[label] = sae_activations[indices].mean(axis=0)

    feature_score_class_array = np.zeros((unique_labels.size,sae_activations.shape[1]))
    #Concatenate features scores of each class for normalization
    for i,label in enumerate(unique_labels):
        feature_score_class_array[i,:] = feature_score_class[label]
    feature_sums = feature_score_class_array.sum(axis=0)
    indices_dead_features = (feature_sums==0.)
    normalized_class_scores = np.zeros((unique_labels.size,feature_sums.size))
    normalized_class_scores[:,~indices_dead_features] = feature_score_class_array[:,~indices_dead_features] / feature_sums[~indices_dead_features]
    normalized_class_scores[:,indices_dead_features] = (np.ones(unique_labels.size) / unique_labels.size).reshape(-1,1) 

    feature_colors = px.colors.qualitative.Plotly[:unique_labels.size]

    scaler = StandardScaler()
    activations_scaled = scaler.fit_transform(original_activations_norm)    
    pca = PCA(n_components=2,random_state=42)  
    pca.fit(activations_scaled)

    W_dec_features_scaled = scaler.transform(W_dec_numpy)
    W_dec_pca = pca.transform(W_dec_features_scaled)

    #Display the hidden representation prototype for each class
    np_prototypes = np.array(list(hidden_size_prototype_class.values()))  # Convert dictionary values to a NumPy array
    np_prototypes_scaled = scaler.transform(np_prototypes)
    np_prototypes_pca = pca.transform(np_prototypes_scaled)
    sizes = mean_activations.numpy()

    ax, fig = design_figure(W_dec_pca, sizes, np_prototypes_pca,feature_colors, normalized_class_scores,label_names)  
    
    return fig, normalized_class_scores
    
    

def analyze_features(tune_sae,mean_activations,dict_analysis_features,label_names):
    
    W_dec_numpy = tune_sae.W_dec.detach().cpu().numpy()
    fig_pca, _ = pca_activations_projection(W_dec_numpy,mean_activations,dict_analysis_features,label_names)
    
    return fig_pca
    
'''
Retrieve the logits predicted by the model on a subset of sentences `ids_samples` from the tested dataset.
In addition to forcing the forward pass through the concepts-layer bottleneck z_class, we also apply targeted ablations to the features of z_class to compute our causality metrics (with `perturb` a tensor of integer values containing the indices of features in z_class to ablate)
'''
def get_predictions(
    decoder,
    model,
    interp_model,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    perturb,
    device,
    labels_tokens_id=None,
    ids_samples=torch.tensor([]),
    mixing_matrix=None,
):

    if decoder:
        # Filter predictions based on the logits corresponding to an accepted answer
        # Useful when using a Decoder-only LLM to make the classification prediction so that it get rids of the other vocabulary tokens when inspecting the logits
        ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                                key=lambda kv: kv[1])]
    else:
        ordered_old_idxs = None


    dict_metrics_reconstruction = {}
    #For accuracy 
    number_matches = 0

    activations_dataloader = DataLoader(activations_dataset,batch_size=1,collate_fn=collate_single_sample)
    
    # Number of sentences for which the prediction is evaluated
    nb_evaluated_sentences = 0 
    
    logits_reconstruction_list = []

    # Evaluation loop
    model.eval()
    if method_name in ["sae","concept_shap","hi_concept"]:
        interp_model.eval()

    # If we do not want to evaluate on all samples in the activation_dataset
    number_sample = 0
    
    with torch.no_grad():
       
        for num_batch, batch in tqdm(enumerate(activations_dataloader), desc=f"Forward Passes with ablation on {0 if perturb is None else len(perturb)} feature(s) ", unit="batch"):
                
                true_labels = batch["label"] # Ground truth labels
                cache = batch["cache"].to(device)
                
                samples_to_keep = (ids_samples - number_sample)
            
                evaluated_samples  = samples_to_keep[(0 <= samples_to_keep) & (samples_to_keep < true_labels.shape[0])]
                nb_evaluated_sentences += len(evaluated_samples)
                
                cache = cache[evaluated_samples,:]
                cache_sentence = cache.unsqueeze(1)
                feature_acts_masked = compute_concepts_from_internal_activations(interp_model,cache_sentence,method_name,selected_features,device)

                if perturb is not None:
                    feature_acts_masked[:,perturb] = 0
                    
                
                if method_name=='concept_shap':
                    reconstruct_act = interp_model.reconstruct_from_concepts_activations(feature_acts_masked)
                    reconstruct_act = reconstruct_act.unsqueeze(1)
    
                elif method_name=='hi_concept':
                    reconstruct_act = interp_model.reconstruct_from_concepts_activations(feature_acts_masked)
                    reconstruct_act = reconstruct_act.unsqueeze(1)

                elif method_name=='sae':
    
                    feature_acts_masked = feature_acts_masked.unsqueeze(1)
                    reconstruct_act = interp_model.decode(feature_acts_masked)                       
    
                elif method_name=='ica':    
                   
                    ica_mean = torch.from_numpy(interp_model.mean_).to(device=device, dtype=cache.dtype)
                    if isinstance(mixing_matrix, np.ndarray):
                        mixing_matrix = torch.from_numpy(mixing_matrix).to(device=device, dtype=cache.dtype)
                    reconstruct_act = feature_acts_masked.cpu() @ mixing_matrix.T + ica_mean
                    reconstruct_act = (reconstruct_act.unsqueeze(1)).to(device)
                
                reconstruct_act = reconstruct_act.to(dtype=next(model.parameters()).dtype)

                logits_reconstruction = inference_through_z_class(decoder,model,cache_sentence,reconstruct_act,hook_layer,hook_name,ordered_old_idxs)
            
                logits_reconstruction_list.append(logits_reconstruction.cpu())

                number_matches += update_metrics(true_labels[evaluated_samples],logits_reconstruction.cpu(),dict_metrics_reconstruction)
                number_sample+=true_labels.shape[0]

    logits_reconstruction_tensor = torch.cat(logits_reconstruction_list,dim=0)
    predicted_probabilities = F.softmax(logits_reconstruction_tensor, dim=1)

    accuracy = (number_matches / nb_evaluated_sentences).item()
    
    return  predicted_probabilities , accuracy, dict_metrics_reconstruction 

    
# Run joint ablation on features from the same class-specific features segment F_c where the category c is provided in `class_int`. 
# The indices of features to ablate are contained in `ablation_features`.
# We compute the 3 global causality metrics in that context. The proportion of each class-specific features segment F_c ablated is varied throughout the experiment.
def eval_causal_effect_concepts(
    ablation_features,
    class_int,
    decoder,
    model,
    interp_model,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    probs_pred, 
    accuracy,
    device,
    labels_tokens_id,
    mixing_matrix=None):

  
    # We look at the impact of ablating multiple features simultaneously
    probs_pred_ablation, accuracy_ablation, dict_metrics_ablation = get_predictions(
                                                                                    decoder,
                                                                                    model,
                                                                                    interp_model,
                                                                                    activations_dataset,
                                                                                    selected_features,
                                                                                    method_name,
                                                                                    hook_layer,
                                                                                    hook_name,
                                                                                    perturb=ablation_features,
                                                                                    device=device,
                                                                                    labels_tokens_id=labels_tokens_id,
                                                                                    ids_samples=torch.arange(probs_pred.shape[0]),
                                                                                    mixing_matrix=mixing_matrix)

    variation_absolute_accuracy = (accuracy_ablation - accuracy)
    label_predicted_ablation = torch.argmax(probs_pred_ablation, dim=1)
    label_predicted_original = torch.argmax(probs_pred, dim=1)

    label_flip_rate =  ((label_predicted_original!=label_predicted_ablation).sum()/label_predicted_original.shape[0]).item()

    tvd = 0.5 * torch.sum(torch.abs(probs_pred - probs_pred_ablation), dim=1).mean().item()
    print(f'Desactivation of {len(ablation_features)} feature(s)  associated to the class {class_int}, it results in the following effects : \n')
    print(f'TVD : {tvd} ; Absolute Accuracy change : {variation_absolute_accuracy} ; Label-flip rate : {label_flip_rate}\n')

    return variation_absolute_accuracy, tvd, label_flip_rate, dict_metrics_ablation




