import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
import numpy as np
from functools import partial
import torch.nn.functional as F
from loguru import logger
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import ctypes, ctypes.util
import random
import os
import re
import gc
from pathlib import Path
from datasets import Dataset

class ActivationDataset(Dataset):
    def __init__(self):
        self.data_block = []

    def append(self, *args):
        if len(args) not in [5, 6]:
            raise ValueError(f"Expected 5 or 6 arguments, but got {len(args)}")

        clean = tuple(a.cpu().detach() for a in args)
        self.data_block.append(clean)


    def __len__(self):
        return len(self.data_block)

    def __getitem__(self, idx):
    
    
        # Handle torch.Tensor indices
        if isinstance(idx, torch.Tensor):
            if idx.numel() != 1:  # Ensure it's a single element
                raise ValueError(f"Index tensor must have a single element, got {idx.numel()}")
            idx = idx.item()  # Convert single-element Tensor to an integer
    
        # Handle list indices
        elif isinstance(idx, list):
            if len(idx) != 1:  # Ensure it's a single-element list
                raise ValueError(f"Index list must have a single element, got {len(idx)}")
            idx = idx[0]
    
        # Ensure idx is now an integer
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer after processing, got {type(idx)}")
    
        # Access the block and return as a dictionary
        item = self.data_block[idx]
        result = {"input_ids": item[0], "cache": item[1], "output": item[2], "label": item[3], "attention_mask": item[4]}
        if len(item) == 6:
            result["activations_reconstruct"] = item[5]
        return result

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
        full_sae_name = f"{sae_name}_d_sae_{d_sae}_activation_rate_{feature_activation_rate}_supervised_classification"
        sae_load_dir = os.path.join(sae_checkpoint, full_sae_name)
    else:
        full_sae_name = f"{sae_name}_d_sae_{d_sae}_activation_rate_{feature_activation_rate}"
        sae_load_dir = os.path.join(sae_checkpoint, full_sae_name)


    # The different versions of the trained sae are saved in folders named 'final_X'
    if not (os.path.exists(sae_load_dir) and os.path.isdir(sae_load_dir)):
        raise ValueError(f"The sae name provided is not present in {sae_checkpoint}")
    
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

    return sae_path, full_sae_name


# In evaluation inference, in the forward pass we ablate every features which are not in z_class, to ensure an upper bound on the number of investigated concepts imposed by the user. 
def forward_on_selected_features(feature_acts,selected_features):

    if isinstance(selected_features,list):
        selected_features_tensor = torch.tensor(selected_features.copy())
    else:
        selected_features_tensor = selected_features
    selected_features_tensor = selected_features_tensor.to(feature_acts.device)

    feature_acts_mask = torch.zeros_like(feature_acts,dtype=torch.bool)
    if len(feature_acts_mask.shape)==2:
        feature_acts_mask[:,selected_features_tensor] = True
    else:
        feature_acts_mask[:,:,selected_features_tensor] = True
    feature_acts_masked = feature_acts * feature_acts_mask

    return feature_acts_masked


# To avoid leakage of information across the latest layers, when we reconstruct the classification hidden state, we ablate all the other hidden states of the previous tokens
def reconstr_hook_classification_token_single_element(activation, hook, replacement):
    n,m,d = activation.shape
    return replacement


# Used to compute the Recovery Accuracy
def count_same_match(
    logits_original : torch.Tensor,
    logits_reconstruction : torch.Tensor
):
 
    y_pred_original = logits_original.argmax(dim=1)
    y_pred_reconstruction = logits_reconstruction.argmax(dim=1)

    return (y_pred_original==y_pred_reconstruction).sum()


# Retrive text from tokens ids
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
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.malloc_trim(0)

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


# Cache LLM classifier activations on the test dataset to speed up the subsequent multiple inferences for the computation of the causality metrics
def cache_activations_with_labels( hook_model,
                                    dataset, #expected to be tokenized
                                    data_collator,
                                    tokenizer,
                                    hook_name,
                                    labels_tokens_id,
                                    eos,
                                    path_directory):

    FLUSH_EVERY_BATCH = 100

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator,shuffle=False, num_workers=0)
    
    # Save the different activations to speed the causality calculations
    activations_dataset = ActivationDataset()

    libc = ctypes.CDLL(ctypes.util.find_library('c')).malloc_trim

    # Filter predictions based on the logits corresponding to an accepted answer
    ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                            key=lambda kv: kv[1])]

    # Evaluation loop
    hook_model.eval()
    with torch.no_grad():
       
        step = 0
        for batch in tqdm(dataloader, desc=f"Caching of the hook {hook_name} activations", unit="batch"):
            step += 1
            input_ids = batch['input_ids'].to(hook_model.cfg.device)
            labels = batch['true_label']
            attention_mask = batch['attention_mask'].to(dtype=int).to(hook_model.cfg.device)
           

            #Manually fixing the issue of sentence longer than the context windows size since it is not automatically handled by transformer-lens and it causes conflict with the positional embedding that it dones on a vector of size the context attention windows
            if input_ids.shape[1] > hook_model.cfg.n_ctx:
                attention_mask = attention_mask[:,-hook_model.cfg.n_ctx:]
                input_ids = input_ids[:,-hook_model.cfg.n_ctx:] #It should not be an issue since we want to predict for almost the last token, so it looses almost nothing of the context it could have seen otherwise

            outputs, cache = hook_model.run_with_cache(input_ids,
                                                attention_mask=attention_mask,
                                                names_filter=hook_name,
                                                prepend_bos=False)
            
            outputs_logits = outputs[:,(-2-int(eos))].contiguous().view(-1, outputs.shape[-1])
            logits_original = outputs_logits[:,ordered_old_idxs]  #shape : (bs,nb_classes)
            cache = cache[hook_name][:,(-2-int(eos)),:]
            
            activations_dataset.append(
                input_ids.cpu(),
                cache.cpu(),     
                logits_original.cpu(),
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


def compute_loss_last_token_classif(
    true_labels: torch.Tensor,
    outputs_logits: torch.Tensor,
    reduction: str = "mean"
):
  """Computes the loss that focuses on the classification."""

  loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)

  return loss_ce(outputs_logits,true_labels)


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



def cosine_similarity_concepts(concepts_matrix,threshold):

    #concept_matrix shape : (embedding_dim, n_concepts)
    n,d = concepts_matrix.shape 
    
    # Normalize the vectors along dimension 0 (for cosine similarity calculation)
    column_norms = concepts_matrix.norm(dim=0, keepdim=True)
    column_norms[column_norms == 0] = 1e-8
    normalized_tensor = concepts_matrix / column_norms

    
    # Compute cosine similarity matrix (d x d)
    cosine_similarity_matrix = torch.mm(normalized_tensor.t(), normalized_tensor)
    # print(f"cosine_similarity_matrix shape : {cosine_similarity_matrix.shape}")
    
    # Extract unique pairs by masking the upper triangular matrix (excluding diagonal)
    i, j = torch.triu_indices(d, d, offset=1)
    cosine_values = cosine_similarity_matrix[i, j]
    # print(f"cosine_values shape : {cosine_values.shape}")

    count = torch.sum(cosine_values > threshold).item()

    print(f"There are {count} pairs of concepts with a cosine similarity higher than {threshold}")


    
def knn_distance_metric(embeddings: torch.Tensor) -> float:
    """
    Computes a local and global diversity metric for the embeddings distribution.
    The distance is computed as 1 minus the cosine similarity
    Local diversity metric : Average 1‑nearest‑neighbour distance
    Global diversity metric : Mean pairwise distance

    Args:
        embeddings (torch.Tensor): Tensor of shape (n_concepts, d) where n_concepts is the number of concepts and d is the dimension.

    Returns:
        float: Mean k-NN distance across all embeddings.
    """
    # Normalize embeddings to compute cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # (N, d)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

    # Convert similarity to distance (cosine distance = 1 - cosine similarity)
    distance_matrix = 1 - similarity_matrix  # (N, N)

    # Mask self-similarity (distance to self should be ignored)
    N = embeddings.shape[0]
    distance_matrix.fill_diagonal_(float('inf'))

    # Local diversity metric 
    mins, _ = distance_matrix.min(dim=1)
    ann = mins.mean().item()

    # Global diversity metric
    upper = torch.triu(distance_matrix, diagonal=1) 
    mpd = (upper.sum() * 2.0 / (N*(N-1))).item()

    return ann, mpd


# Run full inference of the LLM classifier on the test sentence split when plugged with the investigated concepts
# We measure the recovery accuracy of the model's predictions when the original activations goes through the concepts bottleneck layer
def eval_hook_loss(
    hook_model:HookedTransformer,
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
    
    activations_dataloader = DataLoader(activations_dataset,batch_size=1,shuffle=False)
    # Each sample of 'activations_dataloader' contains activations of multiple sentences
    nb_sentences_per_sample = next(iter(activations_dataloader))["input_ids"].shape[0]
    nb_sentences = len(activations_dataset)*nb_sentences_per_sample

    # Filter predictions based on the logits corresponding to an accepted answer
    ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                            key=lambda kv: kv[1])]

    # Evaluation loop
    hook_model.eval()
    if method_name in ["sae","concept_shap"]:
        interp_model.eval()
    with torch.no_grad():
        
        feature_activations_list = []
        original_activation_list = []
        true_labels_list = []
        logits_original_list = []
        logits_reconstruction_list = []
        
        
        for batch in tqdm(activations_dataloader, desc="Forward Passes of the given LLM model", unit="batch"):

            cache = batch["cache"].to(device)
            logits_original = batch["output"].to(hook_model.cfg.device) # shape : [bs, nb classes]
            true_labels = batch["label"].to(hook_model.cfg.device) # Ground truth labels
            
            a,c = cache.shape
            cache_sentence = cache.unsqueeze(1)

            if method_name=='concept_shap':
    
                #concept_score_thres_prob
                feature_acts = interp_model.concepts_activations(cache)
                feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                reconstruct_act = interp_model.reconstruct_from_concepts_activations(feature_acts_masked)
                reconstruct_act = reconstruct_act.unsqueeze(1).to(hook_model.cfg.device)
    
            elif method_name=='sae':

                # Use the SAE
                feature_acts, acts_without_process = interp_model.encode_with_hidden_pre(cache_sentence)
                feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                reconstruct_act = interp_model.decode(feature_acts_masked).to(hook_model.cfg.device)
                feature_acts = feature_acts.squeeze(1)
                feature_acts_masked = feature_acts_masked.squeeze(1) 

            elif method_name=='ica':
                embeddings_sentences_numpy = cache.cpu().numpy()
                #ica_activations
                feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy))
                feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                
                ica_mean = torch.from_numpy(interp_model.mean_)
                if isinstance(mixing_matrix, np.ndarray):
                    mixing_matrix = torch.from_numpy(mixing_matrix)
                reconstruct_act = feature_acts_masked @ mixing_matrix.T + ica_mean
                reconstruct_act = reconstruct_act.unsqueeze(1).to(hook_model.cfg.device)


            # In addition to the SAE metrics, we want to store feature activations and model predictions
            feature_activations_list.append(feature_acts_masked.cpu())
            original_activation_list.append(cache.cpu())
            true_labels_list.append(true_labels.cpu())

   
            # Run the inference by enforcing the forward pass to go through the concepts bottleneck layer. 
            # To avoid leakage of information across the latest layers, when we reconstruct the classification hidden state from the concepts activations, we ablate all the other hidden states of the previous tokens
            logits_reconstruction = hook_model.run_with_hooks(
                cache_sentence,
                start_at_layer=hook_layer,
                fwd_hooks=[
                    (
                        hook_name,
                        partial(reconstr_hook_classification_token_single_element, replacement=reconstruct_act),
                    ) ],
                return_type="logits",
            )
        
            logits_reconstruction = logits_reconstruction.contiguous().view(-1, logits_reconstruction.shape[-1])

            # Filter predictions based on the logits corresponding to an accepted answer
            logits_reconstruction = logits_reconstruction[:,ordered_old_idxs] #shape : (bs,nb_classes)
            logits_original_list.append(logits_original.cpu())
            logits_reconstruction_list.append(logits_reconstruction.cpu())

            # Compute the original classification cross-entropy loss and the same loss obtained by only using the reconstructed hidden state of the token preceding the class' answer.
            original_loss = compute_loss_last_token_classif(true_labels,logits_original)
            original_total_loss  += original_loss
            reconstruction_loss = compute_loss_last_token_classif(true_labels,logits_reconstruction)
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
        
        accuracy_original = total_matches_original / (nb_sentences)
        accuracy_reconstruction = total_matches_reconstruction / (nb_sentences)
        recovery_accuracy = total_same_predictions / (nb_sentences)
        reconstruction_mean_loss = reconstruction_total_loss / len(activations_dataloader)
        original_mean_loss = original_total_loss / len(activations_dataloader)

        print(f"\n Method name : {method_name}")
        print(f"\nOriginal classification crossentropy mean loss : {original_mean_loss:.3f} ({total_matches_original} matches) - Reconstruction classification crossentropy mean loss : {reconstruction_mean_loss:.3f} (Computed over {nb_sentences} sentences) ")
        print(f'\nRecovery accuracy : {recovery_accuracy:.3f}')
        print(f"\nOriginal accuracy of the model : {accuracy_original:.3f} - Accuracy of the model when plugging the reconstruction hidden states : {accuracy_reconstruction:.3f}")

        # One hidden state for each sentence (hidden state of the token preceding the class-generating token). 
        feature_activations_tensor = torch.cat(feature_activations_list,dim=0)
        original_activation_tensor = torch.cat(original_activation_list,dim=0)

        logits_reconstruction_tensor = torch.cat(logits_reconstruction_list,dim=0)
        logits_original_tensor = torch.cat(logits_original_list,dim=0)

        mean_activations = feature_activations_tensor.mean(dim=0)

        max_activity = 0
        mean_activity = 0
        n = feature_activations_tensor.shape[0]
        # Compute statistics on the distribution of activations of the features
        for feature_number in selected_features:
            active_samples = torch.sum(feature_activations_tensor[:,feature_number].abs() > 1e-8)
            percentage_activity = (active_samples / n) * 100
            mean_activity += percentage_activity
            if percentage_activity > max_activity:
                max_activity = percentage_activity
        mean_activity /= len(selected_features)


        print(f"\nAveraged frequency of activation of the selected features across the test dataset : {mean_activity:.3f}% - Highest frequency of activation : {max_activity:.3f}%")

        performance = {'Original Mean Loss':original_mean_loss.item(), 'Reconstruction Mean loss': reconstruction_mean_loss.item(), 'Number sentences':n, 'Recovery accuracy':recovery_accuracy, 'Original accuracy':accuracy_original, 'Reconstruction accuracy' : accuracy_reconstruction, 'Averaged activation frequency' : mean_activity.item(), 'Highest activation frequency' : max_activity.item()}
        # We merge the macro information of the concepts with the dictionary of the variation metrics
        performance = performance |  dict_metrics_original | dict_metrics_reconstruction
        
        # Return the ground truth labels of the prompts
        true_labels_tensor = torch.cat(true_labels_list,dim=0)
        
        return performance , mean_activations, {'concepts activations' : feature_activations_tensor, 'original activations' : original_activation_tensor, 'true labels' : true_labels_tensor, 'original logits' : logits_original_tensor ,'logits from reconstruction' : logits_reconstruction_tensor}
        

def design_figure(W_dec_pca, sizes, np_prototypes_pca, colors,
                  feature_colors, normalized_class_scores,
                  N, labels_names):
    """
    Same behaviour as before, but the pie radius is
    radius = base_r * sizes / sizes.max()
    so tiny classes can vanish if they are tiny enough.
    """

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
    print(f"original_activations shape : {original_activations.shape}")
    print(f"sae_activations shape : {sae_activations.shape}")
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
            feature_score_class[label] = np.sum(sae_activations[indices],axis=0)

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
    # Compute colors for all points
    point_colors_indices = np.argmax(normalized_class_scores,axis=0) 
    colors = [feature_colors[idx] for idx in point_colors_indices]

    scaler = StandardScaler()
    activations_scaled = scaler.fit_transform(original_activations_norm)    
    pca = PCA(n_components=2)  
    pca.fit(activations_scaled)

    W_dec_features_scaled = scaler.transform(W_dec_numpy)
    W_dec_pca = pca.transform(W_dec_features_scaled)

    #Display the hidden representation prototype for each class
    np_prototypes = np.array(list(hidden_size_prototype_class.values()))  # Convert dictionary values to a NumPy array
    np_prototypes_scaled = scaler.transform(np_prototypes)
    np_prototypes_pca = pca.transform(np_prototypes_scaled)
    sizes = mean_activations.numpy()

    ax, fig = design_figure(W_dec_pca, sizes, np_prototypes_pca, colors, feature_colors, normalized_class_scores,N,label_names)  
    
    return fig, normalized_class_scores
    
    

def analyze_features(hook_model,tune_sae,mean_activations,dict_analysis_features,label_names):
    
    W_dec_numpy = tune_sae.W_dec.detach().cpu().numpy()
    fig_pca, _ = pca_activations_projection(W_dec_numpy,mean_activations,dict_analysis_features,label_names)
    
    return fig_pca
    
'''
Retrieve the logits predicted by the model on a subset of sentences `ids_samples` from the tested dataset.
In addition to forcing the forward pass through the concepts-layer bottleneck, we also apply targeted ablations to the features of z_class to compute our causality metrics (with `perturb` a tensor of integer values containing the indices of features in z_class to ablate)
'''
def get_predictions(
    hook_model:HookedTransformer,
    interp_model,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    perturb,
    device,
    labels_tokens_id,
    ids_samples=torch.tensor([]),
    mixing_matrix=None,
):
    # Filter predictions based on the logits corresponding to an accepted answer
    ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                            key=lambda kv: kv[1])]


    dict_metrics_reconstruction = {}
    #For accuracy 
    number_matches = 0

    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    
    # Number of sentences for which the prediction is evaluated
    nb_evaluated_sentences = len(ids_samples)

    logits_reconstruction_list = []

    # Evaluation loop
    hook_model.eval()
    if method_name in ["sae","concept_shap"]:
        interp_model.eval()

    #If we do not want to evaluate on all samples in the activation_dataset
    number_sample = 0
    
    with torch.no_grad():
       
        for num_batch, batch in tqdm(enumerate(activations_dataloader), desc=f"Forward Passes with ablation on {0 if perturb is None else len(perturb)} feature(s) ", unit="batch"):
                
                true_labels = batch["label"] # Ground truth labels
                cache = batch["cache"].to(device)
              
                samples_to_keep = (ids_samples - number_sample)
            
                evaluated_samples  = samples_to_keep[(0 <= samples_to_keep) & (samples_to_keep < true_labels.shape[0])]
                cache = cache[evaluated_samples,:]
                
                cache_sentence = cache.unsqueeze(1)
    
                if method_name=='concept_shap':
        
                    #concept_score_thres_prob
                    feature_acts = interp_model.concepts_activations(cache)
                    feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
        
                elif method_name=='sae':
    
                    # Use the SAE
                    feature_acts, acts_without_process = interp_model.encode_with_hidden_pre(cache_sentence)
                    feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                    feature_acts_masked = feature_acts_masked.squeeze(1)
            
    
                elif method_name=='ica':
                    embeddings_sentences_numpy = cache.cpu().numpy()
                    #ica_activations
                    feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy))
                    feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                    

                if perturb is not None:
                    feature_acts_masked[:,perturb] = 0
                    
                
                if method_name=='concept_shap':
                    reconstruct_act = interp_model.reconstruct_from_concepts_activations(feature_acts_masked)
                    reconstruct_act = reconstruct_act.unsqueeze(1)
    
                elif method_name=='sae':
    
                    feature_acts_masked = feature_acts_masked.unsqueeze(1)
                    reconstruct_act = interp_model.decode(feature_acts_masked)
    
                elif method_name=='ica':    
                   
                    ica_mean = torch.from_numpy(interp_model.mean_)
                    if isinstance(mixing_matrix, np.ndarray):
                        mixing_matrix = torch.from_numpy(mixing_matrix)
                    reconstruct_act = feature_acts_masked @ mixing_matrix.T + ica_mean
                    reconstruct_act = (reconstruct_act.unsqueeze(1)).to(hook_model.cfg.device)
            
                logits_reconstruction = hook_model.run_with_hooks(
                    cache_sentence,
                    start_at_layer=hook_layer,
                    fwd_hooks=[
                        (
                            hook_name,
                            partial(reconstr_hook_classification_token_single_element, replacement=reconstruct_act),
                        ) ],
                    return_type="logits",
                )

                logits_reconstruction = logits_reconstruction.contiguous().view(-1, logits_reconstruction.shape[-1])

                # Filter predictions based on the logits corresponding to an accepted answer
                logits_reconstruction = logits_reconstruction[:,ordered_old_idxs].cpu() #shape : (evaluated samples,nb_classes)            
                logits_reconstruction_list.append(logits_reconstruction)

                number_matches += update_metrics(true_labels[evaluated_samples],logits_reconstruction,dict_metrics_reconstruction)
                number_sample+=true_labels.shape[0]

    logits_reconstruction_tensor = torch.cat(logits_reconstruction_list,dim=0)
    predicted_probabilities = F.softmax(logits_reconstruction_tensor, dim=1)

    accuracy = (number_matches / nb_evaluated_sentences).item()
    
    
    return  predicted_probabilities , accuracy, dict_metrics_reconstruction 

    
# Run joint ablation on features from the same class-specific features segment F_c where the category c is provided in `class_int`. 
# The indices of features to ablate are contained in `ablation_features`.
# We compute the 3 global causality metrics in that context. The proportion of each class-specific features segment F_c ablated is varied throughout the experiment.
def eval_causal_effect_concepts(
    hook_model:HookedTransformer,
    interp_model,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    probs_pred, 
    accuracy,
    class_int,
    device,
    labels_tokens_id,
    ablation_features,
    mixing_matrix=None):

  
    # We look at the impact of ablating multiple features simultaneously
    probs_pred_ablation, accuracy_ablation, dict_metrics_ablation = get_predictions(hook_model,
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




