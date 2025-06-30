from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
import numpy as np
from functools import partial
import torch.nn.functional as F
from loguru import logger
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from scipy.special import softmax
import random

class ActivationDataset(Dataset):
    def __init__(self):
        self.data_block = []

    def append(self, *args):
        if len(args) not in [5, 6]:
            raise ValueError(f"Expected 5 or 6 arguments, but got {len(args)}")
        
        self.data_block.append(args)


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


def mutual_information_continuous(predicted_logits, activations_continuous):
    """
    Compute mutual information between continuous neuron activations and predicted labels.
    """

    predicted_logits_tensor = torch.stack(list(predicted_logits.values()), dim=1)

    activations_continuous = activations_continuous.cpu().numpy()
    predicted_logits_tensor = predicted_logits_tensor.cpu().numpy()
    predicted_labels = np.argmax(predicted_logits_tensor,axis=1)

    mi = mutual_info_regression(activations_continuous, predicted_labels)
    
    return mi.mean()
    

def reconstr_hook_classification_token_single_element(activation, hook, replacement):
    n,m,d = activation.shape
    return replacement


def count_same_match(
    original_logits : torch.Tensor,
    reconstruction_logits : torch.Tensor,
    is_eos: bool
):

    _,vocab_size = original_logits.size()
    
    y_pred_original = original_logits.argmax(dim=1)
    y_pred_reconstruction = reconstruction_logits.argmax(dim=1)

    return (y_pred_original==y_pred_reconstruction).sum()


def decode_dataset(tokenized_dataset,tokenizer):
    
    original_texts = []
    
    for input_ids in tokenized_dataset['input_ids']:
        # Decoding the token IDs back to text
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        original_texts.append(text)

    return original_texts

def cache_activations_with_labels( hook_model,
                                    dataset, #expected to be tokenized
                                    data_collator,
                                    tokenizer,
                                    hook_name,
                                    is_eos):

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)
    
    #Save the different activations to speed the causality calculations done after
    activations_dataset = ActivationDataset()

    # Evaluation loop
    hook_model.eval()
    with torch.no_grad():
       
        for batch in tqdm(dataloader, desc=f"Caching of the hook {hook_name} activations", unit="batch"):

            input_ids = batch['input_ids'].to(hook_model.cfg.device)
            labels = batch['token_labels']
            attention_mask = batch['attention_mask'].to(dtype=int).to(hook_model.cfg.device)
           

            #Manually fixing the issue of sentence longer than the context windows size since it is not automatically handled by transformer-lens and it causes conflict with the positional embedding that it dones on a vector of size the context attention windows
            if input_ids.shape[1] > hook_model.cfg.n_ctx:
                attention_mask = attention_mask[:,-hook_model.cfg.n_ctx:]
                input_ids = input_ids[:,-hook_model.cfg.n_ctx:] #It should not be an issue since we want to predict for almost the last token, so it looses almost nothing of the context it could have seen otherwise

            outputs, cache = hook_model.run_with_cache(input_ids,
                                                attention_mask=attention_mask,
                                                names_filter=hook_name,
                                                prepend_bos=False)
            #cache[hook_name] shape : (bs,max_lenght,dim)
            
            outputs_logits = outputs[:,(-2-int(is_eos))].contiguous().view(-1, outputs.shape[-1])
            #outputs_logits (bs,dim)
            
            #Save the activations and labels
            input_ids = input_ids.cpu()
            cache_to_save = cache[hook_name][:,(-2-int(is_eos)),:].cpu()
            outputs_logits = outputs_logits.cpu()  #shape : [batch size, vocab size]
            attention_mask = attention_mask.cpu()
            activations_dataset.append(input_ids, cache_to_save, outputs_logits,labels,attention_mask)
        

    #Return the decoded sentences used to generate the activations
    original_text_used = decode_dataset(dataset, tokenizer)

    return activations_dataset, original_text_used


def compute_loss_last_token_classif(
    inputs_labels: torch.Tensor,
    outputs_logits: torch.Tensor,
    labels_tokens_id : dict,
    is_eos: bool,
    vocab_size: int,
    reduction: str = "mean"
):
  """Computes the loss that focuses on the classification. Given that we only care about the classification, we sum all the logits of tokens that are not associated to any class and 
  create another class : Other/Unknown"""

  #outputs_logits of size [8,M,50304]
  #8 : batch size
  #M : max length sequence in that batch
  # 50304 : vocabulary size

  labels_to_pred = inputs_labels[:,(-1-int(is_eos))].view(-1)

  loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)

  #We do a clone otherwise, the change that we impose on the labels is also propagated to the original input
  labels_to_pred_clone = labels_to_pred.clone()
    
  prediction_vector = torch.zeros((outputs_logits.shape[0],len(labels_tokens_id)+1)).to(outputs_logits.device)
  probs = F.softmax(outputs_logits,dim=1)
  prob_alternative = 1
  for i , (key,value) in enumerate(labels_tokens_id.items()):

      #Predictions
      prediction_vector[:,i] = probs[:,value]
      prob_alternative -= prediction_vector[:,i]

      #Rename tokens to their labels
      labels_to_pred_clone[labels_to_pred_clone==value] = i
    
  prediction_vector[:,-1] = prob_alternative
  # print(f"prediction_vector : {prediction_vector}")
  # print(f"labels_to_pred_clone : {labels_to_pred_clone}")

  return loss_ce(prediction_vector,labels_to_pred_clone)


def update_metrics(
    labels : torch.Tensor,
    logits : torch.Tensor,
    labels_tokens_id : dict,
    dict_metrics : dict,
    is_eos : bool
):
    _,vocab_size = logits.size()

    
    logits_prediction = logits.argmax(dim=1)
    labels_to_pred = labels[:,(-1-int(is_eos))].view(-1)
    

    exact_matches = (logits_prediction==labels_to_pred)
    # print(f"labels_to_pred : {labels_to_pred}")
    # print(f"logits_prediction : {logits_prediction}")
    count_exact_matches = exact_matches.sum() #tensor

    
    for key,value in labels_tokens_id.items():
      #In case the keys do not already exist (typically the first time we call this function)
      dict_metrics.setdefault(f'number real samples_{key}',0)
      dict_metrics.setdefault(f'true matches_{key}',0)
      dict_metrics.setdefault(f'number predicted samples_{key}',0)
      
      position_key = (labels_to_pred==value)
      number_samples_key = (labels_to_pred==value).sum().item()
      dict_metrics[f'number real samples_{key}'] += number_samples_key

      exact_matches_key = position_key & exact_matches
      count_exact_matches_key = exact_matches_key.sum().item()
      dict_metrics[f'true matches_{key}']+= count_exact_matches_key

      count_predicted_key = (logits_prediction==value).sum().item()
      dict_metrics[f'number predicted samples_{key}'] += count_predicted_key
      
    for key in labels_tokens_id.keys():
      dict_metrics[f'recall_{key}'] = 0 if dict_metrics[f'number real samples_{key}']==0 else dict_metrics[f'true matches_{key}'] / dict_metrics[f'number real samples_{key}']
      dict_metrics[f'precision_{key}'] = 0 if  dict_metrics[f'number predicted samples_{key}']==0  else dict_metrics[f'true matches_{key}'] / dict_metrics[f'number predicted samples_{key}']
      dict_metrics[f'f1-score_{key}'] = 0 if (dict_metrics[f'recall_{key}'] + dict_metrics[f'precision_{key}'])==0  else 2 * dict_metrics[f'recall_{key}'] * dict_metrics[f'precision_{key}'] / (dict_metrics[f'recall_{key}'] + dict_metrics[f'precision_{key}'])

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

    
    
    # Compute statistics
    mean_cosine_similarity = cosine_values.mean().item()
    variance_cosine_similarity = cosine_values.var().item()

    count = torch.sum(cosine_values > threshold).item()

    print(f"There are {count} pairs of concepts with a cosine similarity higher than {threshold}")


def filter_highly_similar_columns(concepts_matrix, threshold=0.95, seed=42):
    """
    Remove columns from 'concepts_matrix' until no pair of columns has a cosine similarity above 'threshold'.
    Each time a pair above threshold is found, randomly remove one column from that pair.
    
    Args:
        concepts_matrix (torch.Tensor): Shape (embedding_dim, n_concepts).
        threshold (float): Cosine similarity threshold.
        seed (int): Random seed for reproducibility.
    
    Returns:
        List[int]: Sorted list of column indices that are kept.
    """
    # Set random seed for reproducibility (optional)
    random.seed(seed)
    
    # -- 1) Normalize columns to compute cosine similarity --
    # Shape: (embedding_dim, n_concepts)
    # We want to compute similarities among columns => each column is a concept vector
    column_norms = concepts_matrix.norm(dim=0, keepdim=True)
    column_norms[column_norms == 0] = 1e-8  # Avoid division by zero
    normalized_tensor = concepts_matrix / column_norms  # Still shape: (embedding_dim, n_concepts)

    # Keep track of which columns remain
    all_columns = list(range(concepts_matrix.shape[1]))  # [0, 1, 2, ..., n_concepts-1]

    
    while True:
        

        # -- 2) Build the normalized subset for currently kept columns --
        subset = normalized_tensor[:, all_columns]  # Shape: (embedding_dim, len(all_columns))

        # If there's 0 or 1 column left, no pairs to check
        if subset.shape[1] < 2:
            break
        
        # -- 3) Compute pairwise cosine similarity among the kept columns --
        # similarity_matrix: shape (len(all_columns), len(all_columns))
        similarity_matrix = torch.mm(subset.t(), subset)

        # -- 4) Identify pairs above the threshold --
        # We'll look only at the upper triangular part (excluding diagonal)
        row_indices, col_indices = torch.triu_indices(
            similarity_matrix.shape[0], 
            similarity_matrix.shape[1],
            offset=1
        )
        sim_values = similarity_matrix[row_indices, col_indices]

        # Find which pairs exceed the threshold
        high_sim_indices = (sim_values > threshold).nonzero(as_tuple=False).view(-1)
        
        if high_sim_indices.numel() == 0:
            # No pairs above threshold => done
            break
        
        # -- 5) Randomly remove 1 column from one high-similarity pair --
        # Pick one such pair
        idx = int(random.choice(high_sim_indices))
        
        # row_indices[idx], col_indices[idx] are positions in 'subset'
        # We need to map them back to the actual column indices from 'all_columns'
        pair_first = row_indices[idx].item()
        pair_second = col_indices[idx].item()

        # Map these subset positions to original columns
        col_first = all_columns[pair_first]
        col_second = all_columns[pair_second]

        # Randomly remove one of them
        if random.random() < 0.5:
            col_to_remove = col_first
        else:
            col_to_remove = col_second
        
        all_columns.remove(col_to_remove)

    # Sort the remaining columns for consistency
    all_columns.sort()

    # -- 6) Return the list of kept columns --
    return all_columns

    
def knn_distance_metric(embeddings: torch.Tensor, k: int = 5) -> float:
    """
    Computes the mean distance to the k-nearest neighbors (k-NN) for each embedding.
    The distance is computed as 1 minus the cosine similarity

    Args:
        embeddings (torch.Tensor): Tensor of shape (n_concepts, d) where n_concepts is the number of concepts and d is the dimension.
        k (int): Number of nearest neighbors to consider.

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

    # Get the k nearest neighbor distances for each sample
    knn_distances, _ = torch.topk(distance_matrix, k, largest=False, dim=1)  # (N, k)
    print(f"knn_distances shape : {knn_distances.shape}")

    # Compute the mean k-NN distance across all points
    mean_knn_distance = knn_distances.mean().item()

    return mean_knn_distance


def eval_hook_loss(
    hook_model:HookedTransformer,
    interp_model,
    labels_dataset,
    original_text_used,
    tokenizer,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    is_eos=True,
    path_to_dr_methods=''):
    
    device = hook_model.cfg.device
    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(np.array(labels_dataset))
    keys_labels = set(unique_labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}
    

    dict_metrics_original = {}
    dict_metrics_reconstruction = {}
    total_matches_original = 0
    total_matches_reconstruction = 0
    total_same_predictions = 0
    original_total_loss = 0
    reconstruction_total_loss = 0
    
    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    bs = next(iter(activations_dataloader))["input_ids"].shape[0]
    
    #Save the different activations to speed the causality calculations done after
    new_activations_dataset = ActivationDataset()

    # Evaluation loop
    hook_model.eval()
    if method_name in ["sae","concept_shap"]:
        interp_model.eval()
    with torch.no_grad():
        
        feature_activations_list = []
        original_activation_list = []
        prompt_labels_list = []
        model_logits_labels = {}
        
        
        for batch in tqdm(activations_dataloader, desc="Forward Passes of the given LLM model", unit="batch"):

            input_ids = batch['input_ids'].to(device)
            cache = batch["cache"].to(device)
            original_output = batch["output"].to(device)
            labels = batch["label"]
            attention_mask = batch['attention_mask'].to(dtype=int).to(device)
            
            

            # if prompt_tuning:
            #     batch_size = input_ids.size(0)
            #     prompt_token_ids = torch.full(
            #         (batch_size, hook_model.num_prompt_tokens),
            #         hook_model.tokenizer.eos_token_id,  # Using EOS token as a placeholder
            #         dtype=torch.long,
            #         device=input_ids.device,
            #     )
            #     input_ids = torch.cat([prompt_token_ids, input_ids], dim=1)
            #     prompt_attention_mask = torch.full(
            #         (batch_size, hook_model.num_prompt_tokens),
            #         1,  # Using EOS token as a placeholder
            #         dtype=torch.long,
            #         device=input_ids.device,
            #     )
            #     attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)            
            
            a,c = cache.shape
            #cache_flatten = cache.view(a*b,-1).unsqueeze(1)
            cache_sentence = cache.unsqueeze(1)

            if method_name=='concept_shap':
    
                #concept_score_thres_prob
                feature_acts = interp_model.concepts_activations(cache)
                feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                reconstruct_act = interp_model.reconstruct_from_concepts_activations(feature_acts_masked)
                reconstruct_act = reconstruct_act.unsqueeze(1)
    
            elif method_name=='sae':

                # Use the SAE
                feature_acts, acts_without_process = interp_model.encode_with_hidden_pre(cache_sentence)
                feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                reconstruct_act = interp_model.decode(feature_acts_masked)
                feature_acts = feature_acts.squeeze(1)
                feature_acts_masked = feature_acts_masked.squeeze(1) 

            elif method_name=='ica':
                embeddings_sentences_numpy = cache.cpu().numpy()
                #ica_activations
                feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)).to(hook_model.cfg.device)
                feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                
                try : 
                    mixing_matrix = np.load(f'{path_to_dr_methods}.npy')
                except FileNotFoundError:
                    raise FileNotFoundError(f"Error: '{path_to_dr_methods}.npy' not found. The mixing matrix for ICA has not been saved.")
                except Exception as e:
                    raise RuntimeError(f"An unexpected error occurred while loading the mixing matrix for ICA: {str(e)}")

                dr_mean = torch.from_numpy(interp_model.mean_).to(hook_model.cfg.device)
                mixing_matrix = torch.tensor(mixing_matrix).to(hook_model.cfg.device)
                reconstruct_act = feature_acts_masked @ mixing_matrix.T + dr_mean
                reconstruct_act = reconstruct_act.unsqueeze(1)

    
            #Save the activations and labels for the causality calculations done after
            bs = input_ids.shape[0]
            d_in = cache.shape[-1]
            inputs_to_save = input_ids.cpu()
            cache_to_save = cache_sentence.squeeze(1).cpu()
            original_output_to_save = original_output.cpu()  #shape : [batch size, vocab size]
            labels_to_save = labels
            attention_mask_to_save = attention_mask.cpu()
            interp_model_activations_to_save = feature_acts_masked.cpu()
            new_activations_dataset.append(inputs_to_save, cache_to_save, original_output_to_save, labels_to_save, attention_mask_to_save, interp_model_activations_to_save)
            

            #In addition to the SAE metrics, we want to store feature activations and model predictions
            feature_activations_list.append(feature_acts_masked.cpu())
            original_activation_list.append(cache.cpu())
            prompt_labels_list.append(labels)
               
                
            #In order to be aware of the corresponding predictions of the model when we work on the sae activations
            #Only select the logits of the tokens id corresponding to the labels (for memory efficiency)
            for key, value in labels_tokens_id.items():
                token_id = value
                logits_tensor =  original_output[:,token_id]
                if key in model_logits_labels:
                    model_logits_labels[key] =  torch.cat((model_logits_labels[key],logits_tensor))
                else:
                    model_logits_labels[key] = logits_tensor               


            
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

            #We compute the original classification cross-entropy loss and the same loss obtained by plugging the reconstructed activations from the SAE features at hook_name
            original_loss = compute_loss_last_token_classif(input_ids,original_output,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[-1])
            original_total_loss  += original_loss
            reconstruction_loss = compute_loss_last_token_classif(input_ids,logits_reconstruction,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[-1])
            reconstruction_total_loss += reconstruction_loss

            
            #We compute the variation in true accuracy
            acc_original = update_metrics(input_ids,original_output,labels_tokens_id,dict_metrics_original,is_eos)
            acc_reconstruction = update_metrics(input_ids,logits_reconstruction,labels_tokens_id,dict_metrics_reconstruction,is_eos)
            total_matches_original += acc_original.item()
            total_matches_reconstruction += acc_reconstruction.item()
            #We compute the recovering accuracy metric
            count_same_predictions = count_same_match(original_output,logits_reconstruction,is_eos)
            total_same_predictions += count_same_predictions.item()
            
        del cache
        
        accuracy_original = total_matches_original / (len(activations_dataset)*bs)
        accuracy_reconstruction = total_matches_reconstruction / (len(activations_dataset)*bs)
        recovering_accuracy = total_same_predictions / (len(activations_dataset)*bs)
        reconstruction_mean_loss = reconstruction_total_loss / len(activations_dataloader)
        original_mean_loss = original_total_loss / len(activations_dataloader)

        print(f"\n Method name : {method_name}")
        print(f"\nOriginal classification crossentropy mean loss : {original_mean_loss} ({total_matches_original} matches) - Reconstruction classification crossentropy mean loss : {reconstruction_mean_loss} (Computed over {len(labels_dataset)} sentences) ")
        print(f'\nRecovering accuracy : {recovering_accuracy}')
        print(f"\nOriginal accuracy of the model : {accuracy_original} - Accuracy of the model when plugging the reconstruction hidden states : {accuracy_reconstruction}")

        #The number of hidden states is the same for each prompt, equals to 1 as we keep only the last token. So we can concatenate as the last two dimensions are the same for each feature activation vector.
        feature_activations_tensor = torch.cat(feature_activations_list,dim=0)
        original_activation_tensor = torch.cat(original_activation_list,dim=0)

        #Compute Mutual Information between predicted labels and features activations
        # mutual_information_activations = mutual_information_continuous(model_logits_labels, feature_activations_tensor)
        # print(f"mutual_information_activations : {mutual_information_activations}")

        mean_activations = feature_activations_tensor.mean(dim=0)

        max_activity = 0
        mean_activity = 0
        n = feature_activations_tensor.shape[0]
        #Compute statistics on the distribution of activations of the feature
        for feature_number in selected_features:
            active_samples = torch.sum(feature_activations_tensor[:,feature_number] > 0)
            percentage_activity = (active_samples / n) * 100
            mean_activity += percentage_activity
            if percentage_activity > max_activity:
                max_activity = percentage_activity
        mean_activity /= len(selected_features)


        print(f"\nAveraged frequency of activation of the selected features across the test dataset : {mean_activity}% - Highest frequency of activation : {max_activity}%")

        performance = {'Original Mean Loss':original_mean_loss.item(), 'Reconstruction Mean loss': reconstruction_mean_loss.item(), 'Number sentences':labels_dataset, 'Recovering accuracy':recovering_accuracy, 'Original accuracy':accuracy_original, 'Reconstruction accuracy' : accuracy_reconstruction, 'Averaged activation frequency' : mean_activity.item(), 'Highest activation frequency' : max_activity.item()}
        #We merge the macro information of the interpretable model with the dictionary of the variation metrics
        performance = performance |  dict_metrics_original | dict_metrics_reconstruction
        
        # #Remove the template part
        # for t,_ in enumerate(original_text_used):
        #     original_text_used[t] = original_text_used[t][len_example:-len_template] #Len specific to the AG News template, has to be adapted to do it automatically for other datasets
        
        #Return the ground truth labels of the prompts
        prompt_labels_tensor = torch.cat(prompt_labels_list,dim=0)
        
        return performance , new_activations_dataset, mean_activations, {'feature activation' : feature_activations_tensor, 'original activation' : original_activation_tensor, 'prompt label' : prompt_labels_tensor, 'model output logits' : model_logits_labels}
        

def design_figure(W_dec_pca, sizes, np_prototypes_pca, colors, feature_colors, normalized_class_scores,N,labels_names):

  feature_colors = np.array(['#636EFA','#EF553B','#00CC96','#3D2B1F'])
  #feature_colors = np.array(['#636EFA','#EF553B'])
    
  # Extracting the x and y components of the vectors
  x = W_dec_pca[:, 0]
  y = W_dec_pca[:, 1]


  # Create the figure and axis
  fig, ax = plt.subplots(figsize=(8, 8))

  labels_names = labels_names.values()

  # Plot the special points with yellow triangles
  ax.scatter(np_prototypes_pca[:,0], np_prototypes_pca[:,1], color='orange', marker='^', s=400, zorder=6)

  # Add placards with labels for special points
  for i, label in enumerate(labels_names):
      ax.text(np_prototypes_pca[:,0][i] - 4.5, np_prototypes_pca[:,1][i] + 0.1, label, fontsize=15,bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'),zorder=100)

  # Add grid, labels, and title
  ax.axhline(0, color='grey', lw=1,zorder=10)
  ax.axvline(0, color='grey', lw=1,zorder=10)
  ax.grid(True, linestyle='--', alpha=0.5,zorder=10)


  # Function to create a pie chart marker using Wedges at a specific location
  def draw_pie(ax, center, sizes, colors, radius=0.5, alpha=0.8):
     
      # Starting angle
      start_angle = 0

      # Iterate through sizes and corresponding colors to draw pie slices
      for size, color in zip(sizes, colors):
          # Calculate the end angle of the wedge
          end_angle = start_angle + size * 360
          # Create a wedge patch for each slice
          wedge = Wedge(
              center, radius, start_angle, end_angle,
              facecolor=color, alpha=alpha, zorder=2
          )
          # Add the wedge to the axes
          ax.add_patch(wedge)
          # Update the start angle for the next slice
          start_angle = end_angle


  for coord, prop,radius in zip(W_dec_pca[:,:2], normalized_class_scores.T,sizes):
      draw_pie(ax,coord, prop, feature_colors, radius=(0.5*radius+1e-8))


  # Set axis limits to give some padding around the vectors
  range_x = np.concatenate((x,np_prototypes_pca[:,0]))
  range_y = np.concatenate((y,np_prototypes_pca[:,1]))
    
  xlim_min = min(range_x) - 1
  xlim_max = max(range_x) + 1
  ylim_min = min(range_y) - 1
  ylim_max = max(range_y) + 1


  ax.set_xlim(xlim_min, xlim_max)
  ax.set_ylim(ylim_min, ylim_max)


  ax.set_xlabel('PCA component 1')
  ax.set_ylabel('PCA component 2')
  ax.set_aspect('equal', 'box')

  return ax, fig  #, (xlim_min, xlim_max, ylim_min,ylim_max)




def pca_activations_projection(W_dec_numpy,mean_activations,dict_analysis_features,label_names):

    #2D projection of the decoder features/rows on PCA axis computed with original activations 
    logger.info(f"Compute the PCA plan of the original activations and projection the features/decoder rows on it")
    '''PCA in the activation space'''
    N = W_dec_numpy.shape[0]
    
    labels = dict_analysis_features['prompt label'].numpy()
    original_activations = dict_analysis_features['original activation'].numpy()
    sae_activations = dict_analysis_features['feature activation'].numpy()
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

    color_labels = np.arange(unique_labels.size)
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
    #np_keys_prototypes = list(hidden_size_prototype_class.keys())  # Get dictionary keys for hover text
    np_prototypes_scaled = scaler.transform(np_prototypes)
    np_prototypes_pca = pca.transform(np_prototypes_scaled)
    sizes = mean_activations.numpy()

    ax, fig = design_figure(W_dec_pca, sizes, np_prototypes_pca, colors, feature_colors, normalized_class_scores,N,label_names)  
    
    # fig.savefig('./layer5_90000.pdf', dpi=300, bbox_inches='tight')

    return fig, normalized_class_scores
    
    

def analyze_features(hook_model,tune_sae,mean_activations,dict_analysis_features,label_names):
    
    W_dec_numpy = tune_sae.W_dec.detach().cpu().numpy()
    fig_pca, _ = pca_activations_projection(W_dec_numpy,mean_activations,dict_analysis_features,label_names)
    
    return fig_pca
    


def get_predictions(
    hook_model:HookedTransformer,
    interp_model,
    labels_dataset,
    tokenizer,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    is_eos=True,
    perturb=None,
    ids_samples=torch.tensor([]),
    path_to_dr_methods=''
):
    dict_metrics_reconstruction = {}
    device = hook_model.cfg.device
    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(np.array(labels_dataset))
    keys_labels = set(unique_labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}

    #This is where we store the prediction logits specific to classification. One for each category plus one that sums up all the logits associated to tokens which do not belong to the tokens of a class
    #classification_predicted_probs = torch.zeros((len(labels_dataset),len(unique_labels)+1)).cpu()
    classification_predicted_probs = []

    #For accuracy 
    number_matches = 0

    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    
    bs = next(iter(activations_dataloader))["input_ids"].shape[0]

    # Evaluation loop
    hook_model.eval()
    if method_name in ["sae","concept_shap"]:
        interp_model.eval()

    #In case we do not want to evaluate on all samples in the activation_dataset
    number_sample = 0
    
    with torch.no_grad():
       
        for num_batch, batch in tqdm(enumerate(activations_dataloader), desc=f"Forward Passes with ablation on {0 if perturb is None else len(perturb)} feature(s) ", unit="batch"):
                
                inputs = batch["input_ids"].to(device)
                cache = batch["cache"].to(device)
              
                #a,c = cache.shape
                #print(f"ids_samples : {ids_samples}")
                samples_to_keep = (ids_samples - number_sample)
                #print(f"samples_to_keep : {samples_to_keep}")
                #print(f"{(0 <= samples_to_keep) & (samples_to_keep < inputs.shape[0])}")
                evaluated_samples  = samples_to_keep[(0 <= samples_to_keep) & (samples_to_keep < inputs.shape[0])]
                cache = cache[evaluated_samples,:]
                
            
                #cache_flatten = cache.view(a*b,-1).unsqueeze(1)
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
                    feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)).to(hook_model.cfg.device)
                    feature_acts_masked = forward_on_selected_features(feature_acts,selected_features)
                    

                if perturb is not None:
                    #feature_acts_all[:,:,perturb] = 0
                    feature_acts_masked[:,perturb] = 0
                    
                
                if method_name=='concept_shap':
                    reconstruct_act = interp_model.reconstruct_from_concepts_activations(feature_acts_masked)
                    reconstruct_act = reconstruct_act.unsqueeze(1)
    
                elif method_name=='sae':
    
                    feature_acts_masked = feature_acts_masked.unsqueeze(1)
                    reconstruct_act = interp_model.decode(feature_acts_masked)
    
                elif method_name=='ica':    
                    try : 
                        mixing_matrix = np.load(f'{path_to_dr_methods}.npy')
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Error: '{path_to_dr_methods}.npy' not found. The mixing matrix for ICA has not been saved.")
                    except Exception as e:
                        raise RuntimeError(f"An unexpected error occurred while loading the mixing matrix for ICA: {str(e)}")
    
                    dr_mean = torch.from_numpy(interp_model.mean_).to(hook_model.cfg.device)
                    mixing_matrix = torch.tensor(mixing_matrix).to(hook_model.cfg.device)
                    reconstruct_act = feature_acts_masked @ mixing_matrix.T + dr_mean
                    reconstruct_act = reconstruct_act.unsqueeze(1)
            
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
                
                #We adapt the logits so that we extract the logits for the class and sum up all the other logits to a category that we could see as undecised
                predicted_logits = logits_reconstruction.contiguous().view(-1, logits_reconstruction.shape[-1])
                class_probs = torch.zeros((predicted_logits.shape[0],len(unique_labels)+1))
                probs = F.softmax(predicted_logits,dim=1)
                prob_alternative = 1
                for i , (key,value) in enumerate(labels_tokens_id.items()):
                      #Predictions
                      class_probs[:,i] = probs[:,value]
                      prob_alternative -= class_probs[:,i]
                class_probs[:,-1] = prob_alternative
                
            
                # if num_batch==(len(activations_dataloader)-1):
                #     classification_predicted_probs[num_batch*inputs.shape[0] : ] = class_probs.cpu()
                # else:
                #     classification_predicted_probs[num_batch*inputs.shape[0] : (num_batch+1)*inputs.shape[0] ] = class_probs.cpu()
                classification_predicted_probs.append( class_probs.cpu())

                number_matches += update_metrics(inputs[evaluated_samples],predicted_logits,labels_tokens_id,dict_metrics_reconstruction,is_eos)
                number_sample+=inputs.shape[0]

    del cache
    del logits_reconstruction
    del predicted_logits
    del class_probs

    classification_predicted_probs_tensor = torch.cat(classification_predicted_probs, dim=0)
    
    accuracy = (number_matches / (len(activations_dataset)*bs)).item()
    
    return  classification_predicted_probs_tensor , accuracy, dict_metrics_reconstruction 

    

def eval_causal_effect_concepts(
    probs_pred : torch.Tensor, 
    accuracy : float,
    class_int : int,
    hook_model:HookedTransformer,
    interp_model,
    ablation_features,
    labels_dataset,
    tokenizer,
    activations_dataset,
    selected_features,
    method_name,
    hook_layer,
    hook_name,
    is_eos=True,
    path_to_dr_methods=''):

  
    #We look at the impact of the ablated features simultaneously
    probs_pred_ablation, accuracy_ablation, dict_metrics_ablation = get_predictions(hook_model,interp_model,labels_dataset,tokenizer,activations_dataset,selected_features,method_name,hook_layer,hook_name,is_eos,perturb=ablation_features,ids_samples=torch.arange(probs_pred.shape[0]),path_to_dr_methods=path_to_dr_methods)

    variation_absolute_accuracy = (accuracy_ablation - accuracy)
    tvd = 0.5 * torch.sum(torch.abs(probs_pred - probs_pred_ablation), dim=1).mean().item()
    print(f'Desactivation of {len(ablation_features)} feature(s)  associated to the class {class_int}, it results in the following effects : \n')
    print(f'TVD : {tvd} ; Absolute Accuracy change: {variation_absolute_accuracy} \n')

    return variation_absolute_accuracy,tvd, dict_metrics_ablation




