# from ..utils import LLMLoadConfig, SAELoadConfig
# from ..model_training import process_dataset,get_hook_model,compute_loss_last_token, PromptTunerForHookedTransformer

# import sys
# sys.path.append("../../")

# from sae_implementation import TrainingSAE

# import torch
# from safetensors import safe_open
# from transformer_lens import HookedTransformer
# from transformers import AutoTokenizer, DataCollatorForLanguageModeling
# from datasets import Dataset
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from functools import partial
# import numpy as np
# import plotly.express as px
# import plotly.io as pio
# import os
# import csv
# import random
# import json
# import pandas as pd
# import umap
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# import plotly.graph_objects as go
# import matplotlib.colors as mcolors
# from tqdm import tqdm
# from loguru import logger
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib.patches import Wedge


# class ActivationDataset(Dataset):
#     def __init__(self):
#         self.data_block = []

#     def append(self, input_ids, cache):
#         self.data_block.append((input_ids,cache))


#     def __len__(self):
#         return len(self.data_block)

#     def __getitem__(self, idx):
    
    
#         # Handle torch.Tensor indices
#         if isinstance(idx, torch.Tensor):
#             if idx.numel() != 1:  # Ensure it's a single element
#                 raise ValueError(f"Index tensor must have a single element, got {idx.numel()}")
#             idx = idx.item()  # Convert single-element Tensor to an integer
    
#         # Handle list indices
#         elif isinstance(idx, list):
#             if len(idx) != 1:  # Ensure it's a single-element list
#                 raise ValueError(f"Index list must have a single element, got {len(idx)}")
#             idx = idx[0]
    
#         # Ensure idx is now an integer
#         if not isinstance(idx, int):
#             raise TypeError(f"Index must be an integer after processing, got {type(idx)}")
    
#         # Access the block and return as a dictionary
#         input_ids, cache = self.data_block[idx]
#         return {
#             "input_ids": input_ids,
#             "cache": cache
#         }



# def display_cosine_similarity_stats(data : torch.tensor, features_number : torch.tensor):
#     #data : shape (n_ind, features)

#     n,d = data.shape 
    
#     # Normalize the vectors along dimension 0 (for cosine similarity calculation)
#     normalized_tensor = data / data.norm(dim=0, keepdim=True)
    
#     # Compute cosine similarity matrix (d x d)
#     cosine_similarity_matrix = torch.mm(normalized_tensor.t(), normalized_tensor)
    
#     # Extract unique pairs by masking the upper triangular matrix (excluding diagonal)
#     i, j = torch.triu_indices(d, d, offset=1)
#     cosine_values = cosine_similarity_matrix[i, j]
    
#     # Compute statistics
#     mean_cosine_similarity = cosine_values.mean().item()
#     variance_cosine_similarity = cosine_values.var().item()

#     # Find the top 3 pairs with the highest cosine similarity
#     top_k = 3
#     top_values, top_indices = torch.topk(cosine_values, top_k, largest=True)

#     # Find the pairs corresponding to the top similarities
#     top_pairs = [(features_number[i[idx].item()], features_number[j[idx].item()]) for idx in top_indices]
    
    
#     # Display results
#     print(f"Mean cosine similarity: {mean_cosine_similarity}")
#     print(f"Variance of cosine similarity: {variance_cosine_similarity}")

#     for rank, (value, pair) in enumerate(zip(top_values, top_pairs), 1):
#         print(f"Top {rank} cosine similarity: {value.item()} (between vectors {pair})")

#     max_activity = 0
#     mean_activity = 0
#     #Save the distribution of the feature
#     for i,feature_number in enumerate(features_number):

#         active_samples = torch.sum(data[:,i] > 0)
#         percentage_activity = (active_samples / n) * 100
#         mean_activity += percentage_activity
#         if percentage_activity > max_activity:
#             max_activity = percentage_activity
        
#         plt.figure(figsize=(8, 6))
#         plt.hist(data[:,i], bins=50, edgecolor='black', alpha=0.7,density=True)
#         plt.title(f"Feature {feature_number} : Activation {percentage_activity}%")
#         plt.xlabel("Value")
#         plt.ylabel("Frequency")
    
#         # Save the plot
#         plt.savefig(f"results/top_texts/{feature_number}.pdf")
#         plt.close()

#     mean_activity /= len(features_number)
#     print(f"Mean activity of the selected features in the segments : {mean_activity:.2f}%")
#     print(f"Max activity of feature among the selected features in the segments : {max_activity:.2f}%")



# def compute_feature_cosine_similarity(data,mean_activations):
#     """
#     Computes the mean and max cosine similarity between features in the given data tensor.

#     Parameters:
#     - data: numpy array of shape (n, 1, f), where n is the number of individuals and f is the number of features.

#     Returns:
#     - mean_cosine_similarity: Mean cosine similarity between different features.
#     - max_cosine_similarity: Maximum cosine similarity between different features.
#     """
#     n, _, f = data.shape

#     # Reshape the tensor to (n, f)
#     X = data.reshape(n, f)

#     # Transpose X so that features are along the rows
#     X_T = X.T  # Now X_T is of shape (f, n)

#     # Compute dot products between feature vectors
#     dot_product_matrix = np.dot(X_T, X_T.T)  # Shape: (f, f)

#     # Compute the norm (magnitude) of each feature vector
#     feature_norms = np.linalg.norm(X_T, axis=1)  # Shape: (f,)

#     # Compute the outer product of norms to get denominator for cosine similarity
#     norm_matrix = np.outer(feature_norms, feature_norms)  # Shape: (f, f)

#     # Avoid division by zero
#     epsilon = 1e-10
#     norm_matrix[norm_matrix == 0] = epsilon

#     # Compute the cosine similarity matrix
#     cosine_similarity_matrix = dot_product_matrix / norm_matrix

#     #Compute weights (outer product of mean activations)
#     weights = np.outer(mean_activations, mean_activations)
#     print(f"weights shape : {weights.shape}")

#     weighted_cosine_similarity_matrix = weights * cosine_similarity_matrix

#     # Create a mask to exclude diagonal elements
#     mask = ~np.eye(f, dtype=bool)

#     # Extract the cosine similarity values excluding the diagonal
#     cosine_values = weighted_cosine_similarity_matrix[mask]
    
#     print(f"cosine_similarity_matrix shape : {cosine_similarity_matrix.shape}")

#     # Compute mean and max cosine similarity
#     mean_cosine_similarity = np.mean(cosine_values)
#     max_cosine_similarity = np.max(cosine_values)

#     return mean_cosine_similarity, max_cosine_similarity

# def design_figure(W_dec_pca, sizes, np_keys_prototypes, np_prototypes_pca, top_logits, bottom_logits, colors, feature_colors, normalized_class_scores,N):

#   feature_colors = np.array(['#636EFA','#EF553B','#00CC96','#3D2B1F'])
#   #feature_colors = np.array(['#636EFA','#EF553B'])
#   # Extracting the x and y components of the vectors
#   x = W_dec_pca[:, 0]
#   y = W_dec_pca[:, 1]


#   # Create the figure and axis
#   fig, ax = plt.subplots(figsize=(8, 8))

#   labels_names = np.array(['World','Sport', 'Business','Sci/Tech'])
#   #labels_names = np.array(['Non Toxic','Toxic'])

#   # Plot the special points with yellow triangles
#   ax.scatter(np_prototypes_pca[:,0], np_prototypes_pca[:,1], color='orange', marker='^', s=400, zorder=6)

#   # Add placards with labels for special points
#   for i, label in enumerate(labels_names):
#       ax.text(np_prototypes_pca[:,0][i] - 4.5, np_prototypes_pca[:,1][i] + 0.1, label, fontsize=15,bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'),zorder=100)

#   # Add grid, labels, and title
#   ax.axhline(0, color='grey', lw=1,zorder=10)
#   ax.axvline(0, color='grey', lw=1,zorder=10)
#   ax.grid(True, linestyle='--', alpha=0.5,zorder=10)

#   # Function to create a pie chart marker at a specific location
#   # def draw_pie(ax,center, sizes, colors, radius=0.5):
#   #     # Starting angle
#   #     start_angle = 0

#   #     # Iterate through sizes and corresponding colors to draw pie slices
#   #     for size, color in zip(sizes, colors):
#   #         # print(size)
#   #         # print(color)
#   #         # print(center)
#   #         # Draw a wedge (slice of the pie)
#   #         wedge = ax.pie(
#   #             [size, 1 - size],
#   #             colors=[color, 'none'],  # second color is transparent to create the effect of a single slice
#   #             startangle=start_angle,
#   #             radius=radius,
#   #             center=center
#   #         )
#   #         # Update the start angle
#   #         start_angle += size * 360


#   # Function to create a pie chart marker using Wedges at a specific location
#   def draw_pie(ax, center, sizes, colors, radius=0.5, alpha=0.8):
     
#       # Starting angle
#       start_angle = 0

#       # Iterate through sizes and corresponding colors to draw pie slices
#       for size, color in zip(sizes, colors):
#           # Calculate the end angle of the wedge
#           end_angle = start_angle + size * 360
#           # Create a wedge patch for each slice
#           wedge = Wedge(
#               center, radius, start_angle, end_angle,
#               facecolor=color, alpha=alpha, zorder=2
#           )
#           # Add the wedge to the axes
#           ax.add_patch(wedge)
#           # Update the start angle for the next slice
#           start_angle = end_angle

#   # Optional: Plot the vector points as scatter
#   #plt.scatter(x, y, s=sizes**2,color=colors, zorder=5)
#   for coord, prop,radius in zip(W_dec_pca[:,:2], normalized_class_scores.T,sizes):
#       draw_pie(ax,coord, prop, feature_colors, radius=(0.3*radius+1e-8))


#   # Set axis limits to give some padding around the vectors
#   range_x = np.concatenate((x,np_prototypes_pca[:,0]))
#   range_y = np.concatenate((y,np_prototypes_pca[:,1]))
    
#   xlim_min = min(range_x) - 1
#   xlim_max = max(range_x) + 1
#   ylim_min = min(range_y) - 1
#   ylim_max = max(range_y) + 1



#   ax.set_xlim(xlim_min, xlim_max)
#   ax.set_ylim(ylim_min, ylim_max)


#   ax.set_xlabel('PCA component 1')
#   ax.set_ylabel('PCA component 2')
#   ax.set_aspect('equal', 'box')

#   return ax, fig  #, (xlim_min, xlim_max, ylim_min,ylim_max)




# def decode_dataset(tokenized_dataset,tokenizer):
    
#     original_texts = []
    
#     for input_ids in tokenized_dataset['input_ids']:
#         # Decoding the token IDs back to text
#         text = tokenizer.decode(input_ids, skip_special_tokens=True)
#         original_texts.append(text)

#     return original_texts


# def reconstr_hook(activation, hook, sae_out):
#     return sae_out


# def reconstr_hook_classification_token(activation, hook, sae_out):
#     #activation[:,-3,:] = sae_out.squeeze(1)
#     #return activation
#     n,m,d = activation.shape
#     sae_activations = sae_out.squeeze(1).view(n,m,d)
#     return sae_activations


# def compute_loss_last_token_classif(
#     inputs_labels: torch.Tensor,
#     outputs_logits: torch.Tensor,
#     labels_tokens_id : dict,
#     is_eos: bool,
#     vocab_size: int,
#     reduction: str = "mean"
# ):
#   """Computes the loss that focuses on the classification. Given that we only care about the classification, we sum all the logits of tokens that are not associated to any class and 
#   create another class : Other/Unknown"""

#   #outputs_logits of size [8,M,50304]
#   #8 : batch size
#   #M : max length sequence in that batch
#   # 50304 : vocabulary size

#   logits_prediction = outputs_logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size)
#   labels_to_pred = inputs_labels[:,(-1-int(is_eos))].view(-1)

#   loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)

#   #We do a clone otherwise, the change that we impose on the labels is also propagated to the original input
#   labels_to_pred_clone = labels_to_pred.clone()
    
#   prediction_vector = torch.zeros((logits_prediction.shape[0],len(labels_tokens_id)+1)).to(logits_prediction.device)
#   probs = F.softmax(logits_prediction,dim=1)
#   prob_alternative = 1
#   for i , (key,value) in enumerate(labels_tokens_id.items()):

#       #Predictions
#       prediction_vector[:,i] = probs[:,value]
#       prob_alternative -= prediction_vector[:,i]

#       #Rename tokens to their labels
#       labels_to_pred_clone[labels_to_pred_clone==value] = i
    
#   prediction_vector[:,-1] = prob_alternative
#   # print(f"prediction_vector : {prediction_vector}")
#   # print(f"labels_to_pred_clone : {labels_to_pred_clone}")

#   return loss_ce(prediction_vector,labels_to_pred_clone)


# def update_metrics(
#     labels : torch.Tensor,
#     logits : torch.Tensor,
#     labels_tokens_id : dict,
#     dict_metrics : dict,
#     is_eos : bool
# ):
#     _,_,vocab_size = logits.size()

    
#     logits_prediction = logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size).argmax(dim=1)
#     labels_to_pred = labels[:,(-1-int(is_eos))].view(-1)
    

#     exact_matches = (logits_prediction==labels_to_pred)
#     # print(f"labels_to_pred : {labels_to_pred}")
#     # print(f"logits_prediction : {logits_prediction}")
#     count_exact_matches = exact_matches.sum() #tensor

    
#     for key,value in labels_tokens_id.items():
#       #In case the keys do not already exist (typically the first time we call this function)
#       dict_metrics.setdefault(f'number real samples_{key}',0)
#       dict_metrics.setdefault(f'true matches_{key}',0)
#       dict_metrics.setdefault(f'number predicted samples_{key}',0)
      
#       position_key = (labels_to_pred==value)
#       number_samples_key = (labels_to_pred==value).sum().item()
#       dict_metrics[f'number real samples_{key}'] += number_samples_key

#       exact_matches_key = position_key & exact_matches
#       count_exact_matches_key = exact_matches_key.sum().item()
#       dict_metrics[f'true matches_{key}']+= count_exact_matches_key

#       count_predicted_key = (logits_prediction==value).sum().item()
#       dict_metrics[f'number predicted samples_{key}'] += count_predicted_key

#     return count_exact_matches


# def compute_same_predictions(
#     original_logits : torch.Tensor,
#     reconstruction_logits : torch.Tensor,
#     is_eos: bool
# ):

#     _,_,vocab_size = original_logits.size()
    
#     y_pred_original = original_logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size).argmax(dim=1)
#     y_pred_reconstruction = reconstruction_logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size).argmax(dim=1)

#     return (y_pred_original==y_pred_reconstruction).sum()


# def compute_metrics_variations(
#     dict_metrics_original : dict,
#     dict_metrics_reconstruction : dict,
#     accuracy_original : float,
#     accuracy_reconstruction : float,
#     labels_tokens_id : dict
# ):
#      dict_metrics_variation = {}
    
#      for key in labels_tokens_id.keys():
#       dict_metrics_original[f'recall_{key}'] = dict_metrics_original[f'true matches_{key}'] / dict_metrics_original[f'number real samples_{key}']
#       dict_metrics_original[f'precision_{key}'] = 0 if  dict_metrics_original[f'number predicted samples_{key}']==0  else dict_metrics_original[f'true matches_{key}'] / dict_metrics_original[f'number predicted samples_{key}']
#       dict_metrics_original[f'f1-score_{key}'] = 0 if (dict_metrics_original[f'recall_{key}'] + dict_metrics_original[f'precision_{key}'])==0  else 2 * dict_metrics_original[f'recall_{key}'] * dict_metrics_original[f'precision_{key}'] / (dict_metrics_original[f'recall_{key}'] + dict_metrics_original[f'precision_{key}'])
      
#       dict_metrics_reconstruction[f'recall_{key}'] = dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number real samples_{key}']
#       dict_metrics_reconstruction[f'precision_{key}'] = 0 if dict_metrics_reconstruction[f'number predicted samples_{key}']==0 else dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number predicted samples_{key}']
#       dict_metrics_reconstruction[f'f1-score_{key}'] = 0 if (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])==0 else 2 * dict_metrics_reconstruction[f'recall_{key}'] * dict_metrics_reconstruction[f'precision_{key}'] / (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])

#       dict_metrics_variation[f'Delta recall_{key}'] = dict_metrics_reconstruction[f'recall_{key}'] - dict_metrics_original[f'recall_{key}']
#       dict_metrics_variation[f'Delta precision_{key}'] = dict_metrics_reconstruction[f'precision_{key}'] - dict_metrics_original[f'precision_{key}']
#       dict_metrics_variation[f'Delta f1-score_{key}'] = dict_metrics_reconstruction[f'f1-score_{key}'] - dict_metrics_original[f'f1-score_{key}']

#      dict_metrics_original['Delta Global accuracy'] = accuracy_original
#      dict_metrics_reconstruction['Delta Global accuracy'] = accuracy_reconstruction
#      dict_metrics_variation['Delta Global accuracy'] = accuracy_reconstruction - accuracy_original
    
#      return dict_metrics_variation, dict_metrics_original


# def compute_metrics_details(
#     dict_metrics_reconstruction : dict,
#     labels_tokens_id : dict
# ):
     
#      for key in labels_tokens_id.keys():
      
#       dict_metrics_reconstruction[f'recall_{key}'] = dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number real samples_{key}']
#       dict_metrics_reconstruction[f'precision_{key}'] = 0 if dict_metrics_reconstruction[f'number predicted samples_{key}']==0 else dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number predicted samples_{key}']
#       dict_metrics_reconstruction[f'f1-score_{key}'] = 0 if (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])==0 else 2 * dict_metrics_reconstruction[f'recall_{key}'] * dict_metrics_reconstruction[f'precision_{key}'] / (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])

    
    


# def eval_hook_loss(
#     hook_model:HookedTransformer,
#     sae:HookedTransformer,
#     dataset:Dataset, #expected to be tokenized
#     data_collator,
#     tokenizer,
#     len_example,
#     len_template,
#     device,
#     return_feature_activations=False,
#     batch_size=1,proportion_to_evaluate=1.,is_eos=True,prompt_tuning=False,loss_type='cross_entropy'):

   
#     #Number of sentences to select among the evaluation dataset
#     n_selected = int(proportion_to_evaluate * len(dataset))

#     #We take a number of sentences proportional to the size of the batch in order to have batches of same size.
#     n_selected -= n_selected % batch_size
#     if n_selected <= 0:
#         raise ValueError("The proportion of data on which the evaluation is done is too small, please increase 'proportion_to_evaluate'")
#     evaluated_dataset = dataset.select(range(n_selected))

    
#     # Create DataLoader
#     dataloader = DataLoader(evaluated_dataset, batch_size=batch_size, collate_fn=data_collator)
    
#     #Retrieve a dictionary matching the labels to their tokens ids
#     vocab = tokenizer.get_vocab()
#     unique_labels = np.unique(np.array(dataset["token_labels"]))
#     keys_labels = set(unique_labels)
#     labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}
    

#     dict_metrics_original = {}
#     dict_metrics_reconstruction = {}
#     total_matches_original = 0
#     total_matches_reconstruction = 0
#     total_same_predictions = 0

#     #Save the different activations to speed the causality calculations done after
#     activations_dataset = ActivationDataset()

#     # Evaluation loop
#     hook_model.eval()
#     sae.eval()
#     with torch.no_grad():

#         store_l0_activations = []
#         original_total_loss = 0
#         sae_reconstruction_total_loss = 0
#         if return_feature_activations:
#             feature_activations_list = []
#             acts_without_process_list = []
#             original_activation_list = []
#             prompt_labels_list = []
#             model_logits_labels = {}
        
        
#         for batch in tqdm(dataloader, desc="Forward Passes of the given LLM model", unit="batch"):

#             input_ids = batch['input_ids'].to(device)
#             prompt_labels = batch['token_labels']
#             attention_mask = batch['attention_mask'].to(dtype=int).to(device)

#             if prompt_tuning:
#                 batch_size = input_ids.size(0)
#                 prompt_token_ids = torch.full(
#                     (batch_size, hook_model.num_prompt_tokens),
#                     hook_model.tokenizer.eos_token_id,  # Using EOS token as a placeholder
#                     dtype=torch.long,
#                     device=input_ids.device,
#                 )
#                 input_ids = torch.cat([prompt_token_ids, input_ids], dim=1)
#                 prompt_attention_mask = torch.full(
#                     (batch_size, hook_model.num_prompt_tokens),
#                     1,  # Using EOS token as a placeholder
#                     dtype=torch.long,
#                     device=input_ids.device,
#                 )
#                 attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)            
            

#             #Manually fixing the issue of sentence longer than the context windows size since it is not automatically handled by transformer-lens and it causes conflict with the positional embedding that it dones on a vector of size the context attention windows
#             if input_ids.shape[1] > hook_model.cfg.n_ctx:
#                 attention_mask = attention_mask[:,-hook_model.cfg.n_ctx:]
#                 input_ids = input_ids[:,-hook_model.cfg.n_ctx:] #It should not be an issue since we want to predict for almost the last token, so it looses almost nothing of the context it could have seen otherwise

            
#             outputs, cache = hook_model.run_with_cache(input_ids,
#                                                 attention_mask=attention_mask,
#                                                 names_filter=sae.cfg.hook_name,
#                                                 prepend_bos=False)

#             '''outputs : Tensor of shape [batches,max_length, vocab_size] it will also be useful when we would like to compute the KL Divergence the logits distribution of the original model and 
#             that of the model with SAE activations reconstruction. 
#             cache : object of type 'ActivationCache' in transformer_lens.ActivationCache.py '''

#             #Small test         
#             # flatten = cache[sae.cfg.hook_name].view(-1,cache[sae.cfg.hook_name].shape[-1]).unsqueeze(1)
#             # n,m,d = cache[sae.cfg.hook_name].shape
#             # unflatten = flatten.squeeze(1).view(n,m,d)
#             # print(f'Is the manipualtion correct : {(unflatten==cache[sae.cfg.hook_name]).all()}')
            

            
#             cache_tensor = cache[sae.cfg.hook_name][:,(-2-int(is_eos)),:].unsqueeze(1)
#             cache_tensor = cache_tensor.to(dtype=torch.float32) #In case the model is quantized
#             cache_flatten = cache[sae.cfg.hook_name].view(-1,cache[sae.cfg.hook_name].shape[-1]).unsqueeze(1)

        

#             # Use the SAE
#             feature_acts, acts_without_process = sae.encode_with_hidden_pre(cache_tensor)
#             sae_out = sae.decode(feature_acts)
            
            
#             feature_acts_all, acts_without_process_all = sae.encode_with_hidden_pre(cache_flatten)
#             sae_out_all = sae.decode(feature_acts_all)

#             #Save the activations and labels for the causality calculations done after
#             bs = input_ids.shape[0]
#             d_in = cache_flatten.shape[-1]
#             cache_to_save = cache_flatten.view(bs,-1,1,d_in)
#             activations_dataset.append(input_ids, cache_to_save)
            

#             #If in addition to the SAE metrics, we want to store feature activations and model predictions
#             if return_feature_activations:
#                 feature_activations_list.append(feature_acts.cpu())
#                 acts_without_process_list.append(acts_without_process.cpu())
#                 original_activation_list.append(cache_tensor.cpu())
#                 prompt_labels_list.append(prompt_labels)
                
#                 #In order to be aware of the corresponding predictions of the model when we work on the sae activations
#                 model_prediction_logits = outputs[:,(-2-int(is_eos))].contiguous().view(-1, outputs.shape[-1])
#                 #Only select the logits of the tokens id corresponding to the labels (for memory efficiency)
#                 for key, value in labels_tokens_id.items():
#                     token_id = value
#                     logits_tensor =  model_prediction_logits[:,token_id]
#                     if key in model_logits_labels:
#                         model_logits_labels[key] =  torch.cat((model_logits_labels[key],logits_tensor))
#                     else:
#                         model_logits_labels[key] = logits_tensor
               
                    
        
#             epsilon = 1e-4
#             #Store L0 sparsity on the batch
#             l0 = (feature_acts > epsilon).float().sum(-1).detach() #l0 of size [len_batch,max_length] so we have the l0 for each hidden state/token of each sentence in the batch
#             l0_batch = l0.flatten().cpu().numpy()
#             store_l0_activations.append(l0_batch)

#             #Cross entropy loss with SAE activations reconstruction on the batch
           
#             #We compute the reconstruction by replacing all the hidden states by their reconstruction and not only the hidden state predicting the token of the class despite that the SAE is only
#             #trained on this hidden state for each sentence. This is because otherwise, even if the reconstruction is not good, it might still be able to retrieve information on the other untouched
#             #hidden states in the sentence.
#             logits_reconstruction = hook_model.run_with_hooks(
#                 input_ids,
#                 attention_mask=attention_mask,
#                 fwd_hooks=[
#                     (
#                         sae.cfg.hook_name,
#                         partial(reconstr_hook_classification_token, sae_out=sae_out_all),
#                     ) ],
#                 return_type="logits",
#             )
        


#             #We compute the original classification cross-entropy loss and the same loss obtained by plugging the reconstructed activations from the SAE features at hook {sae.cfg.hook_name} 
#             original_loss = compute_loss_last_token_classif(input_ids,outputs,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[2])
#             original_total_loss  += original_loss
#             sae_reconstruction_loss = compute_loss_last_token_classif(input_ids,logits_reconstruction,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[2])
#             sae_reconstruction_total_loss += sae_reconstruction_loss

            
#             #We compute the variation in true accuracy
#             acc_original = update_metrics(input_ids,outputs,labels_tokens_id,dict_metrics_original,is_eos)
#             acc_reconstruction = update_metrics(input_ids,logits_reconstruction,labels_tokens_id,dict_metrics_reconstruction,is_eos)
#             total_matches_original += acc_original.item()
#             total_matches_reconstruction += acc_reconstruction.item()
#             #We compute the recovering accuracy metric
#             count_same_predictions = compute_same_predictions(outputs,logits_reconstruction,is_eos)
#             total_same_predictions += count_same_predictions.item()
            
#         del cache

#         accuracy_original = total_matches_original / len(evaluated_dataset)
#         accuracy_reconstruction = total_matches_reconstruction / len(evaluated_dataset)
#         recovering_accuracy = total_same_predictions / len(evaluated_dataset)

#         dict_metrics_variations, dict_metrics_original = compute_metrics_variations(dict_metrics_original,dict_metrics_reconstruction,accuracy_original,accuracy_reconstruction,labels_tokens_id)
            
#         array_store_l0_activations = np.concatenate(store_l0_activations)
#         mean_l0_activations = np.mean(array_store_l0_activations)
#         # Create the histogram
#         fig = px.histogram(
#             array_store_l0_activations, 
#             title=f'L0 sparsity histogram at hook {sae.cfg.hook_name} over {array_store_l0_activations.size} activations.  Average L0 : {mean_l0_activations}', 
#             labels={'value': 'L0 norm'}, 
#             template='plotly_white'  # Choose a template: 'plotly', 'plotly_white', 'plotly_dark', etc.
#         )
        
#         # Customize axes titles
#         fig.update_xaxes(title_text='L0 Norm')
#         fig.update_yaxes(title_text='Number of occurences')
        
        
#         sae_reconstruction_mean_loss = sae_reconstruction_total_loss / len(dataloader)
#         original_mean_loss = original_total_loss / len(dataloader)

#         print(f"\nOriginal classification crossentropy mean loss : {original_mean_loss} ({total_matches_original} matches) - SAE reconstruction classification crossentropy mean loss : {sae_reconstruction_mean_loss} (Computed over {n_selected} sentences) ")
#         print(f'\nRecovering accuracy : {recovering_accuracy}')
#         print(f"\nOriginal accuracy of the model : {accuracy_original} - Accuracy of the model when plugging the SAE reconstruction hidden states : {accuracy_reconstruction}")

        
#         sae_performance = {'Original Mean Loss':original_mean_loss.item(), 'SAE Mean loss':sae_reconstruction_mean_loss.item(), 'Mean L0 activation' : float(mean_l0_activations), 'Number sentences':n_selected, 'Recovering accuracy':recovering_accuracy, 'Original accuracy':accuracy_original, 'Reconstruction accuracy' : accuracy_reconstruction}
#         #We merge the macro information of the sae with the dictionary of the variation metrics
#         sae_performance = sae_performance | dict_metrics_variations | dict_metrics_original
    
#         #This part is not functional when training not only on the single token
#         mean_activations = torch.zeros_like(feature_activations_list[0][0])
#         for sample_activation in feature_activations_list:
#             mean_activations += sample_activation.sum(dim=0)
#         mean_activations = mean_activations[0]
#         mean_activations /= len(feature_activations_list)
        
#         if return_feature_activations:
            
#             #Return the decoded sentences used to generate the activations
#             original_text_used = decode_dataset(evaluated_dataset, tokenizer)

#             #Remove the template part
#             for t,_ in enumerate(original_text_used):
#                 original_text_used[t] = original_text_used[t][len_example:-len_template] #Len specific to the AG News template, has to be adapted to do it automatically for other datasets
            
#             #Return the ground truth labels of the prompts
#             prompt_labels_tensor = torch.cat(prompt_labels_list,dim=0)
            
#             #The number of hidden states is the same for each prompt, equals to 1 as we keep only the (almost) last token. So we can concatenate as the last two dimensions are the same for each feature activation vector.
#             feature_activations_tensor = torch.cat(feature_activations_list,dim=0)
#             acts_without_process_tensor = torch.cat(acts_without_process_list,dim=0)
#             original_activation_tensor = torch.cat(original_activation_list,dim=0)

#             #Compute similarity activations stats between features
#             mean_cos_sim, max_cos_sim = compute_feature_cosine_similarity(feature_activations_tensor,mean_activations)

#             print(f'The mean cosine similarity between the features activations is {mean_cos_sim} with a maximum up to {max_cos_sim}')
            
#             return sae_performance , fig, mean_activations, {'feature activation' : feature_activations_tensor, 'activations without processing' : acts_without_process_tensor,'original activation' : original_activation_tensor, 'prompt label' : prompt_labels_tensor, 'model output logits' : model_logits_labels, 'encoder' : sae.W_enc}, original_text_used, activations_dataset
           
#         return sae_performance, fig, activations_dataset


# def count_matches(labels,logits,is_eos):

#     _,_,vocab_size = logits.size()

#     logits_prediction = logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size).argmax(dim=1)
#     labels_to_pred = labels[:,(-1-int(is_eos))].view(-1)

#     exact_matches = (logits_prediction==labels_to_pred)
#     count_exact_matches = exact_matches.sum() #tensor

#     return count_exact_matches.item()
    

# def run_model_to_get_pred(
#     hook_model:HookedTransformer,
#     sae:HookedTransformer,
#     dataset:Dataset, #expected to be tokenized
#     data_collator,
#     tokenizer,
#     activations_dataset,
#     device,
#     batch_size=1,proportion_to_evaluate=1.,is_eos=True,prompt_tuning=False,loss_type='cross_entropy',
#     perturb=None
# ):
#     dict_metrics_reconstruction = {}
    
#     #Number of sentences to select among the evaluation dataset
#     n_selected = int(proportion_to_evaluate * len(dataset))

#     #We take a number of sentences proportional to the size of the batch in order to have batches of same size.
#     n_selected -= n_selected % batch_size
#     if n_selected <= 0:
#         raise ValueError("The proportion of data on which the evaluation is done is too small, please increase 'proportion_to_evaluate'")
#     evaluated_dataset = dataset.shuffle(seed=0).select(range(n_selected))

#     # Create DataLoader
#     dataloader = DataLoader(evaluated_dataset, batch_size=batch_size, collate_fn=data_collator)
    
#     #Retrieve a dictionary matching the labels to their tokens ids
#     vocab = tokenizer.get_vocab()
#     unique_labels = np.unique(np.array(dataset["token_labels"]))
#     keys_labels = set(unique_labels)
#     labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}

#     # Evaluation loop
#     hook_model.eval()
#     sae.eval()

#     #This is where we store the prediction logits specific to classification. One for each category plus one that sums up all the logits associated to tokens which do not belong to the tokens of a class
#     classification_predicted_probs = torch.zeros((len(evaluated_dataset),len(unique_labels)+1)).cpu()

#     #For accuracy 
#     number_matches = 0

#     activations_dataloader = DataLoader(activations_dataset,batch_size=1)

#     with torch.no_grad():
#         # for num_batch, batch in tqdm(enumerate(dataloader), desc="Forward Passes with the SAE with ablation on feature(s)", unit="batch"):
#         #         inputs = batch['input_ids'].to(device)
#         #         prompt_labels = batch['token_labels']
#         #         attention_mask = batch['attention_mask'].to(dtype=int).to(device)

#         #         if prompt_tuning:
#         #             batch_size = input_ids.size(0)
#         #             prompt_token_ids = torch.full(
#         #                 (batch_size, hook_model.num_prompt_tokens),
#         #                 hook_model.tokenizer.eos_token_id,  # Using EOS token as a placeholder
#         #                 dtype=torch.long,
#         #                 device=input_ids.device,
#         #             )
#         #             input_ids = torch.cat([prompt_token_ids, input_ids], dim=1)
#         #             prompt_attention_mask = torch.full(
#         #                 (batch_size, hook_model.num_prompt_tokens),
#         #                 1,  # Using EOS token as a placeholder
#         #                 dtype=torch.long,
#         #                 device=input_ids.device,
#         #             )
#         #             attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)          
    
#         #         #Manually fixing the issue of sentence longer than the context windows size since it is not automatically handled by transformer-lens and it causes conflict with the positional embedding that it dones on a vector of size the context attention windows
#         #         if inputs.shape[1] > hook_model.cfg.n_ctx:
#         #             attention_mask = attention_mask[:,-hook_model.cfg.n_ctx:]
#         #             inputs = inputs[:,-hook_model.cfg.n_ctx:] #It should not be an issue since we want to predict for almost the last token, so it looses almost nothing of the context it could have seen otherwise
               
                
#         #         _, cache = hook_model.run_with_cache(inputs,
#         #                                             attention_mask=attention_mask,
#         #                                             names_filter=sae.cfg.hook_name,
#         #                                             prepend_bos=False)
#         #         '''outputs : Tensor of shape [batches,max_length, vocab_size] it will also be useful when we would like to compute the KL Divergence the logits distribution of the original model and 
#         #         that of the model with SAE activations reconstruction. 
#         #         cache : object of type 'ActivationCache' in transformer_lens.ActivationCache.py '''
    
                
#         #         cache_flatten = cache[sae.cfg.hook_name].view(-1,cache[sae.cfg.hook_name].shape[-1]).unsqueeze(1)
#         #         cache_flatten = cache_flatten.to(dtype=torch.float32) #In case the model is quantized
              
       

#         for num_batch, batch in tqdm(enumerate(activations_dataloader), desc="Forward Passes with the SAE with ablation on feature(s)", unit="batch"):
                
#                 inputs = batch["input_ids"]
#                 cache = batch["cache"]

#                 a,b,c,_ = cache.shape
#                 cache_flatten = cache.view(a*b,c,-1)
#                 cache = cache.squeeze(2)

#                 # Use the SAE   
#                 feature_acts_all, acts_without_process_all = sae.encode_with_hidden_pre(cache_flatten)
                
            
#                 if perturb is not None:
#                     feature_acts_all[:,:,perturb] = 0
#                     acts_without_process_all[:,:,perturb] = 0
#                     #cache_flatten[:,:,perturb] = 0
            
#                 sae_out_all = sae.decode(feature_acts_all)
#                 # print(f'sae_out_all : {sae_out_all.shape}')
#                 # print(f'cache_flatten : {cache_flatten.shape}')
                        
                
#                 #We compute the reconstruction by replacing all the hidden states by their reconstruction and not only the hidden state predicting the token of the class despite that the SAE is only
#                 #trained on this hidden state for each sentence. This is because otherwise, even if the reconstruction is not good, it might still be able to retrieve information on the other untouched
#                 #hidden states in the sentence.
                
        
#                 # logits_reconstruction = hook_model.run_with_hooks(
#                 #     inputs,
#                 #     attention_mask=attention_mask,
#                 #     fwd_hooks=[
#                 #         (
#                 #             sae.cfg.hook_name,
#                 #             partial(reconstr_hook_classification_token, sae_out=sae_out_all),
#                 #         ) ],
#                 #     return_type="logits",
#                 # )

                
            
#                 logits_reconstruction = hook_model.run_with_hooks(
#                     cache,
#                     start_at_layer=sae.cfg.hook_layer,
#                     fwd_hooks=[
#                         (
#                             sae.cfg.hook_name,
#                             partial(reconstr_hook_classification_token, sae_out=sae_out_all),
#                         ) ],
#                     return_type="logits",
#                 )


#                 #Compute accuracy
#                 number_matches += count_matches(inputs,logits_reconstruction,is_eos)
                
#                 #We adapt the logits so that we extract the logits for the class and sum up all the other logits to a category that we could see as undecised
#                 predicted_logits = logits_reconstruction[:,(-2-int(is_eos))].contiguous().view(-1, logits_reconstruction.shape[-1])
#                 class_probs = torch.zeros((predicted_logits.shape[0],len(unique_labels)+1))
#                 probs = F.softmax(predicted_logits,dim=1)
#                 prob_alternative = 1
#                 for i , (key,value) in enumerate(labels_tokens_id.items()):
#                       #Predictions
#                       class_probs[:,i] = probs[:,value]
#                       prob_alternative -= class_probs[:,i]
#                 class_probs[:,-1] = prob_alternative
    
#                 if num_batch==(len(dataloader)-1):
#                     classification_predicted_probs[num_batch*inputs.shape[0] : ] = class_probs.cpu()
#                 else:
#                     classification_predicted_probs[num_batch*inputs.shape[0] : (num_batch+1)*inputs.shape[0] ] = class_probs.cpu()

#                 #acc_original = update_metrics(inputs,outputs,labels_tokens_id,dict_metrics_original)
#                 acc = update_metrics(inputs,logits_reconstruction,labels_tokens_id,dict_metrics_reconstruction,is_eos)

#     del cache
#     del cache_flatten
#     del logits_reconstruction
#     del predicted_logits
#     del class_probs

#     compute_metrics_details(dict_metrics_reconstruction,labels_tokens_id)
    
#     accuracy = number_matches / len(evaluated_dataset)

#     return  classification_predicted_probs , accuracy, dict_metrics_reconstruction 
    


# def eval_causal_effect_model(
#     class_int : int,
#     causal_features: torch.Tensor,
#     hook_model:HookedTransformer,
#     sae:HookedTransformer,
#     dataset:Dataset, #expected to be tokenized
#     data_collator,
#     tokenizer,
#     activations_dataset,
#     device,
#     return_feature_activations=False,
#     batch_size=1,proportion_to_evaluate=1.,is_eos=True,prompt_tuning=True,loss_type='cross_entropy'):

#     #This is where we save the impact and change of accuracy resulting from the ablation of one feature
#     overall_effects = []
#     overall_accs_change = []
#     overall_dict_metrics = []
    
#     #We first run the model with all the features on, then with one of the feature disable one at a time
#     probs_pred, accuracy, dict_metrics_original = run_model_to_get_pred(hook_model,sae,dataset,data_collator,tokenizer,activations_dataset,device,batch_size,proportion_to_evaluate,is_eos,prompt_tuning,loss_type,perturb=None)
    

#     write = ''

#     #We look at the impact of ablations of all the features expect those contained in 'causal_features'
#     all_features = torch.arange(0,sae.cfg.d_sae) 
#     #all_features = torch.arange(0,sae.cfg.d_in) 
#     # mask = ~torch.isin(all_features, causal_features)
#     # features_to_ablate = all_features[mask]
#     # probs_pred_only_main, accuracy_only_main, dict_metrics_only_main = run_model_to_get_pred(hook_model,sae,dataset,data_collator,tokenizer,activations_dataset,device,batch_size,proportion_to_evaluate,is_eos,prompt_tuning,loss_type,perturb=features_to_ablate)
#     # only_keep_top10_accuracy_change_relative = (accuracy_only_main - accuracy) / accuracy
#     # only_keep_top10_effect = (probs_pred_only_main - probs_pred).abs().mean().item()
#     # print(f'When we only maintain the {causal_features.tolist()} features by the metric of mean activation, associated to the class {class_int}, it results in the following effects : \n')
#     # print(f'Effect : {only_keep_top10_effect}; Realtive Accuracy change: {only_keep_top10_accuracy_change_relative} \n')

#     only_keep_top10_accuracy_change_relative = 0
#     only_keep_top10_effect = 0
#     dict_metrics_only_main = {}
    

#     #We look at the impact of ablations of all the selected features simultaneously
#     probs_pred_without_selected, accuracy_only_without_selected, dict_metrics_without_selected = run_model_to_get_pred(hook_model,sae,dataset,data_collator,tokenizer,activations_dataset,device,batch_size,proportion_to_evaluate,is_eos,prompt_tuning,loss_type,perturb=causal_features)
#     without_selected_accuracy_change_relative = (accuracy_only_without_selected - accuracy) / accuracy
#     without_selected_effect = (probs_pred_without_selected - probs_pred).abs().mean().item()
#     print(f'When we desactivate the {causal_features.tolist()} features by the metric of mean activation,  associated to the class {class_int}, it results in the following effects : \n')
#     print(f'Effect : {without_selected_effect}; Relative Accuracy change: {without_selected_accuracy_change_relative} \n')
    
#     # for feature in causal_features:
#     #     probs_pred_perturbed, accuracy_perturbed, dict_metrics_without_feature = run_model_to_get_pred(hook_model,sae,dataset,data_collator,tokenizer,activations_dataset,device,batch_size,proportion_to_evaluate,is_eos,prompt_tuning,loss_type,perturb=feature)
#     #     accuracy_change_relative = (accuracy_perturbed - accuracy) / accuracy
#     #     effect = (probs_pred_perturbed - probs_pred).abs().mean().item()
#     #     overall_accs_change.append(accuracy_change_relative)
#     #     overall_effects.append(effect)
#     #     overall_dict_metrics.append(dict_metrics_without_feature)
#     #     write += 'Feature {}: Effect {}'.format(feature,effect)
#     #     write += '; Relative Accuracy change: {} \n'.format(accuracy_change_relative)
    
#     # print(write)

#     #Temporary
#     overall_accs_change = [0]
#     overall_effects = [0]

#     return dict_metrics_original, overall_accs_change, overall_effects, overall_dict_metrics,only_keep_top10_accuracy_change_relative, only_keep_top10_effect, dict_metrics_only_main, without_selected_accuracy_change_relative, without_selected_effect, dict_metrics_without_selected

# def umap_projection(
#     W_dec_numpy:np.ndarray,
#     top_logits:np.ndarray,
#     bottom_logits:np.ndarray,
#     mean_activations:torch.Tensor,
#     dict_analysis_features:dict,
# ):

#     #2D UMAP plan of the decoder features/rows
#     logger.info(f"Compute the UMAP projection of the features/decoder rows")
#     N = W_dec_numpy.shape[0]

#     labels = dict_analysis_features['prompt label'].numpy()
#     original_activations = dict_analysis_features['original activation'].squeeze(1).numpy()
#     sae_activations = dict_analysis_features['feature activation'].squeeze(1).numpy()
#     unique_labels = np.unique(labels)

#     #Put each vector to norm 1
#     norms = np.linalg.norm(original_activations, axis=1, keepdims=True)
#     original_activations_norm = original_activations / norms

#     #Compute class prototype activations by using test data
#     hidden_size_prototype_class = {}
#     for label in unique_labels:
#         indices = np.where(labels==label)[0]
#         if indices.size > 0: #Is at least one sample is associated to this label
#             hidden_size_prototype_class[label] = np.mean(original_activations_norm[indices],axis=0)


#     #Assign a score to each feature with regard to each class
#     feature_score_class = {}
#     for label in unique_labels:
#         indices = np.where(labels==label)[0]
#         if indices.size > 0: #Is at least one sample is associated to this label
#             feature_score_class[label] = np.sum(sae_activations[indices],axis=0)

#     color_labels = np.arange(unique_labels.size)
#     feature_score_class_array = np.zeros((unique_labels.size,sae_activations.shape[1]))
#     #Concatenate features scores of each class for normalization
#     for i,label in enumerate(unique_labels):
#         feature_score_class_array[i,:] = feature_score_class[label]
#     feature_sums = feature_score_class_array.sum(axis=0)
#     indices_dead_features = (feature_sums==0.)
#     normalized_class_scores = np.zeros((unique_labels.size,feature_sums.size))
#     normalized_class_scores[:,~indices_dead_features] = feature_score_class_array[:,~indices_dead_features] / feature_sums[~indices_dead_features]
#     normalized_class_scores[:,indices_dead_features] = (np.ones(unique_labels.size) / unique_labels.size).reshape(-1,1) 

#     feature_colors = px.colors.qualitative.Plotly[:unique_labels.size]
#     # Compute colors for all points
#     point_colors_indices = np.argmax(normalized_class_scores,axis=0) 
#     colors = [feature_colors[idx] for idx in point_colors_indices]


    
#     reducer = umap.UMAP(n_neighbors=20, min_dist=0.01, output_metric='haversine' ,n_components=2)
#     reducer.fit(original_activations_norm)

#     reduce_embeddings = reducer.transform(W_dec_numpy)

#     #Display the hidden representation prototype for each class
#     np_prototypes = np.array(list(hidden_size_prototype_class.values()))  # Convert dictionary values to a numpy array
#     np_keys_prototypes = list(hidden_size_prototype_class.keys())  # Get dictionary keys for hover text
#     np_prototypes_umap = reducer.transform(np_prototypes)

#     x = np.sin(reduce_embeddings[:, 0]) * np.cos(reduce_embeddings[:, 1])
#     y = np.sin(reduce_embeddings[:, 0]) * np.sin(reduce_embeddings[:, 1])
#     z = np.cos(reduce_embeddings[:, 0])

#     x_prototypes = np.sin(np_prototypes_umap[:, 0]) * np.cos(np_prototypes_umap[:, 1])
#     y_prototypes = np.sin(np_prototypes_umap[:, 0]) * np.sin(np_prototypes_umap[:, 1])
#     z_prototypes = np.cos(np_prototypes_umap[:, 0])
    
#     # UMAP figure parameters
#     # Prepare hover data
#     sizes = mean_activations.numpy()
#     hover_data = [
#         f"Point {i}<br>Top : {', '.join(top_logits[:10,i])}<br>Bottom: {', '.join(bottom_logits[:10,i])}"
#         for i in range(N)
#     ]


#     # Create interactive plots
#     fig_sphere = px.scatter_3d(
#         x=x,
#         y=y,
#         z=z,
#         size=sizes,
#         title='Interactive UMAP Projection'
#     )

#     fig_sphere.update_traces(
#         mode='markers',
#         marker=dict( opacity=0.8, color=colors,line=dict(width=2, color='DarkSlateGrey')),
#         customdata=hover_data,  
#         hovertemplate='<b>%{customdata}</b><extra></extra>'  
#     )

#      #Add class prototypes to the figure
#     fig_sphere.add_trace(go.Scatter3d(
#         x=x_prototypes,
#         y=y_prototypes,
#         z=z_prototypes,
#         mode='markers',
#         marker=dict(size=12, opacity=0.8, color='yellow'),
#         text=np_keys_prototypes,  
#         hoverinfo='text',  
#         name='Class prototypes'  # Label for the legend
#     ))

#     # Customize layout
#     fig_sphere.update_layout(
#         hovermode='closest',
#         title='Interactive UMAP Plot',
#         showlegend=False,
#         dragmode='pan',
#         width=1600, 
#         height=1200  
#     )

#     x = np.arctan2(x, y)
#     y = -np.arccos(z)
#     x_prototypes = np.arctan2(x_prototypes, y_prototypes)
#     y_prototypes = -np.arccos(z_prototypes)
    
#     fig = px.scatter(
#         x=x,
#         y=y,
#         size=sizes,
#         title='Interactive UMAP Projection',
#         labels={'x': 'UMAP Component 1', 'y': 'UMAP Component 2'}
#     )

#     fig.update_traces(
#         mode='markers',
#         marker=dict( opacity=0.8, color=colors,line=dict(width=2, color='DarkSlateGrey')),
#         customdata=hover_data,  
#         hovertemplate='<b>%{customdata}</b><extra></extra>'  
#     )

#     fig.add_trace(go.Scatter(
#         x=x_prototypes,
#         y=y_prototypes,
#         mode='markers',
#         marker=dict(size=12, opacity=0.8, color='yellow'),
#         text=np_keys_prototypes,  
#         hoverinfo='text',  
#         name='Class prototypes'  
#     ))


#     # Customize layout
#     fig.update_layout(
#         hovermode='closest',
#         title='Interactive UMAP Plot',
#         xaxis_title='UMAP Component 1',
#         yaxis_title='UMAP Component 2',
#         showlegend=False,
#         dragmode='pan',
#         width=1600,  
#         height=1200  
#     )

#     return fig, fig_sphere
    
# def pca_activations_projection(
#     W_dec_numpy:np.ndarray,
#     top_logits:np.ndarray,
#     bottom_logits:np.ndarray,
#     mean_activations:torch.Tensor,
#     dict_analysis_features:dict,
# ):
    
#     #2D projection of the decoder features/rows on PCA axis computed with original activations 
#     logger.info(f"Compute the PCA plan of the original activations and projection the features/decoder rows on it")
#     '''PCA in the activation space'''
#     N = W_dec_numpy.shape[0]
    
#     labels = dict_analysis_features['prompt label'].numpy()
#     original_activations = dict_analysis_features['original activation'].squeeze(1).numpy()
#     sae_activations = dict_analysis_features['feature activation'].squeeze(1).numpy()
#     unique_labels = np.unique(labels) #sorted unique elements of the array

#     #For analysis of the originaneurons
#     #sae_activations = original_activations.copy()
    
#     #Put each vector to norm 1
#     norms = np.linalg.norm(original_activations, axis=1, keepdims=True)
#     original_activations_norm = original_activations / norms

#     #Compute class prototype activations by using test data
#     hidden_size_prototype_class = {}
#     for label in unique_labels:
#         indices = np.where(labels==label)[0]
#         if indices.size > 0: #Is at least one sample is associated to this label
#             hidden_size_prototype_class[label] = np.mean(original_activations_norm[indices],axis=0)

#     #Assign a score to each feature with regard to each class
#     feature_score_class = {}
#     for label in unique_labels:
#         indices = np.where(labels==label)[0]
#         if indices.size > 0: #Is at least one sample is associated to this label
#             feature_score_class[label] = np.sum(sae_activations[indices],axis=0)

#     color_labels = np.arange(unique_labels.size)
#     feature_score_class_array = np.zeros((unique_labels.size,sae_activations.shape[1]))
#     #Concatenate features scores of each class for normalization
#     for i,label in enumerate(unique_labels):
#         feature_score_class_array[i,:] = feature_score_class[label]
#     feature_sums = feature_score_class_array.sum(axis=0)
#     indices_dead_features = (feature_sums==0.)
#     normalized_class_scores = np.zeros((unique_labels.size,feature_sums.size))
#     normalized_class_scores[:,~indices_dead_features] = feature_score_class_array[:,~indices_dead_features] / feature_sums[~indices_dead_features]
#     normalized_class_scores[:,indices_dead_features] = (np.ones(unique_labels.size) / unique_labels.size).reshape(-1,1) 

#     feature_colors = px.colors.qualitative.Plotly[:unique_labels.size]
#     # Compute colors for all points
#     point_colors_indices = np.argmax(normalized_class_scores,axis=0) 
#     colors = [feature_colors[idx] for idx in point_colors_indices]

#     # #Load PCA and scaler
#     # with open('./scaler_model_5.pkl', 'rb') as file:
#     #     scaler = pickle.load(file) 

#     # with open('./pca_model_5.pkl', 'rb') as file:
#     #     pca = pickle.load(file) 

#     #On a normalisé les activations avant de faire la PCA dessus car les vecteurs du decoder sont eux-mêmes normalisés
    
#     scaler = StandardScaler()
#     activations_scaled = scaler.fit_transform(original_activations_norm)    
#     pca = PCA(n_components=3)  
#     pca.fit(activations_scaled)

#     # #Save PCA model and scaler model
#     # with open('./pca_model_1.pkl', 'wb') as file:
#     #     pickle.dump(pca, file)

#     # with open('./scaler_model_1.pkl', 'wb') as file:
#     #     pickle.dump(scaler, file)


#     W_dec_features_scaled = scaler.transform(W_dec_numpy)
#     W_dec_pca = pca.transform(W_dec_features_scaled)

#     #Display the hidden representation prototype for each class
#     np_prototypes = np.array(list(hidden_size_prototype_class.values()))  # Convert dictionary values to a NumPy array
#     np_keys_prototypes = list(hidden_size_prototype_class.keys())  # Get dictionary keys for hover text
#     np_prototypes_scaled = scaler.transform(np_prototypes)
#     np_prototypes_pca = pca.transform(np_prototypes_scaled)


#     sizes = mean_activations.numpy()

#     # dict_save_all = {'W_dec_pca' : W_dec_pca,'sizes' : sizes, 'np_keys_prototypes' : np_keys_prototypes, 'np_prototypes_pca' : np_prototypes_pca, 'top_logits' : top_logits, 'bottom_logits' : bottom_logits,  'colors': colors, 'feature_colors' : feature_colors, 'normalized_class_scores' : normalized_class_scores }
#     # np.savez('data_dict_layer5_0.npz', **dict_save_all)

#     ax, fig = design_figure(W_dec_pca, sizes, np_keys_prototypes, np_prototypes_pca, top_logits, bottom_logits, colors, feature_colors, normalized_class_scores,N)  
#     fig.savefig('./layer5_90000.pdf', dpi=300, bbox_inches='tight')
    
#     hover_data = [
#         f"Point {i}<br>Top : {', '.join(top_logits[:10,i])}<br>Bottom: {', '.join(bottom_logits[:10,i])}"
#         for i in range(N)
#     ]
#     fig_pca = px.scatter(
#         x=W_dec_pca[:,0],
#         y=W_dec_pca[:,1],
#         size=sizes,
#         title='Interactive PCA Projection',
#         labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
#     )

#     fig_pca.update_traces(
#         mode='markers',
#         marker=dict( opacity=0.8, color=colors,line=dict(width=2)),
#         customdata=hover_data,  # Use customdata for hover text
#         hovertemplate='<b>%{customdata}</b><extra></extra>'  # Format the custom data for hover
#     )

#     #Add class prototypes to the figure
#     fig_pca.add_trace(go.Scatter(
#         x=np_prototypes_pca[:, 0],
#         y=np_prototypes_pca[:, 1],
#         mode='markers',
#         marker=dict(size=12, opacity=0.8, color='yellow'),
#         text=np_keys_prototypes,  # Use text for hover text
#         hoverinfo='text', 
#         name='Class prototypes'  
#     ))

#     # Customize layout
#     fig_pca.update_layout(
#         hovermode='closest',
#         title='Interactive PCA Plot',
#         xaxis_title='PCA Component 1',
#         yaxis_title='PCA Component 2',
#         showlegend=False,
#         dragmode='pan',
#         width=1600,
#         height=1200 
#     )

#     #Same but with the third component
#     fig_pca_3d = px.scatter_3d(
#         x=W_dec_pca[:,0],
#         y=W_dec_pca[:,1],
#         z=W_dec_pca[:,2],
#         size=sizes,
#         title='Interactive PCA Projection',
#         labels={'x': 'PCA Component 1', 'y': 'PCA Component 2','z': 'PCA Component 3'}
#     )

#     fig_pca_3d.update_traces(
#         mode='markers',
#         marker=dict( opacity=0.8, color=colors,line=dict(width=2, color='DarkSlateGrey')),
#         customdata=hover_data, 
#         hovertemplate='<b>%{customdata}</b><extra></extra>' 
#     )

#     #Add class prototypes to the figure
#     fig_pca_3d.add_trace(go.Scatter3d(
#         x=np_prototypes_pca[:, 0],
#         y=np_prototypes_pca[:, 1],
#         z=np_prototypes_pca[:, 2],
#         mode='markers',
#         marker=dict(size=12, opacity=0.8, color='yellow'),
#         text=np_keys_prototypes, 
#         hoverinfo='text',  
#         name='Class prototypes'  
#     ))

#     # Customize layout
#     fig_pca_3d.update_layout(
#         hovermode='closest',
#         title='Interactive PCA Plot',
#         showlegend=True,
#         dragmode='pan',
#         width=1600, 
#         height=1200 
#     )

#     return fig_pca, fig_pca_3d, normalized_class_scores


# def analyze_features(
#     tune_sae:HookedTransformer,
#     hook_model:HookedTransformer,
#     mean_activations: torch.Tensor,
#     dict_analysis_features: dict,
# ):
    
#     logger.info(f"Compute the top 10 tokens for each feature/decoder row")
#     #Get the top 10 Logit Weights/tokens for each feature/decoder row
#     print(f"Shape of the decoder weights {tune_sae.W_dec.shape})")
#     print(f"Shape of the model unembed {hook_model.W_U.shape}")
#     projection_matrix = tune_sae.W_dec @ hook_model.W_U.to(dtype=torch.float32) #We convert here in case the weights are quantized
#     print(f"Shape of the projection matrix {projection_matrix.shape}")
#     # then we take the top_k tokens per feature and decode them
#     top_k = 30
#     N = tune_sae.W_dec.shape[0]
#     _, top_k_tokens = torch.topk(projection_matrix, top_k, dim=1,sorted=True) #decreasing order of values
#     _, bottom_k_tokens = torch.topk(projection_matrix, top_k, dim=1,sorted=True,largest=False) #ascedning order of values

#     feature_df_columns = pd.MultiIndex.from_tuples(
#     [(f'feature {i}', 'top') for i in range(N)] +
#     [(f'feature {i}', 'bottom') for i in range(N)],
#     names=['Feature', 'Positive/Negative']
#     )

#     feature_df = pd.DataFrame(index=range(top_k),columns=feature_df_columns)
#     # feature_df = pd.DataFrame(top_k_tokens.cpu().numpy())
#     # Fill the DataFrame
#     for i in range(N):
#         feature_df[(f'feature {i}', 'top')] = top_k_tokens[i].cpu().numpy()
#         feature_df[(f'feature {i}', 'bottom')] = bottom_k_tokens[i].cpu().numpy()

#     feature_df.index = [f"token_{i}" for i in range(top_k)]
#     feature_df = feature_df.map(lambda x: hook_model.tokenizer.convert_ids_to_tokens(x))
#     #Special case to prevvent too long useless tokens
#     feature_df = feature_df.map(lambda x: 'ĠĠ' if (x is None or 'ĠĠ' in x) else x)
#     #Remove the 'Ġ' at the beggining of the token
#     feature_df = feature_df.map(lambda x: x[1:] if (x[0] == 'Ġ') else x)

#     feature_df = feature_df.reindex(columns=pd.MultiIndex.from_tuples(
#         [(f'feature {i}', sub) for i in range(N) for sub in ['top', 'bottom']],
#         names=['Feature', 'Positive/Negative']
#     ))

#     W_dec_numpy = tune_sae.W_dec.detach().cpu().numpy()
#     top_logits = feature_df.xs('top', axis=1, level='Positive/Negative').values.astype(str)
#     bottom_logits = feature_df.xs('bottom', axis=1, level='Positive/Negative').values.astype(str)

#     #fig_umap, fig_umap_sphere = umap_projection(W_dec_numpy,top_logits,bottom_logits,mean_activations,dict_analysis_features)
#     fig_umap, fig_umap_sphere = None, None
    
#     fig_pca, fig_pca_3d, normalized_class_scores = pca_activations_projection(W_dec_numpy,top_logits,bottom_logits,mean_activations,dict_analysis_features)

#     #Top p actived features per F_c
#     nb_labels = normalized_class_scores.shape[0]
#     top_indice = np.argmax(normalized_class_scores,axis=0)
#     j_select_list = []
#     values_select_list = []
    
#     p=1
    
#     j_select_tensor = torch.zeros((nb_labels,p),dtype=torch.int)
#     values_select_tensor = torch.zeros((nb_labels,p))

    
#     for c in range(nb_labels):
#         features_most_related_to_c =  torch.from_numpy( np.where(top_indice==c)[0] )
#         top_mean_activations_values, top_p_indices  = torch.topk(mean_activations[features_most_related_to_c],k=p)
#         #Map the top_p indices back to the original tensor 'mean_activations'
#         j_select_c = features_most_related_to_c[top_p_indices]
#         j_select_tensor[c,:] = j_select_c
#         values_select_tensor[c,:] = top_mean_activations_values


#     torch.save(j_select_tensor,'j_select_tensor.pt')
#     torch.save(values_select_tensor,'values_select_tensor.pt')

#     #Extract only the logit columns of the features we are interested in
#     j_select_tensor_flatten = j_select_tensor.flatten()
#     columns_to_extract = [f"feature {i}" for i in j_select_tensor_flatten]
#     sub_feature_df = feature_df[columns_to_extract]
    
#     return sub_feature_df, fig_umap, fig_umap_sphere, fig_pca, fig_pca_3d, j_select_tensor

    


# def main_sae_evaluation(
#     config_model: str,
#     config_sae: str):
    
#     #Retrieve the config of the model, dataset and tokenizer
#     cfg_model = LLMLoadConfig.autoconfig(config_model)
    
#     #Retrieve the config of the SAE
#     cfg_sae = SAELoadConfig.autoconfig(config_sae)
    
#     #Load our local tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
#     tokenizer.pad_token = (
#         tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
#     )
#     tokenizer.padding_side = "left"
#     tokenizer.truncation_side = "left"

#     #We retrieve the length of the preprompt added to the model if we specified it
#     len_example = cfg_model.len_example
#     #We retrieve the length of the template added at the end of the sentence
#     len_template = cfg_model.len_template
    
#     #Process the dataset on which we will do the forward passes
#     dataset_tokenized = process_dataset(cfg_model,split="test",tokenizer=tokenizer) 
#     #dataset_tokenized = process_dataset(cfg_model,split="train",tokenizer=tokenizer) 


    
    
#     #Get model hooked (HookedTransformer)
#     cfg_model.task_args['prompt_tuning'] = cfg_sae.evaluation_args['prompt_tuning']
#     hook_model = get_hook_model(cfg_model,tokenizer)

#     if cfg_sae.evaluation_args['prompt_tuning']:
#         logger.info("Using prompt tuning for evaluation inference on the model")
#         hook_model.load_state_dict(torch.load('prompt_tuning/prompt_tuning_llama_32_instruct.pth'))
    
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
#     #Load the local trained SAE
#     tune_sae = TrainingSAE.load_from_pretrained(cfg_sae.sae_path,cfg_sae.evaluation_args['device'])
    
    
#     if cfg_sae.evaluation_args['return_feature_activations']:
#         classification_loss_dict, histogram_sparsity, mean_activations,dict_analysis_features, text_used, activations_dataset = eval_hook_loss(hook_model,
#                                                                                                                             tune_sae,
#                                                                                                                             dataset_tokenized,
#                                                                                                                             data_collator,
#                                                                                                                             tokenizer,
#                                                                                                                             len_example,
#                                                                                                                             len_template,
#                                                                                                                             **cfg_sae.evaluation_args)

        

#         #Create the directory where to save the activations if it does not already exist
#         dir_to_save_sae_activations = os.path.join(cfg_sae.dir_to_save_activations,cfg_sae.sae_name)
#         if not os.path.exists(dir_to_save_sae_activations):
#             os.makedirs(dir_to_save_sae_activations)
        
#         #Save feature activations, ground truth labels, used prompts and model output logits
#         file_to_save_sae_activations = os.path.join(dir_to_save_sae_activations,f"{cfg_model.dataset_name}.pth")
#         #print(f'file where activations are stored : {file_to_save_sae_activations}')
#         torch.save(dict_analysis_features, file_to_save_sae_activations)

#         file_to_save_text_used = os.path.join(dir_to_save_sae_activations,f"{cfg_model.dataset_name}.json")
#         with open(file_to_save_text_used, 'w') as file:
#             json.dump(text_used, file, indent=4)
        
#     else:
#         classification_loss_dict, histogram_sparsity, mean_activations, activations_dataset = eval_hook_loss(hook_model,
#                                                                                         tune_sae,
#                                                                                         dataset_tokenized,
#                                                                                         data_collator,
#                                                                                         tokenizer,
#                                                                                         len_example,
#                                                                                         len_template,
#                                                                                         **cfg_sae.evaluation_args)


#     #text_used : list


#     #Create the directory where to save the sae metrics if it does not already exist
#     dir_to_save_sae_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name)
#     if not os.path.exists(dir_to_save_sae_metrics):
#         os.makedirs(dir_to_save_sae_metrics)

#     '''For each SAE feature (decoder row), we compute the 10 tokens the most promoted by the feature 
#     thanks to Logit Lens. We also compute a 2D representation of the features with UMAP'''
#     feature_df, fig_umap, fig_umap_sphere, fig_pca, fig_pca_3d, j_select_tensor = analyze_features(tune_sae,hook_model,mean_activations,dict_analysis_features)

    

#     #Interpretability part
#     # #Get the sentences which activate the most the selected features
#     sae_activations = dict_analysis_features['feature activation'].squeeze(1)

#     #Compute the cosine similarity between the most activated features by class
#     most_activated_features_by_class = sae_activations[:,j_select_tensor.view(-1)]
#     display_cosine_similarity_stats(most_activated_features_by_class,j_select_tensor.view(-1))
    
#     directory_to_save_texts = "./results/top_texts/" 
#     directory_to_save_csv = "./results/top_texts_csv/"
#     for c in range(4):
#         features_to_inspect = j_select_tensor[c]
    
#         dict_list_texts = {}
#         for pos,feature in enumerate(features_to_inspect):
#             activations_inspected = sae_activations[:,feature]
#             top_values, top_indices = torch.topk(activations_inspected,10)
#             top_texts = [text_used[i] for i in top_indices]

#             n = activations_inspected.size(0)
#             active_samples = torch.sum(activations_inspected > 0)
#             percentage_activity = (active_samples / n) * 100

#             if pos==0:
#                 file_mode = "w"
#             else:
#                 file_mode = "a"
            
#             with open(os.path.join(directory_to_save_texts,f'category_{c}.txt'),file_mode) as file:
#                 file.write(f'\n####### Feature {feature} ##########\n\n')

#                 # #Write top logits
#                 # for token_logit in feature_df[f'feature {feature}']['top']:
#                 #     file.write(f"{token_logit}\n")

#                 file.write("\n")
                
#                 for text, value in zip(top_texts, top_values):
#                     # Write each text and value on a new line
#                     file.write(f"\n{text}\t{value}\n")

#             # CSV writing
#             with open(os.path.join(directory_to_save_csv,f'category_{c}.csv'), mode=file_mode, newline='', encoding='utf-8') as file:
               
#                 writer = csv.writer(file)
                
#                 # Write the header row
#                 if file_mode=="w":
#                     writer.writerow(['sentence', 'feature_id', 'sparsity', 'class'])
                
#                 # Write the data rows
#                 for sentence in top_texts:
#                     writer.writerow([sentence, feature.item(), percentage_activity.item(), c])
            

            
#             dict_list_texts[f'top_text_by_{feature}'] = top_texts
#             dict_list_texts[f'top_values_by_{feature}'] = top_values.tolist()
    
#         with open(f'./results/top_texts/category_{c}.json', 'w') as json_file:
#             json.dump(dict_list_texts, json_file, indent=4)

#         #Generate n*10 random sentences within the class
            
#         labels_to_filter = dict_analysis_features['prompt label']
#         indices = (labels_to_filter == c).nonzero(as_tuple=True)[0].tolist()
#         sampled_indices = random.sample(indices, 10*features_to_inspect.shape[0])
#         random_text = [text_used[i] for i in sampled_indices]

#         # CSV writing
#         with open(os.path.join(directory_to_save_csv,f'category_{c}_random.csv'), mode="w", newline='', encoding='utf-8') as file:
           
#             writer = csv.writer(file)
            
#             # Write the header row
#             writer.writerow(['sentence', 'class'])
            
#             # Write the data rows
#             for sentence in random_text:
#                 writer.writerow([sentence, c])
        

    
    
#     #We evaluate the causality of the most prominent features (by mean activation)
#     if cfg_sae.causality:
#         #First, we pre-select the most activated features in average
#         #_,top_features_mean_activated_avg = torch.topk(mean_activations,cfg_sae.topk_mean_activation)
#         # overall_accs_change, overall_effects, only_keep_top10_accuracy_change, only_keep_top10_effect = eval_causal_effect_model(top_features_mean_activated_avg,hook_model,tune_sae,dataset_tokenized,data_collator,tokenizer,**cfg_sae.evaluation_args)


#         p_to_select = [1]

#         for p in p_to_select:
        
#             global_mean_acc_change = 0
#             global_mean_effect = 0
#             global_mean_acc_without_selected = 0
#             global_mean_effect_without_selected = 0

#             j_select_tensor_restrict = j_select_tensor[:,:p]
            
#             #We do it by class
#             for c in range(j_select_tensor.shape[0]):
#                 list_ablated_features = [int(feature_number) for feature_number in j_select_tensor_restrict[c]]
    
#                 dict_metrics_original, overall_accs_change, overall_effects, overall_dict_metrics,only_keep_top10_accuracy_change_relative, only_keep_top10_effect, dict_metrics_only_main, without_selected_accuracy_change_relative, without_selected_effect, dict_metrics_without_selected = eval_causal_effect_model(c,j_select_tensor_restrict[c],hook_model,tune_sae,dataset_tokenized,data_collator,tokenizer,activations_dataset,**cfg_sae.evaluation_args)
    
#                 mean_accs_change = sum(overall_accs_change) / len(overall_accs_change)
#                 mean_overall_effects = sum(overall_effects) / len(overall_effects)
#                 global_mean_acc_change+=mean_accs_change
#                 global_mean_effect+=mean_overall_effects
#                 global_mean_acc_without_selected += without_selected_accuracy_change_relative
#                 global_mean_effect_without_selected += without_selected_effect
    
        
#                 #list_ablated_features = [int(feature_number) for feature_number in top_features_mean_activated_avg]
                
#                 dict_causal_metrics = {'Selected features' : list_ablated_features, 'Ablation all expect the selected features : impact' : only_keep_top10_effect, 'Ablation all expect the selected features : realtive accuracy change' : only_keep_top10_accuracy_change_relative, 'Ablation on all the selected features : impact' : without_selected_effect, 'Ablation on all the selected features : realtive accuracy change' : without_selected_accuracy_change_relative,'Ablation relative accuracy change' : overall_accs_change, 'Ablation impact' : overall_effects,  'Ablation mean accuracy change' : mean_accs_change, 'Ablation mean impact' : mean_overall_effects}
    
                
#                 if c==0:
#                     file_to_save_original_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_original_sae.json") 
#                     with open(file_to_save_original_metrics, 'w') as file:
#                         json.dump(dict_metrics_original, file, indent=4)
                
#                 file_to_save_causal_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_p_{p}_causal_{c}.json") 
#                 with open(file_to_save_causal_metrics, 'w') as file:
#                     json.dump(dict_causal_metrics, file, indent=4)
                
#                 # #Save the more detailled dictionaries
#                 # file_to_save_causal_metrics_only = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_causal_keep_only_p_{p}_class_{c}.json") 
#                 # with open(file_to_save_causal_metrics_only, 'w') as file:
#                 #     json.dump(dict_metrics_only_main, file, indent=4)
    
#                 file_to_save_causal_metrics_without = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_causal_without_p_{p}_class_{c}.json") 
#                 with open(file_to_save_causal_metrics_without, 'w') as file:
#                     json.dump(dict_metrics_without_selected, file, indent=4)
    
#                 # for l,feature_number in enumerate(list_ablated_features):
#                 #     file_to_save_causal_metrics_feature = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_causal_without_{feature_number}.json") 
#                 #     with open(file_to_save_causal_metrics_feature, 'w') as file:
#                 #         json.dump(overall_dict_metrics[l], file, indent=4)
    
            
    
#             global_mean_acc_change/=j_select_tensor.shape[0]
#             global_mean_effect/=j_select_tensor.shape[0]
#             print(f"Global mean accuracy change over all features (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_acc_change}")
#             print(f"Global mean effect over all features (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_effect}")
#             global_mean_acc_without_selected/=j_select_tensor.shape[0]
#             global_mean_effect_without_selected/=j_select_tensor.shape[0]
#             print(f"Global mean accuracy change over all the clusters (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_acc_without_selected}")
#             print(f"Global mean effect over all the clusters (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_effect_without_selected}")
        
#     #Save sparsity histogram 
#     file_to_save_histogram = os.path.join(dir_to_save_sae_metrics,f"{cfg_model.dataset_name}.png")
#     pio.write_image(histogram_sparsity, file_to_save_histogram)
    
#     #Save SAE metrics performance dict as json
#     file_to_save_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}.json") 
#     with open(file_to_save_metrics, 'w') as file:
#         json.dump(classification_loss_dict, file, indent=4)



#     #Create the directory where to save the top logits and UMAP plan if it does not already exist
#     dir_to_save_top_logits = cfg_sae.dir_to_save_top_logits 
#     if not os.path.exists(dir_to_save_top_logits):
#         os.makedirs(dir_to_save_top_logits)
        
    
#     #Save top logits for each feature
#     file_to_save_top_logits = os.path.join(dir_to_save_top_logits,f"{cfg_sae.sae_name}.csv")
#     feature_df.to_csv(file_to_save_top_logits,index=False,encoding='utf-8')
#     #Save the UMAP plots
#     # file_to_save_umap_plot = os.path.join(dir_to_save_top_logits,f"UMAP_{cfg_sae.sae_name}.html")
#     # fig_umap.write_html(file_to_save_umap_plot,config={'scrollZoom': True})
#     # file_to_save_umap_plot_sphere = os.path.join(dir_to_save_top_logits,f"UMAP_3d_SPHERE{cfg_sae.sae_name}.html")
#     # fig_umap_sphere.write_html(file_to_save_umap_plot_sphere,config={'scrollZoom': True})
    
#     #Save PCA plot
#     file_to_save_pca_plot = os.path.join(dir_to_save_top_logits,f"PCA_{cfg_sae.sae_name}.html")
#     fig_pca.write_html(file_to_save_pca_plot,config={'scrollZoom': True})
#     file_to_save_3d_pca_plot = os.path.join(dir_to_save_top_logits,f"PCA_3d{cfg_sae.sae_name}.html")
#     fig_pca_3d.write_html(file_to_save_3d_pca_plot,config={'scrollZoom': True})



from ..utils import LLMLoadConfig, SAELoadConfig
from ..model_training import process_dataset,get_hook_model,compute_loss_last_token, PromptTunerForHookedTransformer

import sys
sys.path.append("../../")

from sae_implementation import TrainingSAE

import torch
from safetensors import safe_open
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from functools import partial
import numpy as np
import plotly.express as px
import plotly.io as pio
import os
import csv
import random
import json
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from tqdm import tqdm
from loguru import logger
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge


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


def display_cosine_similarity_stats(data : torch.tensor, features_number : torch.tensor):
    #data : shape (n_ind, features)

    n,d = data.shape 
    
    # Normalize the vectors along dimension 0 (for cosine similarity calculation)
    column_norms = data.norm(dim=0, keepdim=True)
    column_norms[column_norms == 0] = 1e-8
    normalized_tensor = data / column_norms
    
    
    # Compute cosine similarity matrix (d x d)
    cosine_similarity_matrix = torch.mm(normalized_tensor.t(), normalized_tensor)
    
    # Extract unique pairs by masking the upper triangular matrix (excluding diagonal)
    i, j = torch.triu_indices(d, d, offset=1)
    cosine_values = cosine_similarity_matrix[i, j]
    
    # Compute statistics
    mean_cosine_similarity = cosine_values.mean().item()
    variance_cosine_similarity = cosine_values.var().item()

    # Find the top 3 pairs with the highest cosine similarity
    top_k = 3
    top_values, top_indices = torch.topk(cosine_values, top_k, largest=True)

    # Find the pairs corresponding to the top similarities
    top_pairs = [(features_number[i[idx].item()], features_number[j[idx].item()]) for idx in top_indices]
    
    
    # Display results
    print(f"Mean cosine similarity: {mean_cosine_similarity}")
    print(f"Variance of cosine similarity: {variance_cosine_similarity}")

    for rank, (value, pair) in enumerate(zip(top_values, top_pairs), 1):
        print(f"Top {rank} cosine similarity: {value.item()} (between vectors {pair})")

    max_activity = 0
    mean_activity = 0
    #Save the distribution of the feature
    for i,feature_number in enumerate(features_number):

        active_samples = torch.sum(data[:,i] > 0)
        percentage_activity = (active_samples / n) * 100
        mean_activity += percentage_activity
        if percentage_activity > max_activity:
            max_activity = percentage_activity
        
        # plt.figure(figsize=(8, 6))
        # plt.hist(data[:,i], bins=50, edgecolor='black', alpha=0.7,density=True)
        # plt.title(f"Feature {feature_number} : Activation {percentage_activity}%")
        # plt.xlabel("Value")
        # plt.ylabel("Frequency")
    
        # # Save the plot
        # plt.savefig(f"results/top_texts/{feature_number}.pdf")
        # plt.close()

    mean_activity /= len(features_number)
    print(f"Mean activity of the selected features in the segments : {mean_activity:.2f}%")
    print(f"Max activity of feature among the selected features in the segments : {max_activity:.2f}%")



def cosine_similarity_concepts(concepts_matrix,threshold):

    #concept_matrix shape : (embedding_dim, n_concepts)
    n,d = concepts_matrix.shape 
    
    # Normalize the vectors along dimension 0 (for cosine similarity calculation)
    column_norms = concepts_matrix.norm(dim=0, keepdim=True)
    column_norms[column_norms == 0] = 1e-8
    normalized_tensor = concepts_matrix / column_norms

    
    # Compute cosine similarity matrix (d x d)
    cosine_similarity_matrix = torch.mm(normalized_tensor.t(), normalized_tensor)
    print(f"cosine_similarity_matrix shape : {cosine_similarity_matrix.shape}")
    
    # Extract unique pairs by masking the upper triangular matrix (excluding diagonal)
    i, j = torch.triu_indices(d, d, offset=1)
    cosine_values = cosine_similarity_matrix[i, j]
    print(f"cosine_values shape : {cosine_values.shape}")

    # Compute statistics
    mean_cosine_similarity = cosine_values.mean().item()
    variance_cosine_similarity = cosine_values.var().item()

    count = torch.sum(cosine_values > threshold).item()

    print(f"There are {count} pairs of concepts with a cosine similarity higher than {threshold}")
    
    
def knn_distance_metric(embeddings: torch.Tensor, k: int = 5) -> float:
    """
    Computes the mean distance to the k-nearest neighbors (k-NN) for each embedding.

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


    
# def compute_feature_cosine_similarity(data,mean_activations):
#     """
#     Computes the mean and max cosine similarity between features in the given data tensor.

#     Parameters:
#     - data: numpy array of shape (n, 1, f), where n is the number of individuals and f is the number of features.

#     Returns:
#     - mean_cosine_similarity: Mean cosine similarity between different features.
#     - max_cosine_similarity: Maximum cosine similarity between different features.
#     """
#     n, _, f = data.shape

#     # Reshape the tensor to (n, f)
#     X = data.reshape(n, f)

#     # Transpose X so that features are along the rows
#     X_T = X.T  # Now X_T is of shape (f, n)

#     # Compute dot products between feature vectors
#     dot_product_matrix = np.dot(X_T, X_T.T)  # Shape: (f, f)

#     # Compute the norm (magnitude) of each feature vector
#     feature_norms = np.linalg.norm(X_T, axis=1)  # Shape: (f,)
    

#     # Compute the outer product of norms to get denominator for cosine similarity
#     norm_matrix = np.outer(feature_norms, feature_norms)  # Shape: (f, f)


#     # Avoid division by zero
#     epsilon = 1e-10
#     norm_matrix[norm_matrix == 0] = epsilon

#     # Compute the cosine similarity matrix
#     cosine_similarity_matrix = dot_product_matrix / norm_matrix

#     #Compute weights (outer product of mean activations)
#     weights = np.outer(mean_activations, mean_activations)
#     print(f"weights shape : {weights.shape}")

#     weighted_cosine_similarity_matrix = weights * cosine_similarity_matrix

#     # Create a mask to exclude diagonal elements
#     mask = ~np.eye(f, dtype=bool)

#     # Extract the cosine similarity values excluding the diagonal
#     cosine_values = weighted_cosine_similarity_matrix[mask]
    
#     print(f"cosine_similarity_matrix shape : {cosine_similarity_matrix.shape}")

#     # Compute mean and max cosine similarity
#     mean_cosine_similarity = np.mean(cosine_values)
#     max_cosine_similarity = np.max(cosine_values)

#     return mean_cosine_similarity, max_cosine_similarity

def design_figure(W_dec_pca, sizes, np_keys_prototypes, np_prototypes_pca, top_logits, bottom_logits, colors, feature_colors, normalized_class_scores,N):

  feature_colors = np.array(['#636EFA','#EF553B','#00CC96','#3D2B1F'])
  #feature_colors = np.array(['#636EFA','#EF553B'])
  # Extracting the x and y components of the vectors
  x = W_dec_pca[:, 0]
  y = W_dec_pca[:, 1]


  # Create the figure and axis
  fig, ax = plt.subplots(figsize=(8, 8))

  labels_names = np.array(['World','Sport', 'Business','Sci/Tech'])
  #labels_names = np.array(['Non Toxic','Toxic'])

  # Plot the special points with yellow triangles
  ax.scatter(np_prototypes_pca[:,0], np_prototypes_pca[:,1], color='orange', marker='^', s=400, zorder=6)

  # Add placards with labels for special points
  for i, label in enumerate(labels_names):
      ax.text(np_prototypes_pca[:,0][i] - 4.5, np_prototypes_pca[:,1][i] + 0.1, label, fontsize=15,bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'),zorder=100)

  # Add grid, labels, and title
  ax.axhline(0, color='grey', lw=1,zorder=10)
  ax.axvline(0, color='grey', lw=1,zorder=10)
  ax.grid(True, linestyle='--', alpha=0.5,zorder=10)

  # Function to create a pie chart marker at a specific location
  # def draw_pie(ax,center, sizes, colors, radius=0.5):
  #     # Starting angle
  #     start_angle = 0

  #     # Iterate through sizes and corresponding colors to draw pie slices
  #     for size, color in zip(sizes, colors):
  #         # print(size)
  #         # print(color)
  #         # print(center)
  #         # Draw a wedge (slice of the pie)
  #         wedge = ax.pie(
  #             [size, 1 - size],
  #             colors=[color, 'none'],  # second color is transparent to create the effect of a single slice
  #             startangle=start_angle,
  #             radius=radius,
  #             center=center
  #         )
  #         # Update the start angle
  #         start_angle += size * 360


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


  #plt.scatter(x, y, s=sizes**2,color=colors, zorder=5)
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




def decode_dataset(tokenized_dataset,tokenizer):
    
    original_texts = []
    
    for input_ids in tokenized_dataset['input_ids']:
        # Decoding the token IDs back to text
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        original_texts.append(text)

    return original_texts


def reconstr_hook(activation, hook, sae_out):
    return sae_out


def reconstr_hook_classification_token(activation, hook, sae_out):
    n,m,d = activation.shape
    sae_activations = sae_out.squeeze(1).view(n,m,d)
    return sae_activations


def reconstr_hook_classification_token_single_element(activation, hook, sae_out):
    n,m,d = activation.shape
    return sae_out


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


def compute_same_predictions(
    original_logits : torch.Tensor,
    reconstruction_logits : torch.Tensor,
    is_eos: bool
):

    _,vocab_size = original_logits.size()
    
    y_pred_original = original_logits.argmax(dim=1)
    y_pred_reconstruction = reconstruction_logits.argmax(dim=1)

    return (y_pred_original==y_pred_reconstruction).sum()


def compute_metrics_variations(
    dict_metrics_original : dict,
    dict_metrics_reconstruction : dict,
    accuracy_original : float,
    accuracy_reconstruction : float,
    labels_tokens_id : dict
):
     dict_metrics_variation = {}
    
     for key in labels_tokens_id.keys():
      dict_metrics_original[f'recall_{key}'] = dict_metrics_original[f'true matches_{key}'] / dict_metrics_original[f'number real samples_{key}']
      dict_metrics_original[f'precision_{key}'] = 0 if  dict_metrics_original[f'number predicted samples_{key}']==0  else dict_metrics_original[f'true matches_{key}'] / dict_metrics_original[f'number predicted samples_{key}']
      dict_metrics_original[f'f1-score_{key}'] = 0 if (dict_metrics_original[f'recall_{key}'] + dict_metrics_original[f'precision_{key}'])==0  else 2 * dict_metrics_original[f'recall_{key}'] * dict_metrics_original[f'precision_{key}'] / (dict_metrics_original[f'recall_{key}'] + dict_metrics_original[f'precision_{key}'])
      
      dict_metrics_reconstruction[f'recall_{key}'] = dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number real samples_{key}']
      dict_metrics_reconstruction[f'precision_{key}'] = 0 if dict_metrics_reconstruction[f'number predicted samples_{key}']==0 else dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number predicted samples_{key}']
      dict_metrics_reconstruction[f'f1-score_{key}'] = 0 if (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])==0 else 2 * dict_metrics_reconstruction[f'recall_{key}'] * dict_metrics_reconstruction[f'precision_{key}'] / (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])

      dict_metrics_variation[f'Delta recall_{key}'] = dict_metrics_reconstruction[f'recall_{key}'] - dict_metrics_original[f'recall_{key}']
      dict_metrics_variation[f'Delta precision_{key}'] = dict_metrics_reconstruction[f'precision_{key}'] - dict_metrics_original[f'precision_{key}']
      dict_metrics_variation[f'Delta f1-score_{key}'] = dict_metrics_reconstruction[f'f1-score_{key}'] - dict_metrics_original[f'f1-score_{key}']

     dict_metrics_original['Delta Global accuracy'] = accuracy_original
     dict_metrics_reconstruction['Delta Global accuracy'] = accuracy_reconstruction
     dict_metrics_variation['Delta Global accuracy'] = accuracy_reconstruction - accuracy_original
    
     return dict_metrics_variation, dict_metrics_original


def compute_metrics_details(
    dict_metrics_reconstruction : dict,
    labels_tokens_id : dict
):
     
     for key in labels_tokens_id.keys():
      
      dict_metrics_reconstruction[f'recall_{key}'] = dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number real samples_{key}']
      dict_metrics_reconstruction[f'precision_{key}'] = 0 if dict_metrics_reconstruction[f'number predicted samples_{key}']==0 else dict_metrics_reconstruction[f'true matches_{key}'] / dict_metrics_reconstruction[f'number predicted samples_{key}']
      dict_metrics_reconstruction[f'f1-score_{key}'] = 0 if (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])==0 else 2 * dict_metrics_reconstruction[f'recall_{key}'] * dict_metrics_reconstruction[f'precision_{key}'] / (dict_metrics_reconstruction[f'recall_{key}'] + dict_metrics_reconstruction[f'precision_{key}'])

    
    
def eval_hook_loss(
    hook_model:HookedTransformer,
    sae:HookedTransformer,
    labels_dataset,
    original_text_used,
    tokenizer,
    activations_dataset,
    len_example,
    len_template,
    device,
    return_feature_activations=False,
    batch_size=1,proportion_to_evaluate=1.,is_eos=True,prompt_tuning=False,loss_type='cross_entropy'):
    
    
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
    sae_reconstruction_total_loss = 0
    
    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    bs = next(iter(activations_dataloader))["input_ids"].shape[0]
    
    #Save the different activations to speed the causality calculations done after
    new_activations_dataset = ActivationDataset()

    # Evaluation loop
    hook_model.eval()
    sae.eval()
    with torch.no_grad():

        store_l0_activations = []
        
        if return_feature_activations:
            feature_activations_list = []
            acts_without_process_list = []
            original_activation_list = []
            prompt_labels_list = []
            model_logits_labels = {}
        
        
        for batch in tqdm(activations_dataloader, desc="Forward Passes of the given LLM model", unit="batch"):

            input_ids = batch['input_ids'].to(device)
            cache = batch["cache"].to(device)
            original_output = batch["output"].to(device)
            labels = batch["label"]
            attention_mask = batch['attention_mask'].to(dtype=int).to(device)
            
            

            if prompt_tuning:
                batch_size = input_ids.size(0)
                prompt_token_ids = torch.full(
                    (batch_size, hook_model.num_prompt_tokens),
                    hook_model.tokenizer.eos_token_id,  # Using EOS token as a placeholder
                    dtype=torch.long,
                    device=input_ids.device,
                )
                input_ids = torch.cat([prompt_token_ids, input_ids], dim=1)
                prompt_attention_mask = torch.full(
                    (batch_size, hook_model.num_prompt_tokens),
                    1,  # Using EOS token as a placeholder
                    dtype=torch.long,
                    device=input_ids.device,
                )
                attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)            
            

            #Manually fixing the issue of sentence longer than the context windows size since it is not automatically handled by transformer-lens and it causes conflict with the positional embedding that it dones on a vector of size the context attention windows
            if input_ids.shape[1] > hook_model.cfg.n_ctx:
                attention_mask = attention_mask[:,-hook_model.cfg.n_ctx:]
                input_ids = input_ids[:,-hook_model.cfg.n_ctx:] #It should not be an issue since we want to predict for almost the last token, so it looses almost nothing of the context it could have seen otherwise


            a,c = cache.shape
            #cache_flatten = cache.view(a*b,-1).unsqueeze(1)
            cache_sentence = cache.unsqueeze(1)
    
            # Use the SAE
            feature_acts, acts_without_process = sae.encode_with_hidden_pre(cache_sentence)
            sae_out = sae.decode(feature_acts)
            
            # feature_acts_all, acts_without_process_all = sae.encode_with_hidden_pre(cache_flatten)
            # sae_out_all = sae.decode(feature_acts_all)

            #Save the activations and labels for the causality calculations done after
            bs = input_ids.shape[0]
            d_in = cache.shape[-1]
            inputs_to_save = input_ids.cpu()
            cache_to_save = cache.cpu()
            original_output_to_save = original_output.cpu()  #shape : [batch size, vocab size]
            labels_to_save = labels
            attention_mask_to_save = attention_mask.cpu()
            sae_activations_to_save = feature_acts.squeeze(1).cpu()
            new_activations_dataset.append(inputs_to_save, cache_to_save, original_output_to_save, labels_to_save, attention_mask_to_save, sae_activations_to_save)
            

            #If in addition to the SAE metrics, we want to store feature activations and model predictions
            if return_feature_activations:
                feature_activations_list.append(feature_acts.cpu())
                acts_without_process_list.append(acts_without_process.cpu())
                original_activation_list.append(cache_sentence.cpu())
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
               
                    
        
            epsilon = 1e-4
            #Store L0 sparsity on the batch
            l0 = (feature_acts > epsilon).float().sum(-1).detach() #l0 of size [len_batch,max_length] so we have the l0 for each hidden state/token of each sentence in the batch
            l0_batch = l0.flatten().cpu().numpy()
            store_l0_activations.append(l0_batch)

            #Cross entropy loss with SAE activations reconstruction on the batch
           
            #We compute the reconstruction by replacing all the hidden states by their reconstruction and not only the hidden state predicting the token of the class despite that the SAE is only
            #trained on this hidden state for each sentence. This is because otherwise, even if the reconstruction is not good, it might still be able to retrieve information on the other untouched
            #hidden states in the sentence.
            
            # logits_reconstruction = hook_model.run_with_hooks(
            #     cache,
            #     attention_mask=attention_mask,
            #     start_at_layer=sae.cfg.hook_layer,
            #     fwd_hooks=[
            #         (
            #             sae.cfg.hook_name,
            #             partial(reconstr_hook_classification_token, sae_out=sae_out_all),
            #         ) ],
            #     return_type="logits",
            # )

            # logits_reconstruction = hook_model.run_with_hooks(
            #     cache,
            #     attention_mask=attention_mask,
            #     start_at_layer=sae.cfg.hook_layer,
            #     fwd_hooks=[
            #         (
            #             sae.cfg.hook_name,
            #             partial(reconstr_hook_classification_token, sae_out=sae_out_all),
            #         ) ],
            #     return_type="logits",
            # )

            logits_reconstruction = hook_model.run_with_hooks(
                cache_sentence,
                start_at_layer=sae.cfg.hook_layer,
                fwd_hooks=[
                    (
                        sae.cfg.hook_name,
                        partial(reconstr_hook_classification_token_single_element, sae_out=sae_out),
                    ) ],
                return_type="logits",
            )
            
            logits_reconstruction = logits_reconstruction.contiguous().view(-1, logits_reconstruction.shape[-1])

            #We compute the original classification cross-entropy loss and the same loss obtained by plugging the reconstructed activations from the SAE features at hook {sae.cfg.hook_name} 
            original_loss = compute_loss_last_token_classif(input_ids,original_output,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[-1])
            original_total_loss  += original_loss
            sae_reconstruction_loss = compute_loss_last_token_classif(input_ids,logits_reconstruction,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[-1])
            sae_reconstruction_total_loss += sae_reconstruction_loss

            
            #We compute the variation in true accuracy
            acc_original = update_metrics(input_ids,original_output,labels_tokens_id,dict_metrics_original,is_eos)
            acc_reconstruction = update_metrics(input_ids,logits_reconstruction,labels_tokens_id,dict_metrics_reconstruction,is_eos)
            total_matches_original += acc_original.item()
            total_matches_reconstruction += acc_reconstruction.item()
            #We compute the recovering accuracy metric
            count_same_predictions = compute_same_predictions(original_output,logits_reconstruction,is_eos)
            total_same_predictions += count_same_predictions.item()
            
        del cache

        accuracy_original = total_matches_original / (len(activations_dataset)*bs)
        accuracy_reconstruction = total_matches_reconstruction / (len(activations_dataset)*bs)
        recovering_accuracy = total_same_predictions / (len(activations_dataset)*bs)
        sae_reconstruction_mean_loss = sae_reconstruction_total_loss / len(activations_dataloader)
        original_mean_loss = original_total_loss / len(activations_dataloader)

            
        array_store_l0_activations = np.concatenate(store_l0_activations)
        mean_l0_activations = np.mean(array_store_l0_activations)
        # Create the histogram
        fig = px.histogram(
            array_store_l0_activations, 
            title=f'L0 sparsity histogram at hook {sae.cfg.hook_name} over {array_store_l0_activations.size} activations.  Average L0 : {mean_l0_activations}', 
            labels={'value': 'L0 norm'}, 
            template='plotly_white'  # Choose a template: 'plotly', 'plotly_white', 'plotly_dark', etc.
        )
        
        # Customize axes titles
        fig.update_xaxes(title_text='L0 Norm')
        fig.update_yaxes(title_text='Number of occurences')
        


        print(f"\nOriginal classification crossentropy mean loss : {original_mean_loss} ({total_matches_original} matches) - SAE reconstruction classification crossentropy mean loss : {sae_reconstruction_mean_loss} (Computed over {len(labels_dataset)} sentences) ")
        print(f'\nRecovering accuracy : {recovering_accuracy}')
        print(f"\nOriginal accuracy of the model : {accuracy_original} - Accuracy of the model when plugging the SAE reconstruction hidden states : {accuracy_reconstruction}")

        
        sae_performance = {'Original Mean Loss':original_mean_loss.item(), 'SAE Mean loss':sae_reconstruction_mean_loss.item(), 'Mean L0 activation' : float(mean_l0_activations), 'Number sentences':labels_dataset, 'Recovering accuracy':recovering_accuracy, 'Original accuracy':accuracy_original, 'Reconstruction accuracy' : accuracy_reconstruction}
        #We merge the macro information of the sae with the dictionary of the variation metrics
        sae_performance = sae_performance |  dict_metrics_original | dict_metrics_reconstruction
    
        # #This part is not functional when training not only on the single token
        # mean_activations = torch.zeros_like(feature_activations_list[0][0])
        # for sample_activation in feature_activations_list:
        #     mean_activations += sample_activation.sum(dim=0)
        # mean_activations = mean_activations[0]
        # mean_activations /= len(feature_activations_list)
        # print(f"mean_activations : {mean_activations}")
        
        if return_feature_activations:
            

            #Remove the template part
            for t,_ in enumerate(original_text_used):
                original_text_used[t] = original_text_used[t][len_example:-len_template] 
            
            #Return the ground truth labels of the prompts
            prompt_labels_tensor = torch.cat(prompt_labels_list,dim=0)
            
            #The number of hidden states is the same for each prompt, equals to 1 as we keep only the (almost) last token. So we can concatenate as the last two dimensions are the same for each feature activation vector.
            feature_activations_tensor = torch.cat(feature_activations_list,dim=0)
            acts_without_process_tensor = torch.cat(acts_without_process_list,dim=0)
            original_activation_tensor = torch.cat(original_activation_list,dim=0)

            mean_activations = feature_activations_tensor.mean(dim=0).squeeze(0)
            print(f"mean_activations: {mean_activations }")

            #Compute similarity activations stats between features
            #mean_cos_sim, max_cos_sim = compute_feature_cosine_similarity(feature_activations_tensor,mean_activations)

            #print(f'The mean cosine similarity between the features activations is {mean_cos_sim} with a maximum up to {max_cos_sim}')
            
            return sae_performance , new_activations_dataset, fig, mean_activations, {'feature activation' : feature_activations_tensor, 'activations without processing' : acts_without_process_tensor,'original activation' : original_activation_tensor, 'prompt label' : prompt_labels_tensor, 'model output logits' : model_logits_labels, 'encoder' : sae.W_enc}, original_text_used
           
        return sae_performance, new_activations_dataset ,fig


def count_matches(labels,logits,is_eos):

    _,_,vocab_size = logits.size()

    logits_prediction = logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size).argmax(dim=1)
    labels_to_pred = labels[:,(-1-int(is_eos))].view(-1)

    exact_matches = (logits_prediction==labels_to_pred)
    count_exact_matches = exact_matches.sum() #tensor

    return count_exact_matches.item()
    

def run_model_to_get_pred(
    hook_model:HookedTransformer,
    sae:HookedTransformer,
    labels_dataset,
    tokenizer,
    activations_dataset,
    mean_activations,
    perturb=None,
    device='cpu',
    return_feature_activations=False,
    batch_size=1,proportion_to_evaluate=1.,is_eos=True,prompt_tuning=False,loss_type='cross_entropy'
):
    dict_metrics_reconstruction = {}
    
    
    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(np.array(labels_dataset))
    keys_labels = set(unique_labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}

    # Evaluation loop
    hook_model.eval()
    sae.eval()

    #This is where we store the prediction logits specific to classification. One for each category plus one that sums up all the logits associated to tokens which do not belong to the tokens of a class
    classification_predicted_probs = torch.zeros((len(labels_dataset),len(unique_labels)+1)).cpu()

    #For accuracy 
    number_matches = 0

    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    
    bs = next(iter(activations_dataloader))["input_ids"].shape[0]

    mean_coef_activation = mean_activations.mean()

    with torch.no_grad():
       
        count = 0
        for num_batch, batch in tqdm(enumerate(activations_dataloader), desc=f"Forward Passes with the SAE with ablation on {0 if perturb is None else perturb.shape[0]} feature(s) ", unit="batch"):
                
                inputs = batch["input_ids"].to(device)
                cache = batch["cache"].to(device)
                labels = batch["label"].to(device)

                a,c = cache.shape
                #cache_flatten = cache.view(a*b,-1).unsqueeze(1)
                cache_sentence = cache.unsqueeze(1)
    
                # Use the SAE
                feature_acts, acts_without_process = sae.encode_with_hidden_pre(cache_sentence)
                #feature_acts_all, acts_without_process_all = sae.encode_with_hidden_pre(cache_flatten)

                if perturb is not None:
                    #feature_acts_all[:,:,perturb] = 0
                    feature_acts[:,:,perturb] = 0
                    
                #sae_out_all = sae.decode(feature_acts_all)
                sae_out = sae.decode(feature_acts) 

                #print(f"norm diff : {torch.norm(cache_sentence.squeeze(1)-sae_out.squeeze(1),p=2,dim=1).mean()}")
            
                # logits_reconstruction = hook_model.run_with_hooks(
                #     cache,
                #     start_at_layer=sae.cfg.hook_layer,
                #     fwd_hooks=[
                #         (
                #             sae.cfg.hook_name,
                #             partial(reconstr_hook_classification_token, sae_out=sae_out_all),
                #         ) ],
                #     return_type="logits",
                # )


                logits_reconstruction = hook_model.run_with_hooks(
                    cache_sentence,
                    start_at_layer=sae.cfg.hook_layer,
                    fwd_hooks=[
                        (
                            sae.cfg.hook_name,
                            partial(reconstr_hook_classification_token_single_element, sae_out=sae_out),
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
                #print(f"class_probs : {class_probs}")
    
                if num_batch==(len(activations_dataloader)-1):
                    classification_predicted_probs[num_batch*inputs.shape[0] : ] = class_probs.cpu()
                else:
                    classification_predicted_probs[num_batch*inputs.shape[0] : (num_batch+1)*inputs.shape[0] ] = class_probs.cpu()

                number_matches += update_metrics(inputs,predicted_logits,labels_tokens_id,dict_metrics_reconstruction,is_eos)

    del cache
    del logits_reconstruction
    del predicted_logits
    del class_probs
    
    accuracy = (number_matches / (len(activations_dataset)*bs)).item()
    
    return  classification_predicted_probs , accuracy, dict_metrics_reconstruction 
    


def eval_causal_effect_model(
    probs_pred : torch.Tensor, 
    accuracy : float,
    class_int : int,
    causal_features: torch.Tensor,
    hook_model:HookedTransformer,
    sae:HookedTransformer,
    labels_dataset:Dataset, #expected to be tokenized
    tokenizer,
    activations_dataset,
    mean_activations,
    device,
    return_feature_activations=False,
    batch_size=1,proportion_to_evaluate=1.,is_eos=True,prompt_tuning=True,loss_type='cross_entropy'):

    #This is where we save the impact and change of accuracy resulting from the ablation of one feature
    overall_effects = []
    overall_accs_change = []
    overall_dict_metrics = []
    
    write = ''

    #We look at the impact of ablations of all the features expect those contained in 'causal_features'
    all_features = torch.arange(0,sae.cfg.d_sae) 
    #all_features = torch.arange(0,sae.cfg.d_in) 
    # mask = ~torch.isin(all_features, causal_features)
    # features_to_ablate = all_features[mask]
    # probs_pred_only_main, accuracy_only_main, dict_metrics_only_main = run_model_to_get_pred(hook_model,sae,dataset,data_collator,tokenizer,activations_dataset,device,batch_size,proportion_to_evaluate,is_eos,prompt_tuning,loss_type,perturb=features_to_ablate)
    # only_keep_top10_accuracy_change_relative = (accuracy_only_main - accuracy) / accuracy
    # only_keep_top10_effect = (probs_pred_only_main - probs_pred).abs().mean().item()
    # print(f'When we only maintain the {causal_features.tolist()} features by the metric of mean activation, associated to the class {class_int}, it results in the following effects : \n')
    # print(f'Effect : {only_keep_top10_effect}; Realtive Accuracy change: {only_keep_top10_accuracy_change_relative} \n')

    only_keep_top10_accuracy_change_relative = 0
    only_keep_top10_effect = 0
    dict_metrics_only_main = {}
    

    #We look at the impact of ablations of all the selected features simultaneously
    probs_pred_without_selected, accuracy_only_without_selected, dict_metrics_without_selected = run_model_to_get_pred(hook_model,sae,labels_dataset,tokenizer,activations_dataset,mean_activations,perturb=causal_features,device=device,batch_size=batch_size,proportion_to_evaluate=proportion_to_evaluate,is_eos=is_eos,prompt_tuning=prompt_tuning,loss_type=loss_type)
    without_selected_accuracy_change_relative = (accuracy_only_without_selected - accuracy) / accuracy
    without_selected_accuracy_change_absolute = (accuracy_only_without_selected - accuracy)
    without_selected_tvd = 0.5 * torch.sum(torch.abs(probs_pred - probs_pred_without_selected), dim=1).mean().item()  
    print(f'When we desactivate the {causal_features.tolist()} features by the metric of mean activation,  associated to the class {class_int}, it results in the following effects : \n')
    print(f'TVD : {without_selected_tvd}; Relative Accuracy change: {without_selected_accuracy_change_relative}; Absolute Accuracy change: {without_selected_accuracy_change_absolute} \n')
    
    # for feature in causal_features:
    #     probs_pred_perturbed, accuracy_perturbed, dict_metrics_without_feature = run_model_to_get_pred(hook_model,sae,dataset,data_collator,tokenizer,activations_dataset,device,batch_size,proportion_to_evaluate,is_eos,prompt_tuning,loss_type,perturb=feature)
    #     accuracy_change_relative = (accuracy_perturbed - accuracy) / accuracy
    #     effect = (probs_pred_perturbed - probs_pred).abs().mean().item()
    #     overall_accs_change.append(accuracy_change_relative)
    #     overall_effects.append(effect)
    #     overall_dict_metrics.append(dict_metrics_without_feature)
    #     write += 'Feature {}: Effect {}'.format(feature,effect)
    #     write += '; Relative Accuracy change: {} \n'.format(accuracy_change_relative)
    
    # print(write)

    #Temporary
    overall_accs_change = [0]
    overall_effects = [0]
    
    return overall_accs_change, overall_effects, overall_dict_metrics,only_keep_top10_accuracy_change_relative, only_keep_top10_effect, dict_metrics_only_main, without_selected_accuracy_change_relative, without_selected_tvd, dict_metrics_without_selected

def umap_projection(
    W_dec_numpy:np.ndarray,
    top_logits:np.ndarray,
    bottom_logits:np.ndarray,
    mean_activations:torch.Tensor,
    dict_analysis_features:dict,
):

    #2D UMAP plan of the decoder features/rows
    logger.info(f"Compute the UMAP projection of the features/decoder rows")
    N = W_dec_numpy.shape[0]

    labels = dict_analysis_features['prompt label'].numpy()
    original_activations = dict_analysis_features['original activation'].squeeze(1).numpy()
    sae_activations = dict_analysis_features['feature activation'].squeeze(1).numpy()
    unique_labels = np.unique(labels)

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


    
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.01, output_metric='haversine' ,n_components=2)
    reducer.fit(original_activations_norm)

    reduce_embeddings = reducer.transform(W_dec_numpy)

    #Display the hidden representation prototype for each class
    np_prototypes = np.array(list(hidden_size_prototype_class.values()))  # Convert dictionary values to a numpy array
    np_keys_prototypes = list(hidden_size_prototype_class.keys())  # Get dictionary keys for hover text
    np_prototypes_umap = reducer.transform(np_prototypes)

    x = np.sin(reduce_embeddings[:, 0]) * np.cos(reduce_embeddings[:, 1])
    y = np.sin(reduce_embeddings[:, 0]) * np.sin(reduce_embeddings[:, 1])
    z = np.cos(reduce_embeddings[:, 0])

    x_prototypes = np.sin(np_prototypes_umap[:, 0]) * np.cos(np_prototypes_umap[:, 1])
    y_prototypes = np.sin(np_prototypes_umap[:, 0]) * np.sin(np_prototypes_umap[:, 1])
    z_prototypes = np.cos(np_prototypes_umap[:, 0])
    
    # UMAP figure parameters
    # Prepare hover data
    sizes = mean_activations.numpy()
    hover_data = [
        f"Point {i}<br>Top : {', '.join(top_logits[:10,i])}<br>Bottom: {', '.join(bottom_logits[:10,i])}"
        for i in range(N)
    ]


    # Create interactive plots
    fig_sphere = px.scatter_3d(
        x=x,
        y=y,
        z=z,
        size=sizes,
        title='Interactive UMAP Projection'
    )

    fig_sphere.update_traces(
        mode='markers',
        marker=dict( opacity=0.8, color=colors,line=dict(width=2, color='DarkSlateGrey')),
        customdata=hover_data,  
        hovertemplate='<b>%{customdata}</b><extra></extra>'  
    )

     #Add class prototypes to the figure
    fig_sphere.add_trace(go.Scatter3d(
        x=x_prototypes,
        y=y_prototypes,
        z=z_prototypes,
        mode='markers',
        marker=dict(size=12, opacity=0.8, color='yellow'),
        text=np_keys_prototypes,  
        hoverinfo='text',  
        name='Class prototypes'  # Label for the legend
    ))

    # Customize layout
    fig_sphere.update_layout(
        hovermode='closest',
        title='Interactive UMAP Plot',
        showlegend=False,
        dragmode='pan',
        width=1600, 
        height=1200  
    )

    x = np.arctan2(x, y)
    y = -np.arccos(z)
    x_prototypes = np.arctan2(x_prototypes, y_prototypes)
    y_prototypes = -np.arccos(z_prototypes)
    
    fig = px.scatter(
        x=x,
        y=y,
        size=sizes,
        title='Interactive UMAP Projection',
        labels={'x': 'UMAP Component 1', 'y': 'UMAP Component 2'}
    )

    fig.update_traces(
        mode='markers',
        marker=dict( opacity=0.8, color=colors,line=dict(width=2, color='DarkSlateGrey')),
        customdata=hover_data,  
        hovertemplate='<b>%{customdata}</b><extra></extra>'  
    )

    fig.add_trace(go.Scatter(
        x=x_prototypes,
        y=y_prototypes,
        mode='markers',
        marker=dict(size=12, opacity=0.8, color='yellow'),
        text=np_keys_prototypes,  
        hoverinfo='text',  
        name='Class prototypes'  
    ))


    # Customize layout
    fig.update_layout(
        hovermode='closest',
        title='Interactive UMAP Plot',
        xaxis_title='UMAP Component 1',
        yaxis_title='UMAP Component 2',
        showlegend=False,
        dragmode='pan',
        width=1600,  
        height=1200  
    )

    return fig, fig_sphere
    
def pca_activations_projection(
    W_dec_numpy:np.ndarray,
    top_logits:np.ndarray,
    bottom_logits:np.ndarray,
    mean_activations:torch.Tensor,
    dict_analysis_features:dict,
):
    
    #2D projection of the decoder features/rows on PCA axis computed with original activations 
    logger.info(f"Compute the PCA plan of the original activations and projection the features/decoder rows on it")
    '''PCA in the activation space'''
    N = W_dec_numpy.shape[0]
    
    labels = dict_analysis_features['prompt label'].numpy()
    original_activations = dict_analysis_features['original activation'].squeeze(1).numpy()
    sae_activations = dict_analysis_features['feature activation'].squeeze(1).numpy()
    unique_labels = np.unique(labels) #sorted unique elements of the array

    #For analysis of the originaneurons
    #sae_activations = original_activations.copy()
    
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

    # #Load PCA and scaler
    # with open('./scaler_model_5.pkl', 'rb') as file:
    #     scaler = pickle.load(file) 

    # with open('./pca_model_5.pkl', 'rb') as file:
    #     pca = pickle.load(file) 

    #On a normalisé les activations avant de faire la PCA dessus car les vecteurs du decoder sont eux-mêmes normalisés
    
    scaler = StandardScaler()
    activations_scaled = scaler.fit_transform(original_activations_norm)    
    pca = PCA(n_components=3)  
    pca.fit(activations_scaled)

    # #Save PCA model and scaler model
    # with open('./pca_model_1.pkl', 'wb') as file:
    #     pickle.dump(pca, file)

    # with open('./scaler_model_1.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)


    W_dec_features_scaled = scaler.transform(W_dec_numpy)
    W_dec_pca = pca.transform(W_dec_features_scaled)

    #Display the hidden representation prototype for each class
    np_prototypes = np.array(list(hidden_size_prototype_class.values()))  # Convert dictionary values to a NumPy array
    np_keys_prototypes = list(hidden_size_prototype_class.keys())  # Get dictionary keys for hover text
    np_prototypes_scaled = scaler.transform(np_prototypes)
    np_prototypes_pca = pca.transform(np_prototypes_scaled)


    sizes = mean_activations.numpy()

    # dict_save_all = {'W_dec_pca' : W_dec_pca,'sizes' : sizes, 'np_keys_prototypes' : np_keys_prototypes, 'np_prototypes_pca' : np_prototypes_pca, 'top_logits' : top_logits, 'bottom_logits' : bottom_logits,  'colors': colors, 'feature_colors' : feature_colors, 'normalized_class_scores' : normalized_class_scores }
    # np.savez('data_dict_layer5_0.npz', **dict_save_all)

    ax, fig = design_figure(W_dec_pca, sizes, np_keys_prototypes, np_prototypes_pca, top_logits, bottom_logits, colors, feature_colors, normalized_class_scores,N)  
    fig.savefig('./layer5_90000.pdf', dpi=300, bbox_inches='tight')
    
    hover_data = [
        f"Point {i}<br>Top : {', '.join(top_logits[:10,i])}<br>Bottom: {', '.join(bottom_logits[:10,i])}"
        for i in range(N)
    ]
    fig_pca = px.scatter(
        x=W_dec_pca[:,0],
        y=W_dec_pca[:,1],
        size=sizes,
        title='Interactive PCA Projection',
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}
    )

    fig_pca.update_traces(
        mode='markers',
        marker=dict( opacity=0.8, color=colors,line=dict(width=2)),
        customdata=hover_data,  # Use customdata for hover text
        hovertemplate='<b>%{customdata}</b><extra></extra>'  # Format the custom data for hover
    )

    #Add class prototypes to the figure
    fig_pca.add_trace(go.Scatter(
        x=np_prototypes_pca[:, 0],
        y=np_prototypes_pca[:, 1],
        mode='markers',
        marker=dict(size=12, opacity=0.8, color='yellow'),
        text=np_keys_prototypes,  # Use text for hover text
        hoverinfo='text', 
        name='Class prototypes'  
    ))

    # Customize layout
    fig_pca.update_layout(
        hovermode='closest',
        title='Interactive PCA Plot',
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2',
        showlegend=False,
        dragmode='pan',
        width=1600,
        height=1200 
    )

    #Same but with the third component
    fig_pca_3d = px.scatter_3d(
        x=W_dec_pca[:,0],
        y=W_dec_pca[:,1],
        z=W_dec_pca[:,2],
        size=sizes,
        title='Interactive PCA Projection',
        labels={'x': 'PCA Component 1', 'y': 'PCA Component 2','z': 'PCA Component 3'}
    )

    fig_pca_3d.update_traces(
        mode='markers',
        marker=dict( opacity=0.8, color=colors,line=dict(width=2, color='DarkSlateGrey')),
        customdata=hover_data, 
        hovertemplate='<b>%{customdata}</b><extra></extra>' 
    )

    #Add class prototypes to the figure
    fig_pca_3d.add_trace(go.Scatter3d(
        x=np_prototypes_pca[:, 0],
        y=np_prototypes_pca[:, 1],
        z=np_prototypes_pca[:, 2],
        mode='markers',
        marker=dict(size=12, opacity=0.8, color='yellow'),
        text=np_keys_prototypes, 
        hoverinfo='text',  
        name='Class prototypes'  
    ))

    # Customize layout
    fig_pca_3d.update_layout(
        hovermode='closest',
        title='Interactive PCA Plot',
        showlegend=True,
        dragmode='pan',
        width=1600, 
        height=1200 
    )

    return fig_pca, fig_pca_3d, normalized_class_scores


def analyze_features(
    tune_sae:HookedTransformer,
    hook_model:HookedTransformer,
    mean_activations: torch.Tensor,
    dict_analysis_features: dict,
):
    
    logger.info(f"Compute the top 10 tokens for each feature/decoder row")
    #Get the top 10 Logit Weights/tokens for each feature/decoder row
    print(f"Shape of the decoder weights {tune_sae.W_dec.shape})")
    print(f"Shape of the model unembed {hook_model.W_U.shape}")
    projection_matrix = tune_sae.W_dec @ hook_model.W_U.to(dtype=torch.float32) #We convert here in case the weights are quantized
    print(f"Shape of the projection matrix {projection_matrix.shape}")
    # then we take the top_k tokens per feature and decode them
    top_k = 30
    N = tune_sae.W_dec.shape[0]
    _, top_k_tokens = torch.topk(projection_matrix, top_k, dim=1,sorted=True) #decreasing order of values
    _, bottom_k_tokens = torch.topk(projection_matrix, top_k, dim=1,sorted=True,largest=False) #ascedning order of values

    feature_df_columns = pd.MultiIndex.from_tuples(
    [(f'feature {i}', 'top') for i in range(N)] +
    [(f'feature {i}', 'bottom') for i in range(N)],
    names=['Feature', 'Positive/Negative']
    )

    feature_df = pd.DataFrame(index=range(top_k),columns=feature_df_columns)
    # feature_df = pd.DataFrame(top_k_tokens.cpu().numpy())
    # Fill the DataFrame
    for i in range(N):
        feature_df[(f'feature {i}', 'top')] = top_k_tokens[i].cpu().numpy()
        feature_df[(f'feature {i}', 'bottom')] = bottom_k_tokens[i].cpu().numpy()

    feature_df.index = [f"token_{i}" for i in range(top_k)]
    feature_df = feature_df.map(lambda x: hook_model.tokenizer.convert_ids_to_tokens(x))
    #Special case to prevvent too long useless tokens
    feature_df = feature_df.map(lambda x: 'ĠĠ' if (x is None or 'ĠĠ' in x) else x)
    #Remove the 'Ġ' at the beggining of the token
    feature_df = feature_df.map(lambda x: x[1:] if (x[0] == 'Ġ') else x)

    feature_df = feature_df.reindex(columns=pd.MultiIndex.from_tuples(
        [(f'feature {i}', sub) for i in range(N) for sub in ['top', 'bottom']],
        names=['Feature', 'Positive/Negative']
    ))

    W_dec_numpy = tune_sae.W_dec.detach().cpu().numpy()
    top_logits = feature_df.xs('top', axis=1, level='Positive/Negative').values.astype(str)
    bottom_logits = feature_df.xs('bottom', axis=1, level='Positive/Negative').values.astype(str)

    #fig_umap, fig_umap_sphere = umap_projection(W_dec_numpy,top_logits,bottom_logits,mean_activations,dict_analysis_features)
    fig_umap, fig_umap_sphere = None, None
    
    fig_pca, fig_pca_3d, normalized_class_scores = pca_activations_projection(W_dec_numpy,top_logits,bottom_logits,mean_activations,dict_analysis_features)

    #Top p actived features per F_c
    nb_labels = normalized_class_scores.shape[0]
    top_indice = np.argmax(normalized_class_scores,axis=0)
    j_select_list = []
    values_select_list = []
    
    p=10
    
    j_select_tensor = torch.zeros((nb_labels,p),dtype=torch.int)
    values_select_tensor = torch.zeros((nb_labels,p))

    
    for c in range(nb_labels):
        features_most_related_to_c =  torch.from_numpy( np.where(top_indice==c)[0] )
        top_mean_activations_values, top_p_indices  = torch.topk(mean_activations[features_most_related_to_c],k=p)
        #Map the top_p indices back to the original tensor 'mean_activations'
        j_select_c = features_most_related_to_c[top_p_indices]
        j_select_tensor[c,:] = j_select_c
        values_select_tensor[c,:] = top_mean_activations_values


    torch.save(j_select_tensor,'j_select_tensor.pt')
    torch.save(values_select_tensor,'values_select_tensor.pt')

    #Extract only the logit columns of the features we are interested in
    j_select_tensor_flatten = j_select_tensor.flatten()
    columns_to_extract = [f"feature {i}" for i in j_select_tensor_flatten]
    sub_feature_df = feature_df[columns_to_extract]

    print(f"j_select_tensor : {j_select_tensor}")
    
    return sub_feature_df, fig_umap, fig_umap_sphere, fig_pca, fig_pca_3d, j_select_tensor

    
def cache_activations_with_labels( hook_model,
                                    dataset, #expected to be tokenized
                                    data_collator,
                                    tokenizer,
                                    hook_name,
                                    is_eos):

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=data_collator)
    
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


def main_sae_evaluation(
    config_model: str,
    config_sae: str):
    
    #Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    #Retrieve the config of the SAE
    cfg_sae = SAELoadConfig.autoconfig(config_sae)
    
    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    #We retrieve the length of the preprompt added to the model if we specified it
    len_example = cfg_model.len_example
    #We retrieve the length of the template added at the end of the sentence
    len_template = cfg_model.len_template
    
    #Process the dataset on which we will do the forward passes
    dataset_tokenized = process_dataset(cfg_model,split="test",tokenizer=tokenizer) 
    #dataset_tokenized = process_dataset(cfg_model,split="train",tokenizer=tokenizer) 


    
    
    #Get model hooked (HookedTransformer)
    cfg_model.task_args['prompt_tuning'] = cfg_sae.evaluation_args['prompt_tuning']
    hook_model = get_hook_model(cfg_model,tokenizer)

    if cfg_sae.evaluation_args['prompt_tuning']:
        logger.info("Using prompt tuning for evaluation inference on the model")
        hook_model.load_state_dict(torch.load('prompt_tuning/prompt_tuning_llama_32_instruct.pth'))
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    #Load the local trained SAE
    tune_sae = TrainingSAE.load_from_pretrained(cfg_sae.sae_path,cfg_sae.evaluation_args['device'])
    
    #Check if the acivations and labels have already been cached
    dir_activations_with_labels = os.path.join(cfg_sae.dir_acts_with_labels,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    cache_activations_file_path = os.path.join(dir_activations_with_labels,f'layer_{tune_sae.cfg.hook_layer}.pkl')
    original_text_file_path = os.path.join(dir_activations_with_labels,f'layer_{tune_sae.cfg.hook_layer}.json')
    if not os.path.exists(dir_activations_with_labels):
            os.makedirs(dir_activations_with_labels)
            
    

    if not os.path.isfile(cache_activations_file_path):
            activations_dataset, original_text_used = cache_activations_with_labels(hook_model,
                                                               dataset_tokenized,
                                                               data_collator,
                                                               tokenizer,
                                                               hook_name = tune_sae.cfg.hook_name,
                                                               is_eos = cfg_model.task_args['is_eos'])
    
            torch.save(activations_dataset.data_block,cache_activations_file_path)

        
            with open(original_text_file_path, 'w') as f:
                json.dump(original_text_used, f)
    else:
        data_block = torch.load(cache_activations_file_path)
        activations_dataset = ActivationDataset()
        activations_dataset.data_block = data_block
                
        with open(original_text_file_path, 'r') as f:
            original_text_used = json.load(f)

    
    labels_dataset = dataset_tokenized["token_labels"]
    
    
    if cfg_sae.evaluation_args['return_feature_activations']:
        classification_loss_dict, new_activations_dataset, histogram_sparsity, mean_activations,dict_analysis_features, text_used  = eval_hook_loss(hook_model,
                                                                                                                                                    tune_sae,
                                                                                                                                                    labels_dataset,
                                                                                                                                                    original_text_used,
                                                                                                                                                    tokenizer,
                                                                                                                                                    activations_dataset,
                                                                                                                                                    len_example,
                                                                                                                                                    len_template,
                                                                                                                                                    **cfg_sae.evaluation_args)

        

        #Create the directory where to save the activations if it does not already exist
        dir_to_save_sae_activations = os.path.join(cfg_sae.dir_to_save_activations,cfg_sae.sae_name)
        if not os.path.exists(dir_to_save_sae_activations):
            os.makedirs(dir_to_save_sae_activations)
        
        #Save feature activations, ground truth labels, used prompts and model output logits
        file_to_save_sae_activations = os.path.join(dir_to_save_sae_activations,f"{cfg_model.dataset_name}.pth")
        #print(f'file where activations are stored : {file_to_save_sae_activations}')
        torch.save(dict_analysis_features, file_to_save_sae_activations)

        file_to_save_text_used = os.path.join(dir_to_save_sae_activations,f"{cfg_model.dataset_name}.json")
        with open(file_to_save_text_used, 'w') as file:
            json.dump(text_used, file, indent=4)
        
    else:
        classification_loss_dict, new_activations_dataset, histogram_sparsity, mean_activations, activations_dataset = eval_hook_loss(hook_model,
                                                                                                                                        tune_sae,
                                                                                                                                        labels_dataset,
                                                                                                                                        original_text_used,
                                                                                                                                        tokenizer,
                                                                                                                                        len_example,
                                                                                                                                        len_template,
                                                                                                                                        **cfg_sae.evaluation_args)


    #text_used : list


    #Create the directory where to save the sae metrics if it does not already exist
    dir_to_save_sae_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name)
    if not os.path.exists(dir_to_save_sae_metrics):
        os.makedirs(dir_to_save_sae_metrics)

    '''For each SAE feature (decoder row), we compute the 10 tokens the most promoted by the feature 
    thanks to Logit Lens. We also compute a 2D representation of the features with UMAP'''
    feature_df, fig_umap, fig_umap_sphere, fig_pca, fig_pca_3d, j_select_tensor = analyze_features(tune_sae,hook_model,mean_activations,dict_analysis_features)

    

    #Interpretability part
    # #Get the sentences which activate the most the selected features
    sae_activations = dict_analysis_features['feature activation'].squeeze(1)

    #Compute the cosine similarity between the most activated features by class
    most_activated_features_by_class = sae_activations[:,j_select_tensor.view(-1)]
    display_cosine_similarity_stats(most_activated_features_by_class,j_select_tensor.view(-1))


    #print(f"tune_sae.W_dec shape : {tune_sae.W_dec.T.shape}")
    
    cosine_similarity_concepts(tune_sae.W_dec.T,0.8)
    mean_knn_distance = knn_distance_metric(tune_sae.W_dec,5)
    print(f"Mean k-NN Distance across the concepts : {mean_knn_distance}")
    
    directory_to_save_texts = "./results/top_texts/" 
    directory_to_save_csv = "./results/top_texts_csv/"

    nb_classes = len(np.unique(np.array(labels_dataset)))
    
    for c in range(nb_classes):
        features_to_inspect = j_select_tensor[c]
    
        dict_list_texts = {}
        for pos,feature in enumerate(features_to_inspect):
            activations_inspected = sae_activations[:,feature]
            top_values, top_indices = torch.topk(activations_inspected,10)
            top_texts = [text_used[i] for i in top_indices]

            n = activations_inspected.size(0)
            active_samples = torch.sum(activations_inspected > 0)
            percentage_activity = (active_samples / n) * 100

            if pos==0:
                file_mode = "w"
            else:
                file_mode = "a"
            
            with open(os.path.join(directory_to_save_texts,f'category_{c}.txt'),file_mode) as file:
                file.write(f'\n####### Feature {feature} ##########\n\n')

                # #Write top logits
                # for token_logit in feature_df[f'feature {feature}']['top']:
                #     file.write(f"{token_logit}\n")

                file.write("\n")
                
                for text, value in zip(top_texts, top_values):
                    # Write each text and value on a new line
                    file.write(f"\n{text}\t{value}\n")

            # CSV writing
            with open(os.path.join(directory_to_save_csv,f'category_{c}.csv'), mode=file_mode, newline='', encoding='utf-8') as file:
               
                writer = csv.writer(file)
                
                # Write the header row
                if file_mode=="w":
                    writer.writerow(['sentence', 'feature_id', 'sparsity', 'class'])
                
                # Write the data rows
                for sentence in top_texts:
                    writer.writerow([sentence, feature.item(), percentage_activity.item(), c])
            

            
            dict_list_texts[f'top_text_by_{feature}'] = top_texts
            dict_list_texts[f'top_values_by_{feature}'] = top_values.tolist()
    
        with open(f'./results/top_texts/category_{c}.json', 'w') as json_file:
            json.dump(dict_list_texts, json_file, indent=4)

        #Generate n*10 random sentences within the class
            
        labels_to_filter = dict_analysis_features['prompt label']
        indices = (labels_to_filter == c).nonzero(as_tuple=True)[0].tolist()
        sampled_indices = random.sample(indices, 10*features_to_inspect.shape[0])
        random_text = [text_used[i] for i in sampled_indices]

        # CSV writing
        with open(os.path.join(directory_to_save_csv,f'category_{c}_random.csv'), mode="w", newline='', encoding='utf-8') as file:
           
            writer = csv.writer(file)
            
            # Write the header row
            writer.writerow(['sentence', 'class'])
            
            # Write the data rows
            for sentence in random_text:
                writer.writerow([sentence, c])
        


    #We compute the recovering accuracy and accuracy of the model when only rebuilt with the most important SAE features
    all_features = torch.arange(0,tune_sae.cfg.d_sae)
    projection_features = j_select_tensor.flatten()
    remove_features = torch.tensor(list(set(all_features.tolist()) - set(projection_features.tolist())))
    
    probs_pred, accuracy, dict_metrics_original = run_model_to_get_pred(hook_model,tune_sae,labels_dataset,tokenizer,new_activations_dataset,mean_activations,perturb=remove_features,**cfg_sae.evaluation_args)
    print(f"By projecting the activations onto the important retained SAE directions, the accuracy of the model is {accuracy}")
    print(dict_metrics_original)
    
    #We evaluate the causality of the most prominent features (by mean activation)
    if cfg_sae.causality:
        #First, we pre-select the most activated features in average
        #_,top_features_mean_activated_avg = torch.topk(mean_activations,cfg_sae.topk_mean_activation)
        # overall_accs_change, overall_effects, only_keep_top10_accuracy_change, only_keep_top10_effect = eval_causal_effect_model(top_features_mean_activated_avg,hook_model,tune_sae,dataset_tokenized,data_collator,tokenizer,**cfg_sae.evaluation_args)


        p_to_select = [1,5,10]

        
        #We first run the model with all the features on, then with one of the feature disable one at a time
        probs_pred, accuracy, dict_metrics_original = run_model_to_get_pred(hook_model,tune_sae,labels_dataset,tokenizer,new_activations_dataset,mean_activations,perturb=None,**cfg_sae.evaluation_args)

        for p in p_to_select:
        
            global_mean_acc_change = 0
            global_mean_effect = 0
            global_mean_acc_without_selected = 0
            global_mean_effect_without_selected = 0

            j_select_tensor_restrict = j_select_tensor[:,:p]
            
            #We do it by class
            for c in range(j_select_tensor.shape[0]):
                list_ablated_features = [int(feature_number) for feature_number in j_select_tensor_restrict[c]]
    
                overall_accs_change, overall_effects, overall_dict_metrics,only_keep_top10_accuracy_change_relative, only_keep_top10_effect, dict_metrics_only_main, without_selected_accuracy_change_relative, without_selected_effect, dict_metrics_without_selected = eval_causal_effect_model(probs_pred, accuracy, c,j_select_tensor_restrict[c],hook_model,tune_sae,labels_dataset,tokenizer,new_activations_dataset,mean_activations,**cfg_sae.evaluation_args)
    
                mean_accs_change = sum(overall_accs_change) / len(overall_accs_change)
                mean_overall_effects = sum(overall_effects) / len(overall_effects)
                global_mean_acc_change+=mean_accs_change
                global_mean_effect+=mean_overall_effects
                global_mean_acc_without_selected += without_selected_accuracy_change_relative
                global_mean_effect_without_selected += without_selected_effect
    
        
                #list_ablated_features = [int(feature_number) for feature_number in top_features_mean_activated_avg]
                
                dict_causal_metrics = {'Selected features' : list_ablated_features, 'Ablation all expect the selected features : impact' : only_keep_top10_effect, 'Ablation all expect the selected features : realtive accuracy change' : only_keep_top10_accuracy_change_relative, 'Ablation on all the selected features : impact' : without_selected_effect, 'Ablation on all the selected features : realtive accuracy change' : without_selected_accuracy_change_relative,'Ablation relative accuracy change' : overall_accs_change, 'Ablation impact' : overall_effects,  'Ablation mean accuracy change' : mean_accs_change, 'Ablation mean impact' : mean_overall_effects}
    
                
                if c==0:
                    file_to_save_original_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_original_sae.json") 
                    with open(file_to_save_original_metrics, 'w') as file:
                        json.dump(dict_metrics_original, file, indent=4)
                
                file_to_save_causal_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_p_{p}_causal_{c}.json") 
                with open(file_to_save_causal_metrics, 'w') as file:
                    json.dump(dict_causal_metrics, file, indent=4)
                
                # #Save the more detailled dictionaries
                # file_to_save_causal_metrics_only = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_causal_keep_only_p_{p}_class_{c}.json") 
                # with open(file_to_save_causal_metrics_only, 'w') as file:
                #     json.dump(dict_metrics_only_main, file, indent=4)
    
                file_to_save_causal_metrics_without = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_causal_without_p_{p}_class_{c}.json") 
                with open(file_to_save_causal_metrics_without, 'w') as file:
                    json.dump(dict_metrics_without_selected, file, indent=4)
    
                # for l,feature_number in enumerate(list_ablated_features):
                #     file_to_save_causal_metrics_feature = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_causal_without_{feature_number}.json") 
                #     with open(file_to_save_causal_metrics_feature, 'w') as file:
                #         json.dump(overall_dict_metrics[l], file, indent=4)
    
            
    
            global_mean_acc_change/=j_select_tensor.shape[0]
            global_mean_effect/=j_select_tensor.shape[0]
            print(f"Global mean accuracy change over all features (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_acc_change}")
            print(f"Global mean effect over all features (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_effect}")
            global_mean_acc_without_selected/=j_select_tensor.shape[0]
            global_mean_effect_without_selected/=j_select_tensor.shape[0]
            print(f"Mean accuracy change over all the clusters (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_acc_without_selected}")
            print(f"Mean TVD over all the clusters (for {j_select_tensor_restrict.shape[1]} features selected) : {global_mean_effect_without_selected}")
        
    #Save sparsity histogram 
    file_to_save_histogram = os.path.join(dir_to_save_sae_metrics,f"{cfg_model.dataset_name}.png")
    pio.write_image(histogram_sparsity, file_to_save_histogram)
    
    #Save SAE metrics performance dict as json
    file_to_save_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}.json") 
    with open(file_to_save_metrics, 'w') as file:
        json.dump(classification_loss_dict, file, indent=4)



    #Create the directory where to save the top logits and UMAP plan if it does not already exist
    dir_to_save_top_logits = cfg_sae.dir_to_save_top_logits 
    if not os.path.exists(dir_to_save_top_logits):
        os.makedirs(dir_to_save_top_logits)
        
    
    #Save top logits for each feature
    file_to_save_top_logits = os.path.join(dir_to_save_top_logits,f"{cfg_sae.sae_name}.csv")
    feature_df.to_csv(file_to_save_top_logits,index=False,encoding='utf-8')
    #Save the UMAP plots
    # file_to_save_umap_plot = os.path.join(dir_to_save_top_logits,f"UMAP_{cfg_sae.sae_name}.html")
    # fig_umap.write_html(file_to_save_umap_plot,config={'scrollZoom': True})
    # file_to_save_umap_plot_sphere = os.path.join(dir_to_save_top_logits,f"UMAP_3d_SPHERE{cfg_sae.sae_name}.html")
    # fig_umap_sphere.write_html(file_to_save_umap_plot_sphere,config={'scrollZoom': True})
    
    #Save PCA plot
    file_to_save_pca_plot = os.path.join(dir_to_save_top_logits,f"PCA_{cfg_sae.sae_name}.html")
    fig_pca.write_html(file_to_save_pca_plot,config={'scrollZoom': True})
    file_to_save_3d_pca_plot = os.path.join(dir_to_save_top_logits,f"PCA_3d{cfg_sae.sae_name}.html")
    fig_pca_3d.write_html(file_to_save_3d_pca_plot,config={'scrollZoom': True})


