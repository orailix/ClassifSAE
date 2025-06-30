from ..utils import LLMLoadConfig, SAELoadConfig
from ..model_training import process_dataset,get_hook_model,compute_loss_last_token, PromptTunerForHookedTransformer
from .sae_evaluation import ActivationDataset, cache_activations_with_labels,cosine_similarity_concepts, display_cosine_similarity_stats,knn_distance_metric,update_metrics
     

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors



def filter_unique_embeddings(embeddings: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """
    Iteratively removes individuals with high cosine similarity to keep a more unique subset.

    Args:
        embeddings (torch.Tensor): Tensor of shape (N, d) where N is the number of embeddings and d is the dimension.
        threshold (float): Similarity threshold for considering pairs (default 0.9).

    Returns:
        torch.Tensor: A tensor containing the filtered unique embeddings.
    """
    # Normalize embeddings for cosine similarity computation
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # (N, d)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

    # Mask diagonal (self-similarity should be ignored)
    N = embeddings.shape[0]
    similarity_matrix.fill_diagonal_(0)

    # Find pairs with similarity above threshold
    pairs = (similarity_matrix > threshold).nonzero(as_tuple=False)  # (M, 2), where M is number of pairs
    
    # Convert to set for fast lookup
    A = set(tuple(pair) for pair in pairs.tolist())

    # Set of all individuals
    remaining_indices = set(range(N))

    # While there are still high-similarity pairs
    while A:
        # Randomly select one individual from a pair to remove
        i, j = random.choice(tuple(A))
        individual_to_remove = random.choice([i, j])  # Randomly pick one

        # Remove the chosen individual from remaining population
        remaining_indices.discard(individual_to_remove)

        # Remove all pairs involving the removed individual
        A = {pair for pair in A if individual_to_remove not in pair}

    # Return the filtered embeddings
    return list(remaining_indices)


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


def encode_with_selected_features(sae,cache_sentence,selected_features):
    
    feature_acts, acts_without_process = sae.encode_with_hidden_pre(cache_sentence)
    
    feature_acts_mask = torch.zeros_like(feature_acts,dtype=torch.bool)
    acts_without_process_mask = torch.zeros_like(acts_without_process,dtype=torch.bool)
    
    feature_acts_mask[:,:,selected_features.flatten()] = True
    acts_without_process_mask[:,:,selected_features.flatten()] = True
    
    feature_acts = feature_acts * feature_acts_mask
    acts_without_process = acts_without_process * acts_without_process_mask
    
    return feature_acts, acts_without_process
    
    
def eval_hook_loss(
    hook_model:HookedTransformer,
    sae:HookedTransformer,
    labels_dataset,
    original_text_used,
    tokenizer,
    activations_dataset,
    len_example,
    len_template,
    selected_features,
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
            feature_acts, acts_without_process = encode_with_selected_features(sae,cache_sentence,selected_features)
            sae_out = sae.decode(feature_acts)
           
            #Save the activations and labels for the causality calculations done after
            bs = input_ids.shape[0]
            d_in = cache.shape[-1]
            inputs_to_save = input_ids.cpu()
            cache_to_save = cache_sentence.squeeze(1).cpu()
            original_output_to_save = original_output.cpu()  #shape : [batch size, vocab size]
            labels_to_save = labels
            attention_mask_to_save = attention_mask.cpu()
            sae_activations_to_save = feature_acts.squeeze(1).cpu()
            new_activations_dataset.append(inputs_to_save, cache_to_save, original_output_to_save, labels_to_save, attention_mask_to_save, sae_activations_to_save)
            

            #In addition to the SAE metrics, we want to store feature activations and model predictions
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

        print(f"\nOriginal classification crossentropy mean loss : {original_mean_loss} ({total_matches_original} matches) - SAE reconstruction classification crossentropy mean loss : {sae_reconstruction_mean_loss} (Computed over {len(labels_dataset)} sentences) ")
        print(f'\nRecovering accuracy : {recovering_accuracy}')
        print(f"\nOriginal accuracy of the model : {accuracy_original} - Accuracy of the model when plugging the SAE reconstruction hidden states : {accuracy_reconstruction}")

        
        sae_performance = {'Original Mean Loss':original_mean_loss.item(), 'SAE Mean loss':sae_reconstruction_mean_loss.item(), 'Number sentences':labels_dataset, 'Recovering accuracy':recovering_accuracy, 'Original accuracy':accuracy_original, 'Reconstruction accuracy' : accuracy_reconstruction}
        #We merge the macro information of the sae with the dictionary of the variation metrics
        sae_performance = sae_performance |  dict_metrics_original | dict_metrics_reconstruction
        
        #Remove the template part
        for t,_ in enumerate(original_text_used):
            original_text_used[t] = original_text_used[t][len_example:-len_template] #Len specific to the AG News template, has to be adapted to do it automatically for other datasets
        
        #Return the ground truth labels of the prompts
        prompt_labels_tensor = torch.cat(prompt_labels_list,dim=0)
        
        #The number of hidden states is the same for each prompt, equals to 1 as we keep only the (almost) last token. So we can concatenate as the last two dimensions are the same for each feature activation vector.
        feature_activations_tensor = torch.cat(feature_activations_list,dim=0)
        acts_without_process_tensor = torch.cat(acts_without_process_list,dim=0)
        original_activation_tensor = torch.cat(original_activation_list,dim=0)

        mean_activations = feature_activations_tensor.mean(dim=0).squeeze(0)
       
        #Compute similarity activations stats between features
        #mean_cos_sim, max_cos_sim = compute_feature_cosine_similarity(feature_activations_tensor,mean_activations)

        #print(f'The mean cosine similarity between the features activations is {mean_cos_sim} with a maximum up to {max_cos_sim}')
        
        return sae_performance , new_activations_dataset, mean_activations, {'feature activation' : feature_activations_tensor, 'activations without processing' : acts_without_process_tensor,'original activation' : original_activation_tensor, 'prompt label' : prompt_labels_tensor, 'model output logits' : model_logits_labels, 'encoder' : sae.W_enc}, original_text_used
        
    


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
    selected_features,
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


    with torch.no_grad():
       
        for num_batch, batch in tqdm(enumerate(activations_dataloader), desc=f"Forward Passes with the SAE with ablation on {0 if perturb is None else perturb.shape[0]} feature(s) ", unit="batch"):
                
                inputs = batch["input_ids"].to(device)
                cache = batch["cache"].to(device)
              
                a,c = cache.shape
                #cache_flatten = cache.view(a*b,-1).unsqueeze(1)
                cache_sentence = cache.unsqueeze(1)
    
                # Use the SAE
                feature_acts, _  = encode_with_selected_features(sae,cache_sentence,selected_features)

                if perturb is not None:
                    #feature_acts_all[:,:,perturb] = 0
                    feature_acts[:,:,perturb] = 0
                    
                
                sae_out = sae.decode(feature_acts) 


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
    selected_features,
    verbose,
    device,
    return_feature_activations=False,
    batch_size=1,proportion_to_evaluate=1.,is_eos=True,prompt_tuning=True,loss_type='cross_entropy'):

  
    #We look at the impact of ablations of all the selected features simultaneously
    probs_pred_without_selected, accuracy_only_without_selected, dict_metrics_without_selected = run_model_to_get_pred(hook_model,sae,labels_dataset,tokenizer,activations_dataset,selected_features,perturb=causal_features,device=device,batch_size=batch_size,proportion_to_evaluate=proportion_to_evaluate,is_eos=is_eos,prompt_tuning=prompt_tuning,loss_type=loss_type)
    without_selected_accuracy_change_relative = (accuracy_only_without_selected - accuracy) / accuracy
    without_selected_accuracy_change_absolute = (accuracy_only_without_selected - accuracy)
    without_selected_tvd = 0.5 * torch.sum(torch.abs(probs_pred - probs_pred_without_selected), dim=1).mean().item()
    if verbose:
        print(f'Desactivation of {len(causal_features.tolist())} feature(s)  associated to the class {class_int}, it results in the following effects : \n')
        print(f'TVD : {without_selected_tvd}; Relative Accuracy change: {without_selected_accuracy_change_relative}; Absolute Accuracy change: {without_selected_accuracy_change_absolute} \n')
    

    return without_selected_accuracy_change_relative, without_selected_accuracy_change_absolute, without_selected_tvd, dict_metrics_without_selected




def main_sae_evaluation_restrict(
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
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer) 
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
            

    #Load the selected features
    selected_features = torch.load(os.path.join(cfg_sae.path_select_features, "selected_features.pth"))
    print(f"selected_features : {selected_features}")
    labels_dataset = dataset_tokenized["token_labels"]

    #Heuristic to add more post-processing on some SAE columns to filter out based on similarity
    index_new_selected_features = filter_unique_embeddings(tune_sae.W_dec[selected_features.flatten(),:],0.95)
    new_selected_features = selected_features.flatten()[index_new_selected_features]
    print(f"new_selected_features : {new_selected_features}")
    
    
    
    classification_loss_dict, new_activations_dataset, mean_activations,dict_analysis_features, text_used  = eval_hook_loss(hook_model,
                                                                                                                            tune_sae,
                                                                                                                            labels_dataset,
                                                                                                                            original_text_used,
                                                                                                                            tokenizer,
                                                                                                                            activations_dataset,
                                                                                                                            len_example,
                                                                                                                            len_template,
                                                                                                                            selected_features,
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
        
  
    #Create the directory where to save the sae metrics if it does not already exist
    dir_to_save_sae_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name)
    if not os.path.exists(dir_to_save_sae_metrics):
        os.makedirs(dir_to_save_sae_metrics)
    
    #Interpretability part
    # #Get the sentences which activate the most the selected features

    
    sae_activations = dict_analysis_features['feature activation'].squeeze(1)

    #Display the cosine similarity properties between the learned concepts
    print(f"selected_features : {selected_features}")
    cosine_similarity_concepts(tune_sae.W_dec.T[:,selected_features.flatten()],0.9)
    mean_knn_distance = knn_distance_metric(tune_sae.W_dec[selected_features.flatten(),:],k=5)
    print(f"Mean k-NN Distance across the concepts : {mean_knn_distance}")
    
    #Compute the cosine similarity between the most activated features by class
    display_cosine_similarity_stats(sae_activations[:,selected_features.flatten()],selected_features.view(-1))

    
    nb_classes = len(np.unique(np.array(labels_dataset)))
    
    directory_to_save_texts = "./results/top_texts/" 
    directory_to_save_csv = "./results/top_texts_csv/"
    for c in range(nb_classes):
        features_to_inspect = selected_features[c]
    
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
        
    
    #We evaluate the causality of the most prominent features (by mean activation)
    p_final = selected_features.shape[1]
    if cfg_sae.causality:
        
        #Size of the ablated segments
        p_to_select = torch.arange(0, p_final+5, step=5)
        p_to_select[0] = 1
        
        #We first run the model with all the features on, then with one of the feature disable one at a time
        probs_pred, accuracy, dict_metrics_original = run_model_to_get_pred(hook_model,tune_sae,labels_dataset,tokenizer,new_activations_dataset,selected_features,perturb=None,**cfg_sae.evaluation_args)

        list_mean_tvd = []
        list_mean_accuracy_change_absolute = []

        for p in p_to_select:
          
            #Random ablation on features
            nb_iter = 5
            for num_ablations in range(nb_iter):

                mean_tvd = 0
                mean_accuracy_change_absolute = 0

                flatten_selected_features = selected_features.flatten()
                total_features = len(flatten_selected_features)
                indices_ablation = torch.randperm(total_features)[:p]
                ablated_features = flatten_selected_features[indices_ablation]
                 
                
                without_selected_accuracy_change_relative, without_selected_accuracy_change_absolute, without_selected_effect, dict_metrics_without_selected = eval_causal_effect_model(probs_pred,
                         accuracy, 
                         None,ablated_features,
                         hook_model,
                         tune_sae,
                         labels_dataset,
                         tokenizer,
                         new_activations_dataset,
                         selected_features, 
                         verbose=False,
                         **cfg_sae.evaluation_args)
                mean_tvd += without_selected_effect
                mean_accuracy_change_absolute += without_selected_accuracy_change_absolute
                
            mean_tvd /= nb_iter
            mean_accuracy_change_absolute /= nb_iter

            list_mean_tvd.append(mean_tvd)
            list_mean_accuracy_change_absolute.append(mean_accuracy_change_absolute)

            print(f"For random ablation of {p} features. Mean TVD : {mean_tvd}. Mean change absolute accuracy : {mean_accuracy_change_absolute}.")

        
        dict_random_ablation_features = { 'Sizes of ablation' : p_to_select.tolist() , 'Mean TVD' : list_mean_tvd , 'Mean change absolute accuracy' : list_mean_accuracy_change_absolute}
        file_to_save_random_ablation_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_random_ablations.json") 
        with open(file_to_save_random_ablation_metrics, 'w') as file:
            json.dump(dict_random_ablation_features, file, indent=4)
            
        
        for p in p_to_select:

            global_mean_acc_without_selected = 0
            global_mean_acc_absolute_without_selected = 0
            global_mean_effect_without_selected = 0

            # perm = torch.randperm(selected_features.shape[1])
            # selected_features = selected_features[:,perm]
            
            
            selected_features_restrict = selected_features[:,:p]

                    
            #We do it by class
            for c in range(selected_features.shape[0]):
                list_ablated_features = [int(feature_number) for feature_number in selected_features_restrict[c,:]]
    
                without_selected_accuracy_change_relative, without_selected_accuracy_change_absolute,without_selected_effect, dict_metrics_without_selected = eval_causal_effect_model(probs_pred,
                                                                                                                                             accuracy, 
                                                                                                                                             c,selected_features_restrict[c,:],
                                                                                                                                             hook_model,
                                                                                                                                             tune_sae,
                                                                                                                                             labels_dataset,
                                                                                                                                             tokenizer,
                                                                                                                                             new_activations_dataset,
                                                                                                                                             selected_features,
                                                                                                                                             verbose=True,
                                                                                                                                             **cfg_sae.evaluation_args)
    
                global_mean_acc_without_selected += without_selected_accuracy_change_relative
                global_mean_acc_absolute_without_selected += without_selected_accuracy_change_absolute
                global_mean_effect_without_selected += without_selected_effect
    
                
                dict_causal_metrics = {'Selected features' : list_ablated_features, 
                                       'Ablation on all the selected features : impact' : without_selected_effect, 
                                       'Ablation on all the selected features : realtive accuracy change' : without_selected_accuracy_change_relative,
                                       'Ablation on all the selected features : absolute accuracy change' : without_selected_accuracy_change_absolute}
                    
                if c==0:
                    file_to_save_original_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_original_sae.json") 
                    with open(file_to_save_original_metrics, 'w') as file:
                        json.dump(dict_metrics_original, file, indent=4)
                
                file_to_save_causal_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_p_{p}_causal_{c}.json") 
                with open(file_to_save_causal_metrics, 'w') as file:
                    json.dump(dict_causal_metrics, file, indent=4)
                
    
                file_to_save_causal_metrics_without = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}_causal_without_p_{p}_class_{c}.json") 
                with open(file_to_save_causal_metrics_without, 'w') as file:
                    json.dump(dict_metrics_without_selected, file, indent=4)
    
    
            global_mean_acc_without_selected/=selected_features_restrict.shape[0]
            global_mean_acc_absolute_without_selected/=selected_features_restrict.shape[0]
            global_mean_effect_without_selected/=selected_features_restrict.shape[0]
            print(f"Mean absolute accuracy change over all the clusters (for {selected_features_restrict.shape[1]} features selected per cluster) : {global_mean_acc_absolute_without_selected}")
            print(f"Mean TVD over all the clusters (for {selected_features_restrict.shape[1]} features selected per cluster) : {global_mean_effect_without_selected}")
            

                                     
    #Save SAE metrics performance dict as json
    file_to_save_metrics = os.path.join(cfg_sae.dir_to_save_metrics,cfg_sae.sae_name,f"{cfg_model.dataset_name}.json") 
    with open(file_to_save_metrics, 'w') as file:
        json.dump(classification_loss_dict, file, indent=4)


 


