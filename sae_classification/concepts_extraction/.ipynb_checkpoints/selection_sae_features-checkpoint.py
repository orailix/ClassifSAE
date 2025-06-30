from ..utils import LLMLoadConfig, SAELoadConfig
from ..model_training import process_dataset,get_hook_model,compute_loss_last_token, PromptTunerForHookedTransformer
from .sae_evaluation import cache_activations_with_labels, ActivationDataset

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


def select_segments_features(
    hook_model,
    sae,
    activations_dataset,
    labels_dataset,
    is_eos
):

    unique_labels = np.unique(np.array(labels_dataset))
    
    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    bs = next(iter(activations_dataloader))["input_ids"].shape[0]

    feature_activations_list = []
    prompt_labels_list = []

    # Evaluation loop
    hook_model.eval()
    sae.eval()
    with torch.no_grad():

        for batch in tqdm(activations_dataloader, desc="Forward Passes of the given LLM model", unit="batch"):

            input_ids = batch['input_ids'].to(hook_model.cfg.device)
            cache = batch["cache"].to(hook_model.cfg.device)
            original_output = batch["output"].to(hook_model.cfg.device)
            labels = batch["label"]
            attention_mask = batch['attention_mask'].to(dtype=int).to(hook_model.cfg.device)

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

            feature_activations_list.append(feature_acts)
            prompt_labels_list.append(labels)


        feature_activations_tensor = torch.cat(feature_activations_list,dim=0).squeeze(1).cpu()
        labels_tensor = torch.cat(prompt_labels_list)
        mean_activations = feature_activations_tensor.mean(dim=0)
        #print(f"feature_activations_tensor shape : {feature_activations_tensor.shape}")
        #print(f"mean_activations shape : {mean_activations.shape}")

        

        #Assign a score to each feature with regard to each class
        feature_score_class = {}
        for label in unique_labels:
            indices = torch.where(labels_tensor==label)[0]
            #print(f"indices shape : {indices.shape}")
            #print(f"indices : {indices}")
            if indices.shape[0] > 0: #Is at least one sample is associated to this label
                feature_score_class[label] = torch.sum(feature_activations_tensor[indices,:],dim=0)

        
       
        feature_score_class_array = torch.zeros((len(unique_labels),feature_activations_tensor.shape[1]))
        #Concatenate features scores of each class for normalization
        for i,label in enumerate(unique_labels):
            feature_score_class_array[i,:] = feature_score_class[label]
        feature_sums = feature_score_class_array.sum(dim=0)
        #print(f"feature_sums : {feature_sums}")
        indices_dead_features = (feature_sums==0.)
        normalized_class_scores = torch.zeros((len(unique_labels),len(feature_sums)))
        normalized_class_scores[:,~indices_dead_features] = feature_score_class_array[:,~indices_dead_features] / feature_sums[~indices_dead_features]
        normalized_class_scores[:,indices_dead_features] = (torch.ones(len(unique_labels)) / len(unique_labels)).reshape(-1,1) 
        #print(f"normalized_class_scores shape : {normalized_class_scores.shape}")

        #Top p actived features per F_c
        nb_labels = normalized_class_scores.shape[0]
        top_indice = torch.argmax(normalized_class_scores,dim=0)
        j_select_list = []
        values_select_list = []
        
        p=8
        
        j_select_tensor = torch.zeros((nb_labels,p),dtype=torch.int)
        values_select_tensor = torch.zeros((nb_labels,p))
    
        
        for c in range(nb_labels):
            features_most_related_to_c = torch.where(top_indice==c)[0] 
            top_mean_activations_values, top_p_indices  = torch.topk(mean_activations[features_most_related_to_c],k=p)
            #Map the top_p indices back to the original tensor 'mean_activations'
            j_select_c = features_most_related_to_c[top_p_indices]
            j_select_tensor[c,:] = j_select_c
            values_select_tensor[c,:] = top_mean_activations_values


        print(f"j_select_tensor : {j_select_tensor}")
        return j_select_tensor

            

def selection_sae_features(

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

    #Process the dataset on which we will do the forward passes
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer) 
 
    
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

    cache_activations_file_path = os.path.join(dir_activations_with_labels,f'layer_{tune_sae.cfg.hook_layer}')
    original_text_file_path = os.path.join(dir_activations_with_labels,f'layer_{tune_sae.cfg.hook_layer}.json')
    if not os.path.exists(dir_activations_with_labels):
            os.makedirs(dir_activations_with_labels)
            
    
    print(f"cache_activations_file_path : {cache_activations_file_path}")
    if not os.path.isfile(cache_activations_file_path):
            activations_dataset, original_text_used = cache_activations_with_labels(hook_model,
                                                               dataset_tokenized,
                                                               data_collator,
                                                               tokenizer,
                                                               hook_name = tune_sae.cfg.hook_name,
                                                               is_eos = cfg_model.task_args['is_eos'])
    
            # with open(cache_activations_file_path, 'wb') as f:
            #     pickle.dump(activations_dataset, f)
            torch.save(activations_dataset.data_block,cache_activations_file_path)
            
        
            with open(original_text_file_path, 'w') as f:
                json.dump(original_text_used, f)
            
    else:
    
        # Load the cached activations with their labels
        # with open(cache_activations_file_path, 'rb') as f:
        #     activations_dataset = pickle.load(f)
        data_block = torch.load(cache_activations_file_path)
        activations_dataset = ActivationDataset()
        activations_dataset.data_block = data_block
    

    labels_dataset = dataset_tokenized["token_labels"]
    j_select_tensor = select_segments_features(hook_model, tune_sae, activations_dataset, labels_dataset, is_eos = cfg_model.task_args['is_eos'])
    j_select_tensor_flatten = j_select_tensor.flatten()

    dir_save_selected_features = cfg_sae.path_select_features
    if not os.path.exists(dir_save_selected_features):
            os.makedirs(dir_save_selected_features)

    torch.save(j_select_tensor, os.path.join(dir_save_selected_features, "selected_features.pth"))

    

    
    


    

