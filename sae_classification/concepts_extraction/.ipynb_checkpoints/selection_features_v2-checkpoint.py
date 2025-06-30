from .evaluation_utils import forward_on_selected_features, reconstr_hook_classification_token_single_element, count_same_match, cache_activations_with_labels, ActivationDataset
from ..utils import EvaluationConfig, LLMLoadConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ..model_training import process_dataset,get_hook_model
import os
import numpy as np
import torch
import json
import joblib
import sys
sys.path.append("../../")
from torch.utils.data import DataLoader
from .concept_shap import ConceptNet
from tqdm import tqdm
from loguru import logger
from functools import partial
from sae_implementation import TrainingSAE

def heuristic_filter_out(hook_model, 
                         interp_model,
                         activations_dataloader,
                         labels_dataset, 
                         method_name,
                         mean_activations,
                         batch_size,
                         hook_layer,
                         hook_name,
                         is_eos,
                         path_to_dr_methods=''):

    recovering_accuracy_target = 0.98
    max_iterations = 100
    current_recovering_accuracy = 0.
    current_iteration = 0
    nb_features_to_add = 5

    selected_features = []

    #Add top features based on mean activation
    
    
    while (current_recovering_accuracy < recovering_accuracy_target) and (current_iteration<max_iterations) and (len(selected_features) <= mean_activations.shape[-1]):
        current_iteration+=1
        #Incrementally add the top features based on mean activations
        _,top_features = torch.topk(mean_activations,nb_features_to_add)
        selected_features.extend(top_features.tolist())
        mean_activations[top_features] = 0

        nb_same_predictions = 0

        # Evaluation loop
        hook_model.eval()
        if method_name in ["sae","concept_shap"]:
            interp_model.eval()
        with torch.no_grad():
    
            for batch in tqdm(activations_dataloader, desc="Forward Passes of the given LLM model", unit="batch"):
    
                input_ids = batch['input_ids'].to(hook_model.cfg.device)
                cache = batch["cache"].to(hook_model.cfg.device)
                logits_original = batch["output"].to(hook_model.cfg.device)
                labels = batch["label"]
                attention_mask = batch['attention_mask'].to(dtype=int).to(hook_model.cfg.device)
    
    
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
                nb_same_predictions += count_same_match(logits_original,logits_reconstruction,is_eos)

        recovering_accuracy = nb_same_predictions /  (len(activations_dataloader)*batch_size)

        current_recovering_accuracy=recovering_accuracy
        print(f"In selecting only the top {len(selected_features)} features by mean activation, we reach a recovering accuracy of {recovering_accuracy*100}%")

    
    return selected_features

        
    

def segments_features(
    hook_model, 
    interp_model, 
    activations_dataset, 
    labels_dataset,
    features_selection,
    method_name,
    hook_layer,
    hook_name,
    is_eos,
    path_to_dr_methods=''
):
    unique_labels = np.unique(np.array(labels_dataset))
    
    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    bs = next(iter(activations_dataloader))["input_ids"].shape[0]

    feature_activations_list = []
    prompt_labels_list = []

    # Evaluation loop
    hook_model.eval()
    if method_name in ["sae","concept_shap"]:
        interp_model.eval()
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

            if method_name=='concept_shap':

                #concept_score_thres_prob
                feature_acts = interp_model.concepts_activations(cache)
                #print(f"feature_acts shape : {feature_acts.shape}")

            elif method_name=='sae':

                # Use the SAE
                feature_acts, acts_without_process = interp_model.encode_with_hidden_pre(cache_sentence)
                #print(f"feature_acts shape : {feature_acts.shape}")
                feature_acts = feature_acts.squeeze(1)
                #print(f"feature_acts shape : {feature_acts.shape}")
                #sae_out = sae.decode(feature_acts)

            elif method_name=='ica':
                embeddings_sentences_numpy = cache.cpu().numpy()
                #ica_activations
                feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)).to(hook_model.cfg.device)
                #print(f"feature_acts shape : {feature_acts.shape}")
            
            feature_activations_list.append(feature_acts)
            prompt_labels_list.append(labels)

        feature_activations_tensor = torch.cat(feature_activations_list,dim=0).cpu()
        print(f"feature_activations_tensor shape : {feature_activations_tensor.shape}")
        labels_tensor = torch.cat(prompt_labels_list)
        mean_activations = feature_activations_tensor.mean(dim=0)

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

        nb_labels = normalized_class_scores.shape[0]
        top_indice = torch.argmax(normalized_class_scores,dim=0)

        segmented_features = [[] for _ in range(nb_labels)]
        # Populate each sub-list with the indices of the features the most associated to the class c
        for feature_number, top_class in enumerate(top_indice):
            segmented_features[top_class.item()].append(feature_number)

        #Sort the features in each segment in decreasing order of their mean activations
        for i,segment in enumerate(segmented_features):
            segmented_features[i] =  sorted(segment, key=lambda j: mean_activations[j], reverse=True)


        #If we run our heuristic to filter out only the features participating in the recovering accuracy
        if features_selection:
            #For now, very simple heuristic
            #selected_features = heuristic_filter_out( hook_model, interp_model, activations_dataloader, labels_dataset, method_name, mean_activations, bs, hook_layer, hook_name,is_eos,path_to_dr_methods)
            all_features = torch.arange(len(mean_activations))
            selected_features = [x.item() for x in all_features[:80]]

            #Filter features which are not at all activated among the selected features
            selected_features = [x for x in selected_features if mean_activations[x]>1e-5]
            
            #We keep in segmented_features, only the features present in selected_features
            segmented_features = [[x for x in sublist if x in selected_features] for sublist in segmented_features]
            print(f"segmented_features : {segmented_features}")
        
        else:
            selected_features = torch.arange(mean_activations.shape[0])


        return segmented_features, selected_features
        

def selection_features(
    config_model: str,
    config_concept: str):

    #Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    #Retrieve the config of the extracting concept method (centralize for features selection, interpretation and causality)
    cfg_concept = EvaluationConfig.autoconfig(config_concept)

    supported_approaches = ["ica","concept_shap","sae"]
    assert cfg_concept.method_name in supported_approaches, f"Error: The method {cfg_concept.method_name} is not supported in analysis. Currently the only supported methods are {supported_approaches}"

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
    cfg_model.task_args['prompt_tuning'] = False
    hook_model = get_hook_model(cfg_model,tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

    #We retrieve the length of the template added at the end of the sentence
    len_template = cfg_model.len_template
    
    #Check if the acivations and labels have already been cached
    dir_activations_with_labels = os.path.join(cfg_concept.dir_acts_with_labels,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    labels_dataset = dataset_tokenized["token_labels"]
    nb_classes = len(np.unique(np.array(labels_dataset)))

    cache_activations_file_path = os.path.join(dir_activations_with_labels,f'layer_{cfg_concept.hook_layer}.pkl')
    original_text_file_path = os.path.join(dir_activations_with_labels,f'layer_{cfg_concept.hook_layer}.json')
    if not os.path.exists(dir_activations_with_labels):
            os.makedirs(dir_activations_with_labels)

    
    if not os.path.isfile(cache_activations_file_path):

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
        
            activations_dataset, original_text_used = cache_activations_with_labels(hook_model,
                                                                                   dataset_tokenized,
                                                                                   data_collator,
                                                                                   tokenizer,
                                                                                   hook_name = cfg_concept.hook_name,
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

    if cfg_concept.method_name=='concept_shap':

        n_concepts = cfg_concept.methods_args['n_concepts']
        hidden_dim = cfg_concept.methods_args['hidden_dim']
        thres = cfg_concept.methods_args['thres']
        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(hook_model.cfg.device)
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0) #(dataset size, embedding_dim)
            
        interp_model = ConceptNet(n_concepts, embeddings_sentences_tensor, hidden_dim, thres).to(hook_model.cfg.device)
        interp_model.load_state_dict(torch.load(f'{cfg_concept.path_to_dr_methods}_{thres}.pth'))


        # segmented_features is a list of list where list x contains the features the most associated to class x. If cfg_concept.features_selection is True, we additionally run our simple heuristic to filter out features which do not
        #contribute a lot to the accuracy reconstruction i.e. completeness. 'selected_features' thus contain those kept features. 'segmented_features' only segments the kept features.
        segmented_features, selected_features = segments_features(hook_model, interp_model, activations_dataset, labels_dataset, cfg_concept.features_selection, cfg_concept.method_name ,cfg_concept.hook_layer, cfg_concept.hook_name,is_eos = cfg_model.task_args['is_eos'])
        
    elif cfg_concept.method_name=='sae':

        #Load the local trained SAE
        interp_model = TrainingSAE.load_from_pretrained(cfg_concept.sae_path,cfg_concept.methods_args['device'])
        segmented_features, selected_features = segments_features(hook_model, interp_model, activations_dataset, labels_dataset, cfg_concept.features_selection, cfg_concept.method_name, cfg_concept.hook_layer,  cfg_concept.hook_name,is_eos = cfg_model.task_args['is_eos'])
    
    elif cfg_concept.method_name=='ica':

        interp_model = joblib.load(f'{cfg_concept.path_to_dr_methods}.pkl')
        segmented_features, selected_features = segments_features(hook_model, interp_model, activations_dataset, labels_dataset, cfg_concept.features_selection, cfg_concept.method_name, cfg_concept.hook_layer, cfg_concept.hook_name,is_eos = cfg_model.task_args['is_eos'], path_to_dr_methods=cfg_concept.path_to_dr_methods)


    print(f"Final segmented features : {segmented_features}")
    #We save segmented_features, removed_features for recovering accuracy evaluation, ablation studies (causality tests) and interpretability analysis on the test set.
    segmented_features_dir_path = os.path.join(cfg_concept.post_processed_features_path,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name,cfg_concept.method_name)
    if not os.path.exists(segmented_features_dir_path):
            os.makedirs(segmented_features_dir_path)
    #Create file names
    if cfg_concept.method_name=="sae":
        concept_model_name=f'{cfg_concept.sae_name}_layer_{cfg_concept.hook_layer}'
    elif cfg_concept.method_name=="concept_shap":
        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_concept.hook_layer}'
    elif cfg_concept.method_name=='ica':
        concept_model_name=f'ica_{interp_model.components_.shape[0]}_layer_{cfg_concept.hook_layer}'
    segmented_features_file_path = os.path.join(segmented_features_dir_path,f"{concept_model_name}_segmented_features.json")
    selected_features_file_path = os.path.join(segmented_features_dir_path,f"{concept_model_name}_selected_features.json")
    with open(segmented_features_file_path, "w") as f:
        json.dump(segmented_features, f)
    with open(selected_features_file_path, "w") as f:
        json.dump(selected_features, f)