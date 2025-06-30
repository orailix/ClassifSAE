from loguru import logger
from ..utils import DRMethodsConfig, LLMLoadConfig, EvaluationConfig
import pickle
from ..model_training import process_dataset,get_hook_model,compute_loss_last_token, PromptTunerForHookedTransformer
from .evaluation_utils import compute_loss_last_token_classif, update_metrics, count_same_match, ActivationDataset, cache_activations_with_labels, cosine_similarity_concepts, knn_distance_metric, analyze_features, eval_hook_loss, get_predictions, eval_causal_effect_concepts, filter_highly_similar_columns
from .concept_shap import ConceptNet
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import Dataset
from torch.utils.data import DataLoader
from sae_implementation import TrainingSAE
import os
import sys
import torch
import json
import numpy as np
import joblib
sys.path.append("../../")





def concepts_evaluation(config_concept: str, config_model : str):
 
    #Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
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
    
    #Dataset used for interpretability
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer)

    cfg_model.task_args['prompt_tuning'] = False
    hook_model = get_hook_model(cfg_model,tokenizer)

    #We retrieve the length of the template added at the end of the sentence
    len_template = cfg_model.len_template
    
    #Check if the acivations and labels have already been cached
    dir_activations_with_labels = os.path.join(cfg_concept.dir_acts_with_labels,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    labels_dataset = dataset_tokenized["token_labels"]
    nb_classes = len(np.unique(np.array(labels_dataset)))
    label_names = cfg_model.match_label_category

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


    #Load the selected features
    segmented_features_dir_path = os.path.join(cfg_concept.post_processed_features_path,"test",cfg_model.model_name,cfg_model.dataset_name,cfg_concept.method_name)

     #Create file names
    if cfg_concept.method_name=="sae":
        concept_model_name=f'{cfg_concept.sae_name}_layer_{cfg_concept.hook_layer}'
    elif cfg_concept.method_name=="concept_shap":
        concept_model_name=f'concept_shap_{cfg_concept.methods_args["n_concepts"]}_{cfg_concept.methods_args["thres"]}_layer_{cfg_concept.hook_layer}'
    elif cfg_concept.method_name=='ica':
        concept_model_name=f'ica_{cfg_concept.methods_args["n_components"]}_layer_{cfg_concept.hook_layer}'
    segmented_features_file_path = os.path.join(segmented_features_dir_path,f"{concept_model_name}_segmented_features.json")
    selected_features_file_path = os.path.join(segmented_features_dir_path,f"{concept_model_name}_selected_features.json")
    
    if not os.path.isfile(selected_features_file_path):
        raise FileNotFoundError(f"File {selected_features_file_path} does not exist. Make sure to run 'select-features' before running the analysis of the extracted concepts")
    with open(selected_features_file_path, "r") as file:
        selected_features = json.load(file)
    with open(segmented_features_file_path, "r") as file:
        segmented_features = json.load(file)

    print(f"segmented_features : {segmented_features}")

    path_to_dr_methods=''
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

        matrix_concepts = interp_model.concept #(embedding_dim,n_components)
        path_to_dr_methods = cfg_concept.path_to_dr_methods

        
    elif cfg_concept.method_name=='sae':

        #Load the local trained SAE
        interp_model = TrainingSAE.load_from_pretrained(cfg_concept.sae_path,cfg_concept.methods_args['device'])
        matrix_concepts = interp_model.W_dec.T #(embedding_dim,n_components)
        
     
    elif cfg_concept.method_name=='ica':

        interp_model = joblib.load(f'{cfg_concept.path_to_dr_methods}.pkl')
        matrix_concepts = interp_model.components_.T #(embedding_dim,n_components)
        matrix_concepts = torch.from_numpy(matrix_concepts)
        path_to_dr_methods = cfg_concept.path_to_dr_methods

    
    #Display the cosine similarity properties between the learned concepts
    cosine_similarity_concepts(matrix_concepts[:,selected_features],0.95)

    # indices_selected_features = filter_highly_similar_columns(matrix_concepts[:,selected_features], threshold=0.95)
    # selected_features = [selected_features[index] for index in indices_selected_features]
    # segmented_features = [[x for x in sublist if x in selected_features] for sublist in segmented_features]

    #cosine_similarity_concepts(matrix_concepts[:,selected_features],0.95)
    

    #Look at the mean k-NN distance of each concept vector. The distance is computed as 1 minus the cosine similarity of the pair of vector. 
    k=5
    mean_knn_distance = knn_distance_metric(matrix_concepts.T[selected_features,:],k=k)
    print(f"Mean k-NN Distance across the concepts : {mean_knn_distance} with k={k}")


    #Forward pass on the test dataset
    classification_loss_dict, new_activations_dataset, mean_activations,dict_analysis_features  = eval_hook_loss(hook_model,
                                                                                                                            interp_model,
                                                                                                                            labels_dataset,
                                                                                                                            original_text_used,
                                                                                                                            tokenizer,
                                                                                                                            activations_dataset,
                                                                                                                            selected_features,
                                                                                                                            cfg_concept.method_name,
                                                                                                                            cfg_concept.hook_layer,
                                                                                                                            cfg_concept.hook_name,
                                                                                                                            cfg_model.task_args['is_eos'],
                                                                                                                            path_to_dr_methods)

    zero_count = torch.sum(mean_activations == 0).item()
    print(f"zero_count : {zero_count}")
        

    #Save the 'diversity metric of the features
    classification_loss_dict["diversity_distance"] = mean_knn_distance
    
    #Save metrics performance dict as json
    if cfg_concept.method_name=="sae":
        concept_model_name=f'{cfg_concept.sae_name}_layer_{cfg_concept.hook_layer}'
    elif cfg_concept.method_name=="concept_shap":
        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_concept.hook_layer}'
    elif cfg_concept.method_name=='ica':
        concept_model_name=f'ica_{interp_model.components_.shape[0]}_layer_{cfg_concept.hook_layer}'

    dir_to_save_metrics = os.path.join(cfg_concept.metrics_reconstruction,cfg_model.model_name,cfg_model.dataset_name,cfg_concept.method_name,concept_model_name)
    #Create the directory where to save the reconstruction results if it does not exist
    if not os.path.exists(dir_to_save_metrics):
        os.makedirs(dir_to_save_metrics)
    file_to_save_metrics = os.path.join(dir_to_save_metrics,f"{concept_model_name}.json") 
    with open(file_to_save_metrics, 'w') as file:
        json.dump(classification_loss_dict, file, indent=4)

    dir_to_save_activations = os.path.join(cfg_concept.activations_interpretability_methods,cfg_model.model_name,cfg_model.dataset_name,cfg_concept.method_name,concept_model_name)
    #Create the directory where to save the activations if it does not exist
    if not os.path.exists(dir_to_save_activations):
        os.makedirs(dir_to_save_activations)
    file_to_save_activations = os.path.join(dir_to_save_activations,f"{concept_model_name}.pth") 
    torch.save(dict_analysis_features, file_to_save_activations)

    #If the evaluated method is a sparse autoencoder, we create the figure of the projected SAE directions onto the 2 main PCA components of the activation space 
    if cfg_concept.method_name=="sae":
        fig_pca = analyze_features(hook_model,interp_model,mean_activations,dict_analysis_features,label_names)
        dir_pca_figure = os.path.join(dir_to_save_metrics,"pca_figures")
        if not os.path.exists(dir_pca_figure):
            os.makedirs(dir_pca_figure)
        file_to_save_pca = os.path.join(dir_pca_figure,f"{cfg_concept.sae_name}.pdf")
        fig_pca.savefig(file_to_save_pca, dpi=300, bbox_inches='tight')


    #We run the ablation tests if specified 
    if cfg_concept.test_causality:

        #Create the directory if it does not exist
        dir_ablation_results = os.path.join(dir_to_save_metrics,"ablation")
        if not os.path.exists(dir_ablation_results):
            os.makedirs(dir_ablation_results)

        #We ablate different pecentages of each segment (clustered by their activations with regard the class) and average the absolute accuracy variation and total variation distance over the segments          
        #Size of the ablation (in percentage)
        ablation_sizes = torch.tensor([10,25,50,75,100])

        #We first run the model with all the features on, then with disabled features
        predicted_probabilities, accuracy, dict_metrics_no_ablation = get_predictions(hook_model,
                                                                                   interp_model,
                                                                                   labels_dataset,
                                                                                   tokenizer,
                                                                                   new_activations_dataset,
                                                                                   selected_features,
                                                                                   cfg_concept.method_name,
                                                                                   cfg_concept.hook_layer,
                                                                                   cfg_concept.hook_name,
                                                                                   cfg_model.task_args['is_eos'],
                                                                                   perturb=None,
                                                                                   ids_samples=torch.arange(len(labels_dataset)),
                                                                                   path_to_dr_methods=path_to_dr_methods)

        file_to_save_preds_no_ablation = os.path.join(dir_ablation_results,f"{concept_model_name}_no_ablation.json") 
        with open(file_to_save_preds_no_ablation, 'w') as file:
            json.dump(dict_metrics_no_ablation, file, indent=4)

        list_mean_tvd = []
        list_mean_variation_absolute_accuracy = []

        for ablation_size in ablation_sizes:

            global_mean_variation_absolute_accuracy = 0
            global_mean_tvd = 0

            #We ablate relatively to each segment's class to maximize the accuracy variation
            for c in range(nb_classes):

                number_ablated_features = int(ablation_size * len(segmented_features[c]) / 100)
                #The features in each sublist of the list segmented_features are sorted in decreasing order based on their mean activations on the training dataset
                list_ablated_features = [int(feature_number) for feature_number in segmented_features[c][:number_ablated_features]]
                variation_absolute_accuracy, tvd, dict_metrics_ablation = eval_causal_effect_concepts(predicted_probabilities,
                                                                                                      accuracy,
                                                                                                      c,
                                                                                                      hook_model,
                                                                                                      interp_model,
                                                                                                      list_ablated_features,
                                                                                                      labels_dataset,
                                                                                                      tokenizer,
                                                                                                      new_activations_dataset,
                                                                                                      selected_features,
                                                                                                      cfg_concept.method_name,
                                                                                                      cfg_concept.hook_layer,
                                                                                                      cfg_concept.hook_name,
                                                                                                      cfg_model.task_args['is_eos'],
                                                                                                      path_to_dr_methods)

                global_mean_variation_absolute_accuracy += variation_absolute_accuracy
                global_mean_tvd += tvd
                
                dict_ablation_metrics = {
                                       'category' : label_names[str(c)],
                                       'Percentage ablation of the segment' : ablation_size.item(),
                                       'Ablated features' : list_ablated_features, 
                                       'Variation Absolute Accuracy' : variation_absolute_accuracy, 
                                       'Total Variation Distance with regard to original predictions' : tvd}

                # print(f"type(variation_absolute_accuracy) : {type(variation_absolute_accuracy)}")
                # print(f"type(tvd) : {type(tvd)}")
                # print(f"type(label_names[str(c)]) : {type(label_names[str(c)])}")
                # print(f"type(ablation_size) : {type(ablation_size)}")

                dict_detailed_metrics = dict_ablation_metrics | dict_metrics_ablation
                
                file_to_save_preds_ablation = os.path.join(dir_ablation_results,f"{concept_model_name}_ablation_category_{label_names[str(c)]}_{ablation_size}_%.json") 
                with open(file_to_save_preds_ablation, 'w') as file:
                    json.dump(dict_detailed_metrics, file, indent=4)


            global_mean_variation_absolute_accuracy/=nb_classes
            global_mean_tvd/=nb_classes
            print(f"Averaged absolute accuracy change over all the segments (for an ablation of {ablation_size} % on a segment) : {global_mean_variation_absolute_accuracy}")
            print(f"Averaged TVD over all the segments (for an ablation of {ablation_size} % on a segment) : {global_mean_tvd}")
            list_mean_tvd.append(tvd)
            list_mean_variation_absolute_accuracy.append(global_mean_tvd)
            

        #Now we look more precisely at the percentage of flip resulting in the ablation of a single feature when this one is normally activated
        overall_percentage_flip = []
        overall_restricted_tvd = []
        for feature in selected_features:

            #We retrieve the samples where the feature was activated
            activating_samples = (dict_analysis_features['feature activation'][:, feature] > 0).nonzero(as_tuple=True)[0]
            #print(f"activating_samples : {activating_samples}")
            if activating_samples.shape[0]==0:
                continue
            #Get the outputs for these samples
            predicted_labels = torch.argmax(predicted_probabilities[activating_samples,:], dim=1)

            probs_pred_ablation, _ , _ = get_predictions(hook_model,
                                                                        interp_model,
                                                                        labels_dataset,
                                                                        tokenizer,
                                                                        new_activations_dataset,
                                                                        selected_features,
                                                                        cfg_concept.method_name,
                                                                        cfg_concept.hook_layer,
                                                                        cfg_concept.hook_name,
                                                                        cfg_model.task_args['is_eos'],
                                                                        perturb=[feature],
                                                                        ids_samples=activating_samples,
                                                                        path_to_dr_methods=path_to_dr_methods)

            predicted_labels_ablation = torch.argmax(probs_pred_ablation, dim=1)
            #Compute the percentage of flip (number of changes in label predicted)
            percentage_flip = ((predicted_labels!=predicted_labels_ablation).sum()/predicted_labels.shape[0]).item()
            restricted_tvd = 0.5 * torch.sum(torch.abs(predicted_probabilities[activating_samples,:] - probs_pred_ablation), dim=1).mean().item()
            # print(f"percentage_flip : {percentage_flip}")
            # print(f"restricted_tvd : {restricted_tvd}")
            overall_percentage_flip.append(percentage_flip)
            overall_restricted_tvd.append(restricted_tvd)

        print(f"Averaged Percentage flip : {np.mean(overall_percentage_flip)}")
        print(f"Averaged Restricted TVD : {np.mean(overall_restricted_tvd)}")

        dict_averaged_results_ablation = { 'Proportion of ablation per segment' : ablation_sizes.tolist(), 'Averaged absolute accuracy change' :  list_mean_variation_absolute_accuracy, 'Averaged TVD' : list_mean_tvd, 'Averaged Percentage flip' : sum(overall_percentage_flip)/len(overall_percentage_flip), 'Averaged Restricted TVD' : sum(overall_restricted_tvd)/len(overall_restricted_tvd)}
        file_to_save_ablation_averaged_results = os.path.join(dir_ablation_results,f"{concept_model_name}_ablation_averaged_results.json") 
        with open(file_to_save_ablation_averaged_results, 'w') as file:
            json.dump(dict_averaged_results_ablation, file, indent=4)
            
            
            

            




    

    

    

    
