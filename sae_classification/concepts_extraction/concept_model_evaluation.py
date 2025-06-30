from loguru import logger
from ..utils import BaselineMethodConfig, LLMLoadConfig, EvaluationConceptsConfig
from ..llm_classifier_tuning import process_dataset,get_hook_model
from .evaluation_utils import ( 
    ActivationDataset, 
    eval_causal_effect_concepts,
    cache_activations_with_labels, 
    cosine_similarity_concepts, 
    knn_distance_metric, 
    analyze_features,
    eval_hook_loss,
    get_sae_path,
    get_predictions, 
    is_empty_dir,
    load_all_shards
)
from .baseline_method import ConceptNet
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from sae_implementation import TrainingSAE
import os
import torch
import json
import numpy as np
import joblib



def concepts_evaluation(config_concept: str, config_model : str):
 
    # Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    # Retrieve the config of the extracting concept method (centralize for features selection and recovery accuracy, causality and interpretability measures)
    cfg_concept = EvaluationConceptsConfig.autoconfig(config_concept)
    
    supported_approaches = ["ica","concept_shap","sae"]
    assert cfg_concept.method_name in supported_approaches, f"Error: The method {cfg_concept.method_name} is not supported in analysis. Currently the only supported methods are {supported_approaches}"

    # Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Process the dataset on which we will compute the concepts 'quality' i.e the recovery accuracy and causality metrics
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer)

    # Get model hooked (HookedTransformer). Needed to compute the new classifier outputs with the reconstructed hidden states.
    hook_model = get_hook_model(cfg_model,tokenizer)
    
    # Check if we already cached a dataset of the acivations, labels, attention mask....
    dir_dataset_activations = os.path.join(cfg_concept.dir_dataset_activations,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    # Get the labels number
    labels_dataset = dataset_tokenized["true_label"]
    n_sentences = len(labels_dataset)
    unique_labels = np.unique(np.array(labels_dataset))
    nb_classes = len(unique_labels)

    # Get the corresponding labels names
    label_names = cfg_model.match_label_category

    # Match tokens ids and their associated labels
    vocab = tokenizer.get_vocab()
    keys_labels = set(unique_labels)
    labels_tokens_id = { vocab[str(key)] : key for key in keys_labels if str(key) in vocab}
    # Example : { 17: 0, 18: 1 }

    cache_activations_dir = os.path.join(dir_dataset_activations,f'layer_{cfg_concept.hook_layer}')
    if not os.path.exists(cache_activations_dir):
            os.makedirs(cache_activations_dir)

    # Check if we already cached a dataset of the acivations, labels, attention mask....
    if is_empty_dir(cache_activations_dir):


        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
        original_text_used = cache_activations_with_labels(hook_model,
                                                            dataset_tokenized,
                                                            data_collator,
                                                            tokenizer,
                                                            hook_name = cfg_concept.hook_name,
                                                            labels_tokens_id=labels_tokens_id,
                                                            eos = cfg_model.task_args['eos'],
                                                            path_directory=cache_activations_dir)

        original_text_file_path = os.path.join(cache_activations_dir,f'original_text.json')
        with open(original_text_file_path, 'w') as f:
            json.dump(original_text_used, f)


    activations_dataset = load_all_shards(cache_activations_dir)
    original_text_file_path = os.path.join(cache_activations_dir,f'original_text.json')
    with open(original_text_file_path, 'r') as f:
        original_text_used = json.load(f)    

    device = cfg_concept.methods_args['device']


    logger.info(f"Loading the evaluated concept-based method selected : method type : {cfg_concept.method_name}")
    # Set up everything to load the evaluated concept-based method
    if cfg_concept.method_name=='concept_shap':
        method_name = cfg_concept.method_name

        n_concepts = cfg_concept.methods_args['n_concepts']
        hidden_dim = cfg_concept.methods_args['hidden_dim']
        thres = cfg_concept.methods_args['thres']
        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(device)
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0) #(dataset size, embedding_dim)
        
        interp_model = ConceptNet(n_concepts, embeddings_sentences_tensor, hidden_dim, thres).to(device)
        conceptnet_weights_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}_{thres}',f'conceptshap_weights.pth')
        
        try:
            interp_model.load_state_dict(torch.load(conceptnet_weights_path,weights_only=True))
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: '{conceptnet_weights_path}' not found. Ensure you have trained the corresponding ConceptNet before evaluation of its concepts with 'train-baseline'")

        matrix_concepts = interp_model.concept #(embedding_dim,n_components)

        mixing_matrix=None

        # Name of the specifc concept-based method used
        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_concept.hook_layer}'
    
    elif cfg_concept.method_name=='ica': 

        method_name = cfg_concept.method_name
        ica_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}',f'ica.pkl')
        mixing_matrix_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}',f'mixing_matrix.npy')
        

        try : 
            interp_model = joblib.load(ica_path)
            mixing_matrix = np.load(mixing_matrix_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Either '{ica_path}' or '{mixing_matrix_path}' not found. Ensure you have fitted the corresponding ICA before evaluation of its concepts with 'train-baseline'")


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

        # Load the local trained SAE
        interp_model = TrainingSAE.load_from_pretrained(sae_path,device)

        mixing_matrix=None

        matrix_concepts = interp_model.W_dec.T #(embedding_dim,n_components)


    concepts_dir_path = os.path.join(cfg_concept.post_processed_features_path,cfg_model.model_name,cfg_model.dataset_name,method_name,concept_model_name,f"{cfg_concept.methods_args['n_concepts']}_concepts")
    
    # Retrieve the results from the post-processing of the concepts
    segmented_features_file_path = os.path.join(concepts_dir_path,f"segmented_features.json")
    selected_features_file_path = os.path.join(concepts_dir_path,f"selected_features.json")
    
    if not os.path.isfile(segmented_features_file_path):
        raise FileNotFoundError(f"File {segmented_features_file_path} does not exist. Make sure to run 'post-process-concepts' before running the analysis of the extracted concepts")
    if not os.path.isfile(selected_features_file_path):
        raise FileNotFoundError(f"File {selected_features_file_path} does not exist. Make sure to run 'post-process-concepts' before running the analysis of the extracted concepts")
    
    with open(selected_features_file_path, "r") as file:
        selected_features = json.load(file)
    with open(segmented_features_file_path, "r") as file:
        segmented_features = json.load(file)

    print(f"segmented_features : {segmented_features}")

    
    # Display the cosine similarity properties between the learned concepts
    cosine_similarity_concepts(matrix_concepts[:,selected_features],0.95)

    '''
    Global and local diversity metrics on the selected concept vectors
    The distance measure selected is '1 - cosine similarity'

    Local diversity metric :    
        * Retrieve the distance of each concept vector to its nearest neighbor and average the results over all selected concept vectors.
        * 1‑nearest‑neighbour distance (Clark–Evans nearest‑neighbour index )
        * This is a measure of local diversity for the concepts "are the concept vectors near‑duplicates?"

    Global diversity metric :
        * Average the distance for every pair of concept vectors
        * Mean pairwise distance
        * How widely does the whole set spread ?
    '''
    
    local_diversity, global_diversity = knn_distance_metric(matrix_concepts.T[selected_features,:])
    print(f"Average 1-nearest neighbour distance across the concepts : {local_diversity}")
    print(f"Average pairwise distance across the concepts : {global_diversity}")



    # Forward pass on the tested dataset to retrieve statistics on the concepts activations (features activation rate, mean activations per feature...) and recovery accuracy, recovery recall and precision for each class
    classification_loss_dict, mean_activations,dict_analysis_concepts  = eval_hook_loss(hook_model,
                                                                                        interp_model,
                                                                                        activations_dataset,
                                                                                        selected_features,
                                                                                        cfg_concept.method_name,
                                                                                        cfg_concept.hook_layer,
                                                                                        cfg_concept.hook_name,
                                                                                        device,
                                                                                        labels_tokens_id,
                                                                                        mixing_matrix)

    zero_count = torch.sum(mean_activations == 0).item()
    print(f"zero_count : {zero_count}")
        

    # Save the 'diversity metrics of the concepts
    classification_loss_dict["local_diversity_metric"] = local_diversity
    classification_loss_dict["global_diversity_metric"] = global_diversity


    dir_to_save_metrics = os.path.join(concepts_dir_path,cfg_model.split)
    #Create the directory where to save the reconstruction results if it does not exist
    if not os.path.exists(dir_to_save_metrics):
        os.makedirs(dir_to_save_metrics)
    file_to_save_metrics = os.path.join(dir_to_save_metrics,"metrics.json") 
    with open(file_to_save_metrics, 'w') as file:
        json.dump(classification_loss_dict, file, indent=4)

    dir_to_save_activations = os.path.join(cfg_concept.activations_interpretability_methods_post_analysis,cfg_model.model_name,cfg_model.dataset_name,method_name,concept_model_name)
    # Create the directory where to save the activations if it does not exist
    if not os.path.exists(dir_to_save_activations):
        os.makedirs(dir_to_save_activations)
    file_to_save_activations = os.path.join(dir_to_save_activations,f"evaluation_activations.pth") 
    torch.save(dict_analysis_concepts, file_to_save_activations)

    # If the evaluated method is a sparse autoencoder, we create the figure of the projected SAE directions onto the 2 main PCA components of the activation space 
    if cfg_concept.method_name=="sae":
        fig_pca = analyze_features(hook_model,interp_model,mean_activations,dict_analysis_concepts,label_names)
        dir_pca_figure = os.path.join(dir_to_save_metrics,"pca_figures")
        if not os.path.exists(dir_pca_figure):
            os.makedirs(dir_pca_figure)
        file_to_save_pca = os.path.join(dir_pca_figure,f"2D_PCA_SAE_directions.pdf")
        fig_pca.savefig(file_to_save_pca, dpi=300, bbox_inches='tight')


    # We run the ablation tests if specified 
    if cfg_concept.test_causality:

        #Create the directory if it does not exist
        dir_ablation_results = os.path.join(dir_to_save_metrics,"causality")
        if not os.path.exists(dir_ablation_results):
            os.makedirs(dir_ablation_results)

        # We ablate different levels of each class-specific features segments  (clustered by their activations with regard to the class) and average the global accuracy variation and total variation distance over the segments          
        # Size of the ablation (in percentage)
        ablation_sizes = torch.tensor([25,50,75,100])

        # We first run the model with all the features on, then with disabled features
        predicted_probabilities, accuracy, dict_metrics_no_ablation = get_predictions(hook_model,
                                                                                   interp_model,
                                                                                   activations_dataset,
                                                                                   selected_features,
                                                                                   cfg_concept.method_name,
                                                                                   cfg_concept.hook_layer,
                                                                                   cfg_concept.hook_name,
                                                                                   perturb=None,
                                                                                   device=device,
                                                                                   labels_tokens_id=labels_tokens_id,
                                                                                   ids_samples=torch.arange(n_sentences),
                                                                                   mixing_matrix=mixing_matrix)

        file_to_save_preds_no_ablation = os.path.join(dir_ablation_results,f"metrics_no_ablation.json") 
        with open(file_to_save_preds_no_ablation, 'w') as file:
            json.dump(dict_metrics_no_ablation, file, indent=4)

        list_mean_tvd = []
        list_mean_absolute_accuracy_change = []
        list_mean_label_flip_rate = []   

        for ablation_size in ablation_sizes:

            overall_global_absolute_accuracy_change = 0
            overall_global_tvd = 0
            overall_global_label_flip_rate = 0

            # We ablate features from the same class segment for each class-specific features segments to maximize the accuracy variation
            for c in range(nb_classes):

                number_ablated_features = int(ablation_size * len(segmented_features[c]) / 100)
                # The features in each sublist of the list segmented_features are sorted in decreasing order based on their mean activations on the training dataset
                list_ablated_features = [int(feature_number) for feature_number in segmented_features[c][:number_ablated_features]]
                variation_absolute_accuracy, tvd, label_flip_rate, dict_metrics_ablation = eval_causal_effect_concepts(
                                                                                                hook_model,
                                                                                                interp_model,
                                                                                                activations_dataset,
                                                                                                selected_features,
                                                                                                cfg_concept.method_name,
                                                                                                cfg_concept.hook_layer,
                                                                                                cfg_concept.hook_name,
                                                                                                predicted_probabilities,
                                                                                                accuracy,
                                                                                                c,
                                                                                                device,
                                                                                                labels_tokens_id,
                                                                                                list_ablated_features,
                                                                                                mixing_matrix=mixing_matrix)

                overall_global_absolute_accuracy_change += variation_absolute_accuracy
                overall_global_tvd += tvd
                overall_global_label_flip_rate += label_flip_rate
                
                dict_ablation_metrics = {
                                       'category' : label_names[str(c)],
                                       'Percentage ablation of the segment' : ablation_size.item(),
                                       'Ablated features' : list_ablated_features, 
                                       'Global Absolute Accuracy change' : variation_absolute_accuracy, 
                                       'Global Total Variation Distance with regard to original predictions' : tvd,
                                       'Global label-flip rate' : label_flip_rate}


                dict_detailed_metrics = dict_ablation_metrics | dict_metrics_ablation
                
                file_to_save_preds_ablation = os.path.join(dir_ablation_results,f"metrics_ablation_category_{label_names[str(c)]}_{ablation_size}_%.json") 
                with open(file_to_save_preds_ablation, 'w') as file:
                    json.dump(dict_detailed_metrics, file, indent=4)


            overall_global_absolute_accuracy_change/=nb_classes
            overall_global_tvd/=nb_classes
            overall_global_label_flip_rate/=nb_classes
            print(f"Averaged Global Absolute Accuracy change over all the segments (for an ablation of {ablation_size} % on a segment) : {overall_global_absolute_accuracy_change}")
            print(f"Averaged Global TVD over all the segments (for an ablation of {ablation_size} % on a segment) : {overall_global_tvd}")
            print(f"Averaged Global Label-flip rate over all the segments (for an ablation of {ablation_size} % on a segment) : {overall_global_label_flip_rate}")
            list_mean_absolute_accuracy_change.append(overall_global_absolute_accuracy_change)
            list_mean_tvd.append(overall_global_tvd)
            list_mean_label_flip_rate.append(overall_global_label_flip_rate)
            

        # Now we look more precisely at the percentage of flip predictions resulting in the ablation of a single feature when this one is normally activated
        # It means that we focus now on conditional causality. In that context, we ablate one feature at a time.
        overall_conditional_label_flip_rate = []
        overall_conditional_absolute_accuracy_change = []
        overall_conditional_tvd = []
        for feature in selected_features:

            # We retrieve the sentences where the feature was activated
            activating_samples = (dict_analysis_concepts['concepts activations'][:, feature].abs() > 1e-8).nonzero(as_tuple=True)[0]
            if activating_samples.shape[0]==0:
                continue
            true_labels_samples = dict_analysis_concepts['true labels'][activating_samples]
            # Get the outputs for these sentences
            predicted_labels = torch.argmax(predicted_probabilities[activating_samples,:], dim=1)
            original_accuracy_on_selected_samples = ((true_labels_samples==predicted_labels).sum() / len(activating_samples)).item()

            
            probs_pred_ablation, accuracy_ablation , _ = get_predictions(hook_model,
                                                        interp_model,
                                                        activations_dataset,
                                                        selected_features,
                                                        cfg_concept.method_name,
                                                        cfg_concept.hook_layer,
                                                        cfg_concept.hook_name,
                                                        perturb=[feature],
                                                        device=device,
                                                        labels_tokens_id=labels_tokens_id,
                                                        ids_samples=activating_samples,
                                                        mixing_matrix=mixing_matrix)

            predicted_labels_ablation = torch.argmax(probs_pred_ablation, dim=1)
            # Compute the conditional label flip rate (percentage of changes in label predicted on samples activating the investigated feature)
            conditional_label_flip_rate = ((predicted_labels!=predicted_labels_ablation).sum()/predicted_labels.shape[0]).item()
            # Compute the conditional absolute acccuracy variation
            conditional_absolute_accuracy_variation = (accuracy_ablation - original_accuracy_on_selected_samples)
            # Compute the conditional TVD
            conditional_tvd = 0.5 * torch.sum(torch.abs(predicted_probabilities[activating_samples,:] - probs_pred_ablation), dim=1).mean().item()
            overall_conditional_label_flip_rate.append(conditional_label_flip_rate)
            overall_conditional_tvd.append(conditional_tvd)
            overall_conditional_absolute_accuracy_change.append(conditional_absolute_accuracy_variation)

        print(f"Averaged conditional label-flip rate : {np.mean(overall_conditional_label_flip_rate)}")
        print(f"Averaged Conditional TVD : {np.mean(overall_conditional_tvd)}")
        print(f"Averaged Conditional Absolute Accuracy change : {np.mean(overall_conditional_absolute_accuracy_change)}")

        dict_averaged_results_ablation = { 'Proportion of ablation per segment' : ablation_sizes.tolist(), 
                                           'Global (averaged over categories) absolute accuracy change' :  list_mean_absolute_accuracy_change, 
                                           'Global  (averaged over categories) TVD' : list_mean_tvd, 
                                           'Global  (averaged over categories) Label-flip rate' : list_mean_label_flip_rate, 
                                           'Averaged (over features) conditional label-flip rate' : sum(overall_conditional_label_flip_rate)/len(overall_conditional_label_flip_rate), 
                                           'Averaged (over features) Conditional TVD' : sum(overall_conditional_tvd)/len(overall_conditional_tvd),
                                           'Averaged (over features) Conditional Absolute Accuracy change' : sum(overall_conditional_absolute_accuracy_change)/len(overall_conditional_absolute_accuracy_change),
                                           }
        file_to_save_ablation_averaged_results = os.path.join(dir_ablation_results,f"metrics_ablation_averaged_results.json") 
        with open(file_to_save_ablation_averaged_results, 'w') as file:
            json.dump(dict_averaged_results_ablation, file, indent=4)
            
            
            

            




    

    

    

    
