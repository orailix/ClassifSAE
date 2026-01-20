from loguru import logger
from ..utils import LLMLoadConfig, EvaluationConceptsConfig
from ..llm_classifier_tuning import get_hook_model, get_model, _init_tokenizer, _max_seq_length
from .evaluation_utils import ( 
    analyze_features,
    eval_hook_loss,
    get_predictions, 
    is_empty_dir,
    load_all_shards,
    load_interp_model,
    create_activations_dataset,
    eval_causal_effect_concepts,
)
from .caching import max_seq_length_hook
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
import os
import torch
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt




def concepts_evaluation(config_concept: str, config_model : str):

    print("\n######################################## BEGIN : Computation of Completeness Metrics on z_class Concepts ########################################")

 
    # Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    # Retrieve the config of the extracting concept method (centralize for features selection and recovery accuracy, causality and interpretability measures)
    cfg_concept = EvaluationConceptsConfig.autoconfig(config_concept)
    
    supported_approaches = ["ica","concept_shap","sae","hi_concept"]
    if cfg_concept.method_name not in supported_approaches:
        raise ValueError(
            f"Unsupported method '{cfg_concept.method_name}'. "
            f"Supported: {supported_approaches}"
        )
    logger.info(f"Concept method: {cfg_concept.method_name}")
    
    # Tokenizer, Model Init
    tokenizer, decoder = _init_tokenizer(cfg_model)
    nb_classes = len(cfg_model.match_label_category)
    unique_labels = np.sort(np.array([int(k) for k in cfg_model.match_label_category.keys()]))

    
    if decoder: # (causal LM with template prompting for classification)
        model = get_hook_model(cfg_model,tokenizer)
        vocab = tokenizer.get_vocab()
        keys_labels = set(unique_labels)
        labels_tokens_id = { vocab[str(key)] : key for key in keys_labels if str(key) in vocab}
        device = model.cfg.device

        max_ctx = max_seq_length_hook(model)
        add_template = decoder
        eos = cfg_model.task_args['eos']
        if eos is None:
            eos = False
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

    else: # (Encoder only LM for classification)
        model = get_model(cfg_model,decoder,nb_classes) 
        labels_tokens_id = None
        device = model.device

        max_ctx = _max_seq_length(model)
        add_template = decoder
        eos = False
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        
    # Get the corresponding labels names
    label_names = cfg_model.match_label_category
    
    # Check if already cached a dataset of the activations, labels, attention mask....
    dir_dataset_activations = os.path.join(cfg_concept.dir_dataset_activations,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)
    cache_activations_dir = os.path.join(dir_dataset_activations,f'layer_{cfg_concept.hook_layer}')
    os.makedirs(cache_activations_dir, exist_ok=True)
    
    # Build cache if empty
    if is_empty_dir(cache_activations_dir):
        
            logger.info(f"Activation cache not found—creating it from the {cfg_model.split} split…")

            # If not, create a dataset with the activations from the given split that we save to speed up the causality metrics computation
            create_activations_dataset(
                cfg_model=cfg_model,
                cfg_concept=cfg_concept,
                model=model,
                tokenizer=tokenizer,
                data_collator=data_collator,
                decoder=decoder,
                add_template=add_template,
                max_ctx=max_ctx,
                device=device,
                cache_activations_dir=cache_activations_dir,
                labels_tokens_id=labels_tokens_id,
                eos=eos
            )
 
    # Load cached activations 
    activations_dataset = load_all_shards(cache_activations_dir)
    
    # Original text/ sentences (without template)
    original_text_used = []
    original_text_file_path = Path(cache_activations_dir) / "original_text.json"
    if original_text_file_path.exists():
        try:
            original_text_used = json.loads(original_text_file_path.read_text())
            logger.debug(f"Loaded {len(original_text_used)} sentences from cache.")
        except Exception as e:
            logger.warning(f"Could not read original_text.json: {e}")
    # Fallback if empty: derive from dataset length
    n_sentences = len(original_text_used) if original_text_used else len(activations_dataset)
    
    
    # Load interpretation model
    logger.info(f"Loading model for concept-based method={cfg_concept.method_name}")
    interp_model, _, mixing_matrix, method_name, concept_model_name, device_interp_model = load_interp_model(
                                                                                                cfg_concept=cfg_concept,
                                                                                                activations_dataset=activations_dataset,
                                                                                                decoder=decoder,
                                                                                                device=device
                                                                                            )
   

    # Load path where we saved the features indices of z_class
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
        if selected_features==[]:
                raise ValueError(
                    "selected_features exists but is empty. Post-processing found only dead/zero activations "
                    "on the evaluated data; the evaluated method appears to have failed."
                )

    with open(segmented_features_file_path, "r") as file:
        segmented_features = json.load(file)
        
    logger.info(f"z_class features: {len(selected_features)} | "
            f"Categories Segment sizes: {[len(s) for s in segmented_features]}")


    # Forward pass through z_class bottleneck to retrieve statistics on the concepts activations (features activation rate, mean activations per feature...)
    # and recovery accuracy, recovery recall and precision for each class
    classification_loss_dict, mean_activations, dict_analysis_concepts = eval_hook_loss(
                                                                            decoder=decoder,
                                                                            model=model,
                                                                            interp_model=interp_model,
                                                                            activations_dataset=activations_dataset,
                                                                            selected_features=selected_features,
                                                                            method_name=cfg_concept.method_name,
                                                                            hook_layer=cfg_concept.hook_layer,
                                                                            hook_name=cfg_concept.hook_name,
                                                                            device=device_interp_model,
                                                                            labels_tokens_id=labels_tokens_id,
                                                                            mixing_matrix=mixing_matrix,
                                                                        )

    # Save reconstructions results
    dir_to_save_metrics = os.path.join(concepts_dir_path,cfg_model.split)
    # Create the directory where to save the reconstruction results if it does not exist
    os.makedirs(dir_to_save_metrics, exist_ok=True)
    file_to_save_metrics = os.path.join(dir_to_save_metrics,"metrics.json") 
    with open(file_to_save_metrics, 'w') as file:
        json.dump(classification_loss_dict, file, indent=4)

    # Save original, z_class and reconstructed activations for post-analysis
    dir_to_save_activations = os.path.join(cfg_concept.activations_interpretability_methods_post_analysis,cfg_model.model_name,cfg_model.dataset_name,method_name,concept_model_name)
    os.makedirs(dir_to_save_activations, exist_ok=True)
    file_to_save_activations = os.path.join(dir_to_save_activations,f"evaluation_activations.pth") 
    torch.save(dict_analysis_concepts, file_to_save_activations)

    # If the evaluated method is a sparse autoencoder, create the figure of the projected SAE directions onto the 2 main PCA components of the activation space 
    if cfg_concept.method_name=="sae":
        fig_pca = analyze_features(interp_model,mean_activations,dict_analysis_concepts,label_names)
        dir_pca_figure = os.path.join(dir_to_save_metrics,"pca_figures")
        os.makedirs(dir_pca_figure, exist_ok=True)
        file_to_save_pca = os.path.join(dir_pca_figure,f"2D_PCA_SAE_directions.pdf")
        fig_pca.savefig(file_to_save_pca, dpi=300, bbox_inches='tight')
        plt.close(fig_pca)


    print("######################################## END : Computation of Completeness Metrics on z_class Concepts ########################################\n")


    # Run the ablation tests if specified 
    if cfg_concept.test_causality:

        print("\n######################################## BEGIN : Computation of Causality Metrics on z_class Concepts ########################################")


        #Create the directory if it does not exist
        dir_ablation_results = os.path.join(dir_to_save_metrics,"causality")
        os.makedirs(dir_ablation_results, exist_ok=True)
        
        # No-ablation baseline
        predicted_probabilities, accuracy, dict_metrics_no_ablation = get_predictions(
                decoder=decoder,
                model=model,
                interp_model=interp_model,
                activations_dataset=activations_dataset,
                selected_features=selected_features,
                method_name=cfg_concept.method_name,
                hook_layer=cfg_concept.hook_layer,
                hook_name=cfg_concept.hook_name,
                perturb=None,
                device=device_interp_model,
                labels_tokens_id=labels_tokens_id,
                ids_samples=torch.arange(n_sentences),
                mixing_matrix=mixing_matrix,
        )
       
      
        file_to_save_preds_no_ablation = os.path.join(dir_ablation_results,f"metrics_no_ablation.json") 
        with open(file_to_save_preds_no_ablation, 'w') as file:
            json.dump(dict_metrics_no_ablation, file, indent=4)

        # Ablate different levels of each class-specific features segments  (clustered by their activations with regard to the class) and average the global accuracy variation and total variation distance over the segments          
        # Size of the ablation (in percentage)
        ablation_sizes = [25,50,75,100]


        list_mean_tvd = []
        list_mean_absolute_accuracy_change = []
        list_mean_label_flip_rate = []   


        for ablation_size in ablation_sizes:

            overall_global_absolute_accuracy_change = 0
            overall_global_tvd = 0
            overall_global_label_flip_rate = 0

            # # We ablate features from the same class segment for each class-specific features segments to maximize the accuracy variation (see Section A - Features segmentation strategy)
            # for c in range(nb_classes):

            #     segment = segmented_features[c]
                
            #     if len(segment) == 0:
            #         continue

            #     number_ablated_features = int(ablation_size * len(segment) / 100)
            #     # The features in each sublist of the list segmented_features are sorted in decreasing order based on their mean activations on the training dataset
            #     list_ablated_features = [int(feature_number) for feature_number in segment[:number_ablated_features]]
                
            #     variation_absolute_accuracy, tvd, label_flip_rate, _ = eval_causal_effect_concepts(
            #                                                                 ablation_features=list_ablated_features,
            #                                                                 class_int=c,
            #                                                                 decoder=decoder,
            #                                                                 model=model,
            #                                                                 interp_model=interp_model,
            #                                                                 activations_dataset=activations_dataset,
            #                                                                 selected_features=selected_features,
            #                                                                 method_name=cfg_concept.method_name,
            #                                                                 hook_layer=cfg_concept.hook_layer,
            #                                                                 hook_name=cfg_concept.hook_name,
            #                                                                 probs_pred=predicted_probabilities,
            #                                                                 accuracy=accuracy,
            #                                                                 device=device_interp_model,
            #                                                                 labels_tokens_id=labels_tokens_id,
            #                                                                 mixing_matrix=mixing_matrix,
            #                                                             )

            #     overall_global_absolute_accuracy_change += variation_absolute_accuracy
            #     overall_global_tvd += tvd
            #     overall_global_label_flip_rate += label_flip_rate
                
    


            # overall_global_absolute_accuracy_change/=nb_classes
            # overall_global_tvd/=nb_classes
            # overall_global_label_flip_rate/=nb_classes
            # print(f"Averaged Global Absolute Accuracy change over all the segments (for an ablation of {ablation_size} % on a segment) : {overall_global_absolute_accuracy_change}")
            # print(f"Averaged Global TVD over all the segments (for an ablation of {ablation_size} % on a segment) : {overall_global_tvd}")
            # print(f"Averaged Global Label-flip rate over all the segments (for an ablation of {ablation_size} % on a segment) : {overall_global_label_flip_rate}")
            # list_mean_absolute_accuracy_change.append(overall_global_absolute_accuracy_change)
            # list_mean_tvd.append(overall_global_tvd)
            # list_mean_label_flip_rate.append(overall_global_label_flip_rate)
        

        # Now we look more precisely at the percentage of flip predictions resulting in the ablation of a single feature when this one is normally activated
        # It means that we focus now on conditional causality. In that context, we ablate one feature at a time.
        overall_conditional_label_flip_rate = []
        overall_conditional_absolute_accuracy_change = []
        overall_conditional_tvd = []
        for feature in selected_features:

            # Retrieve the sentences where the feature was activated
            activating_samples = (dict_analysis_concepts['concepts activations'][:, feature].abs() > 1e-8).nonzero(as_tuple=True)[0]
            if activating_samples.shape[0]==0:
                continue
           
            
            probs_pred_ablation, accuracy_ablation , _ = get_predictions(
                                                            decoder=decoder,
                                                            model=model,
                                                            interp_model=interp_model,
                                                            activations_dataset=activations_dataset,
                                                            selected_features=selected_features,
                                                            method_name=cfg_concept.method_name,
                                                            hook_layer=cfg_concept.hook_layer,
                                                            hook_name=cfg_concept.hook_name,
                                                            perturb=[feature],
                                                            device=device_interp_model,
                                                            labels_tokens_id=labels_tokens_id,
                                                            ids_samples=activating_samples,
                                                            mixing_matrix=mixing_matrix,
                                                        )

            true_labels_samples = dict_analysis_concepts['true labels'][activating_samples]
            # Get the outputs for these sentences
            predicted_labels = torch.argmax(predicted_probabilities[activating_samples,:], dim=1)
            original_accuracy_on_selected_samples = ((true_labels_samples==predicted_labels).sum() / len(activating_samples)).item()

            predicted_labels_ablation = torch.argmax(probs_pred_ablation, dim=1)
            # Compute the conditional label flip rate (percentage of changes in label predicted on samples activating the investigated feature)
            conditional_label_flip_rate = float((predicted_labels != predicted_labels_ablation).float().mean())
            # Compute the conditional absolute accuracy variation
            conditional_absolute_accuracy_variation = (accuracy_ablation - original_accuracy_on_selected_samples)
            # Compute the conditional TVD
            conditional_tvd = 0.5 * torch.sum(torch.abs(predicted_probabilities[activating_samples,:] - probs_pred_ablation), dim=1).mean().item()
            
            
            overall_conditional_label_flip_rate.append(conditional_label_flip_rate)
            overall_conditional_tvd.append(conditional_tvd)
            overall_conditional_absolute_accuracy_change.append(conditional_absolute_accuracy_variation)

        print(f"Averaged conditional label-flip rate : {np.mean(overall_conditional_label_flip_rate)}")
        print(f"Averaged Conditional TVD : {np.mean(overall_conditional_tvd)}")
        print(f"Averaged Conditional Absolute Accuracy change : {np.mean(overall_conditional_absolute_accuracy_change)}")

        # Save all causality metrics
        dict_averaged_results_ablation = { 'Proportion of ablation per segment' : ablation_sizes, 
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
        
        
        print("######################################## END : Computation of Causality Metrics on z_class Concepts ########################################\n")
