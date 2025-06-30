from .evaluation_utils import ( 
    cache_activations_with_labels, 
    ActivationDataset, 
    get_sae_path,
    is_empty_dir,
    load_all_shards
)
from ..utils import EvaluationConceptsConfig, LLMLoadConfig
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ..llm_classifier_tuning import process_dataset,get_hook_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
import os
import gc
import numpy as np
import torch
import json
import joblib
from torch.utils.data import DataLoader
from .baseline_method import ConceptNet
from tqdm import tqdm
from loguru import logger
from sae_implementation import TrainingSAE



# Features segmentation strategy operated on z_class
def segment_features(
    mean_activations,
    feature_activations_tensor,
    labels_tensor,
    unique_labels,
    nb_classes
):

    # Assign a score to each feature with regard to each class
    feature_score_class = {}
    for label in unique_labels:
        indices = torch.where(labels_tensor==label)[0]
        if indices.shape[0] > 0: #Is at least one sample is associated to this label
            feature_score_class[label] = torch.sum(feature_activations_tensor[indices,:],dim=0)

    feature_score_class_array = torch.zeros((nb_classes,feature_activations_tensor.shape[1]))
    #Concatenate features scores of each class for normalization
    for i,label in enumerate(unique_labels):
        feature_score_class_array[i,:] = feature_score_class[label]
    feature_sums = feature_score_class_array.sum(dim=0)
    indices_dead_features = (feature_sums==0.)
    normalized_class_scores = torch.zeros((nb_classes,len(feature_sums)))
    normalized_class_scores[:,~indices_dead_features] = feature_score_class_array[:,~indices_dead_features] / feature_sums[~indices_dead_features]
    normalized_class_scores[:,indices_dead_features] = (torch.ones(nb_classes) / nb_classes).reshape(-1,1) 

    top_indice = torch.argmax(normalized_class_scores,dim=0)

    segmented_features = [[] for _ in range(nb_classes)]
    # Populate each sub-list with the indices of the features the most associated to the class c
    for feature_number, top_class in enumerate(top_indice):
        segmented_features[top_class.item()].append(feature_number)

    #Sort the features in each segment in decreasing order of their mean activations
    for i,segment in enumerate(segmented_features):
        segmented_features[i] =  sorted(segment, key=lambda j: mean_activations[j], reverse=True)

    return segmented_features

# Train a logistic regression to automatically select the expected required number of concepts from z_sae to generate z_class
def select_top_k_sae_features(
    X : torch.Tensor,
    y : torch.Tensor,
    k: int,
    Cs=(0.01, 0.1, 1),
    random_state: int | None = 42,
):

    
    """
    Rank features by importance in a multinomial logistic regression
    and return the indices of the `k` strongest ones.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    if k > X.shape[1]:
        raise ValueError("k cannot exceed the number of SAE features")

    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    max_iter = 1000
    penalty="l1"

    logger.info(f"Start of the logistic regression training with the following parameters :\n penalty : {penalty} \n Cs : {Cs} \n solver : saga \n max_iter : {max_iter}")

    clf_cv = LogisticRegressionCV(
        penalty=penalty,
        solver="saga",          
        Cs=Cs,
        cv=3,
        max_iter=1000,
        n_jobs=-1,
        tol=1e-3,
        random_state=random_state,
        scoring="accuracy",
    )

    pipe = make_pipeline(StandardScaler(), clf_cv)
    pipe.fit(X_np, y_np)


    logger.info(f"Selection of the top {k} SAE features for the classification task based on the logistic regression scores")
    
    # Score of each SAE feature as their coefficient euclidean norm in the logitsic regression 
    scores = np.linalg.norm(pipe[-1].coef_, axis=0)  
    topk_idx = np.argsort(scores)[::-1][:k]

    # Accuracy of the logistic regression on the reduced set of SAE features for information purpose only
    logger.info(f"Accuracy of the logistic regression with the reduced set of the {k} selected SAE features for information purpose")

    pipe_reduced = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty, solver="saga", C=pipe[-1].C_[0],
            max_iter=1000, n_jobs=1, random_state=random_state
        )
    )
    pipe_reduced.fit(X_np[:, topk_idx],y_np)
    acc_k = pipe_reduced.score(X_np[:, topk_idx], y_np)

    logger.info(f"Accuracy of the logistic regression on the classification task with the selected reduced set of SAE features : {acc_k}")

    return torch.as_tensor(topk_idx.copy(), dtype=torch.long)             

# Forward pass on the LLM classifier through the elements-layer bottleneck.
# Once the concepts activations are retrieved, it encodes the features segmentation strategy and post-selection for z_class if the method used is based on SAE 
def post_process_features(
    interp_model,
    activations_dataset,
    unique_labels,
    nb_classes,
    cfg_concept,
    device):

    activations_dataloader = DataLoader(activations_dataset,batch_size=1)

    feature_activations_list = []
    prompt_labels_list = []

    with torch.no_grad():

        for batch in tqdm(activations_dataloader, desc="Forward Passes of the given LLM model", unit="batch"):

            cache = batch["cache"].to(device)
            labels = batch["label"]

            cache_sentence = cache.unsqueeze(1)

            if cfg_concept.method_name=='concept_shap':

                #concept_score_thres_prob
                feature_acts = interp_model.concepts_activations(cache)

            elif cfg_concept.method_name=='sae':

                # Use the SAE
                feature_acts, acts_without_process = interp_model.encode_with_hidden_pre(cache_sentence)
                feature_acts = feature_acts.squeeze(1)

            elif cfg_concept.method_name=='ica':
                embeddings_sentences_numpy = cache.cpu().numpy()
                #ica_activations
                feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy))

            feature_activations_list.append(feature_acts)
            prompt_labels_list.append(labels)

        del activations_dataloader
        del activations_dataset
        gc.collect()

        feature_activations_tensor = torch.cat(feature_activations_list,dim=0).cpu()
        labels_tensor = torch.cat(prompt_labels_list)
        selected_features = torch.arange(feature_activations_tensor.shape[1])

        # Compute the mean activation of each feature across the sentences dataset. We impose the absolute activation to take into account the ICA method which does not include a positive threshold.
        mean_activations = feature_activations_tensor.abs().mean(dim=0)
        # Segment features 
        segmented_features = segment_features(mean_activations,feature_activations_tensor,labels_tensor,unique_labels,nb_classes)

        # Additional post-selection if the method used is based on SAE
        if cfg_concept.method_name=='sae':

            if cfg_concept.methods_args['features_selection'] == 'truncation':
                
                logger.info("The post-processing feature selection method chosen for the SAE-based approach is the simple truncation of the first SAE features in the hidden layer")
                selected_features = torch.arange(cfg_concept.methods_args['n_concepts'])
            
            elif cfg_concept.methods_args['features_selection'] == 'logistic_regression':
                logger.info("The post-processing feature selection method chosen for the SAE-based approach is the logistic regression")
                selected_features = select_top_k_sae_features(
                                    X = feature_activations_tensor,
                                    y = labels_tensor,
                                    k = cfg_concept.methods_args['n_concepts'],
                                    random_state=0)
        
        # Filter dead features if needed
        selected_features = [x.item() for x in selected_features if ( (mean_activations[x]>1e-5) or (mean_activations[x]<-1e-5) )]

        # We keep in segmented_features, only the features present in selected_features
        segmented_features = [[x for x in sublist if x in selected_features] for sublist in segmented_features]

        return segmented_features, selected_features
            
# Main function for the selection of indices within z_sae that are assigned to z_class and for the features segmentation strategy
def selection_segmentation_concepts(
    config_concept: str,
    config_model: str
    ):

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

    # Process the dataset on which we will do the forward passes
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer) 
    
    # Check if we already cached a dataset of the acivations, labels, attention mask....
    dir_dataset_activations = os.path.join(cfg_concept.dir_dataset_activations,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    labels_dataset = dataset_tokenized["true_label"]
    unique_labels = np.unique(np.array(labels_dataset))
    nb_classes = len(unique_labels)

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

            # Get model hooked (HookedTransformer)
            hook_model = get_hook_model(cfg_model,tokenizer)

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

    if cfg_concept.method_name=='concept_shap':

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
        interp_model.load_state_dict(torch.load(conceptnet_weights_path,weights_only=True))

    elif cfg_concept.method_name=='sae':

        # Retrieve the trained sae path
        sae_path, sae_name = get_sae_path(cfg_concept)

        #Load the local trained SAE
        interp_model = TrainingSAE.load_from_pretrained(sae_path,cfg_concept.methods_args['device'])

        # For the SAE, we implement additional post-processing steps for the selection of features since the hidden layer dimension may be larger than the number of desired concepts
        assert  cfg_concept.methods_args['features_selection'] in ['truncation', 'logistic_regression'], f"Error: For the post-selection of the SAE features, the options supported in methods_args['features_selection'] are either 'truncation' or 'logistic_regression' "
    
    elif cfg_concept.method_name=='ica':

        ica_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}',f'ica.pkl')
        interp_model = joblib.load(ica_path)

    '''
        'segmented_features' is a list of list where list x contains the features the most associated to class x. 
        
        We may also want additional post-selection of features if we evaluate a SAE-based method. 
        'selected_features' contains a subset of features to match the number of imposed concepts. It contains the indices of features in z_sae that are assigned to the subvector z_class
        'segmented_features' segments only the features from z_class.

    '''
    segmented_features, selected_features = post_process_features(interp_model,activations_dataset,unique_labels,nb_classes,cfg_concept,device)

    # Decide which variant has been used if the concepts were computed with SAE
    if cfg_concept.method_name=="sae":
        if cfg_concept.methods_args['features_selection'] == 'truncation':
            method_name = "sae_truncation"
        elif cfg_concept.methods_args['features_selection'] == 'logistic_regression':
            method_name = "sae_logistic_regression"
    else:
        method_name = cfg_concept.method_name


    # Create the name of the specifc concept-based model used
    if cfg_concept.method_name=="sae":
        concept_model_name=sae_name
    elif cfg_concept.method_name=="concept_shap":
        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_concept.hook_layer}'
    elif cfg_concept.method_name=='ica':
        concept_model_name=f'ica_{interp_model.components_.shape[0]}_layer_{cfg_concept.hook_layer}'
    

    # Directory where to save the results
    concepts_dir_path = os.path.join(cfg_concept.post_processed_features_path,cfg_model.model_name,cfg_model.dataset_name,method_name,concept_model_name,f"{cfg_concept.methods_args['n_concepts']}_concepts")
    if not os.path.exists(concepts_dir_path):
            os.makedirs(concepts_dir_path)
    
    segmented_features_file_path = os.path.join(concepts_dir_path,f"segmented_features.json")
    selected_features_file_path = os.path.join(concepts_dir_path,f"selected_features.json")
    with open(segmented_features_file_path, "w") as f:
        json.dump(segmented_features, f)
    with open(selected_features_file_path, "w") as f:
        json.dump(selected_features, f)

    logger.info(f"Post-processing (segmentation and selection) of the concepts is terminated.")
    logger.info(f"The segmented concepts indices are saved at {concepts_dir_path}")