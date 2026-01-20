from .evaluation_utils import ( 
    is_empty_dir,
    load_all_shards,
    load_interp_model,
    create_activations_dataset,
    collate_single_sample
)
from ..utils import EvaluationConceptsConfig, LLMLoadConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from ..llm_classifier_tuning import get_hook_model, get_model, _init_tokenizer, _max_seq_length
from .caching import max_seq_length_hook
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
import os
import gc
import numpy as np
import torch
import json
from typing import List
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger



# Features segmentation strategy operated on z_class
def segment_features(
    mean_activations: torch.Tensor,
    feature_activations_tensor: torch.Tensor,
    labels_tensor: torch.Tensor,
    unique_labels: np.ndarray,
    nb_classes: int,
) -> List[List[int]]:
    """
    Assign each feature to the class where it is comparatively most active.
    - Uses per-class mean activations (class-balanced).
    - Normalizes across classes per feature.
    - Returns a list of length nb_classes with sorted feature indices per class.
    - It implements the Features segmentation strategy explained in Section A.
    
    """

    device = feature_activations_tensor.device
    dtype = feature_activations_tensor.dtype
    n_features = feature_activations_tensor.shape[1]

    # Per-class mean feature activations (class-balanced)
    feature_score_class_array = torch.zeros(
        (len(unique_labels), n_features), device=device, dtype=dtype
    )
    for i, label in enumerate(unique_labels):
        idx = torch.nonzero(labels_tensor == label, as_tuple=False).squeeze(-1)
        if idx.numel() > 0:
            feature_score_class_array[i] = feature_activations_tensor[idx].mean(dim=0)
        else:
            # Empty class : zeros (no contribution before normalization)
            feature_score_class_array[i] = torch.zeros(n_features, device=device, dtype=dtype)

    # Normalize across classes per feature (column-wise)
    feature_sums = feature_score_class_array.sum(dim=0)
    normalized_class_scores = torch.zeros_like(feature_score_class_array)
    nonzero = feature_sums != 0
    normalized_class_scores[:, nonzero] = feature_score_class_array[:, nonzero] / feature_sums[nonzero]
    # Dead features (all-zero across classes) : uniform
    if (~nonzero).any():
        normalized_class_scores[:, ~nonzero] = 1.0 / nb_classes

    # Assign each feature to its top class
    top_class = torch.argmax(normalized_class_scores, dim=0)

    # Build as many segments as there are classes (row order corresponds to `unique_labels`)
    segmented_features: List[List[int]] = [[] for _ in range(nb_classes)]
    for feat_idx, class_idx in enumerate(top_class):
        row = class_idx.item()
        if row < nb_classes:
            segmented_features[row].append(feat_idx)

    # Sort features in each segment by mean activity (descending)
    # mean_activations is assumed to be length n_features
    for i, segment in enumerate(segmented_features):
        segmented_features[i] = sorted(segment, key=lambda j: mean_activations[j].item(), reverse=True)

    return segmented_features


# Train a logistic regression to automatically select the expected required number of concepts from z_sae to generate z_class
# This function is used for the canonical SAE baseline in post-processing after the training to select the most relevant concepts for the classification task
def select_top_k_sae_features(
    X: torch.Tensor,
    y: torch.Tensor,
    k: int,
    Cs=(0.01, 0.1, 1.0),
    random_state: int | None = 42,
):
    """
    Rank SAE features via multinomial logistic regression coefficients and return indices of the top-k features.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if k > X.shape[1]:
        raise ValueError("k cannot exceed the number of SAE features")

    X_np = X.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    max_iter = 1000
    penalty="l1"

    logger.info(f"Start of the logistic regression training with the following parameters :\n penalty : {penalty} \n Cs : {Cs} \n solver : saga \n max_iter : {max_iter}")


    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty,
            solver="saga",
            multi_class="multinomial",
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            class_weight="balanced",
        ),
    )

    param_grid = {"logisticregression__C": Cs}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    clf = GridSearchCV(base, param_grid, scoring="accuracy", cv=cv, n_jobs=-1)
    clf.fit(X_np, y_np)

    logger.info(f"Selection of the top {k} SAE features for the classification task based on the logistic regression coefficients")

    # Score of each SAE feature as their coefficient euclidean norm in the logistic regression 
    # Coeffs from best estimator (shape: [n_classes, n_features])
    coefs = clf.best_estimator_.named_steps["logisticregression"].coef_
    scores = np.linalg.norm(coefs, axis=0)
    topk_idx = np.argsort(scores, kind="stable")[::-1][:k]

     # Accuracy of the logistic regression on the reduced set of SAE features for information purpose only
    logger.info(f"Accuracy of the logistic regression with the reduced set of the {k} selected SAE features for information purpose")


    best_C = float(clf.best_params_["logisticregression__C"])
    reduced = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty=penalty,
            solver="saga",
            multi_class="multinomial",
            C=best_C,
            max_iter=max_iter,
            tol=1e-3,
            random_state=random_state,
            class_weight="balanced",
        ),
    )
    reduced.fit(X_np[:, topk_idx], y_np)
    acc_k = reduced.score(X_np[:, topk_idx], y_np)  # optimistic; for info only

    logger.info(f"Accuracy of the logistic regression on the classification task with the selected reduced set of SAE features : {acc_k}")


    return torch.as_tensor(topk_idx.copy(), dtype=torch.long)

           

# Forward pass on the LLM classifier through the concepts-layer bottleneck.
# Once the concepts activations are retrieved, it encodes the features segmentation strategy and post-selection for z_class 
def post_process_features(
    interp_model,
    activations_dataset,
    unique_labels,
    nb_classes,
    cfg_concept,
    device):

    """
        Forward pass to get concept activations, segment features, and apply optional SAE post-selection.
        Returns:
            segmented_features: List[List[int]]  # per-class feature indices (after selection/filtering)
            selected_features:  List[int]        # flat list of kept feature indices
    """

    activations_dataloader = DataLoader(activations_dataset,batch_size=1,collate_fn=collate_single_sample)

    feature_activations_list = []
    prompt_labels_list = []

    with torch.no_grad():

        for batch in tqdm(activations_dataloader, desc="Forward Passes on the classifier LLM model", unit="batch"):

            cache = batch["cache"].to(device)
            labels = batch["label"]


            if cfg_concept.method_name=='concept_shap':

                #concept_score_thres_prob
                feature_acts, _, _ = interp_model.concepts_activations(cache)

            elif cfg_concept.method_name=='hi_concept':

                #concept_score_thres_prob
                feature_acts, _, _, _,_ = interp_model.concepts_activations(cache)

            elif cfg_concept.method_name=='sae':

                # SAE expects an extra dim
                cache_sentence = cache.unsqueeze(1)
                feature_acts, _ = interp_model.encode_with_hidden_pre(cache_sentence)
                feature_acts = feature_acts.squeeze(1)

            elif cfg_concept.method_name=='ica':
                
                embeddings_sentences_numpy = cache.detach().cpu().numpy()
                #ica_activations
                feature_acts = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy))
            
            else:
                raise ValueError(f"Unknown concept method: {cfg_concept.method_name}")

            feature_activations_list.append(feature_acts)
            prompt_labels_list.append(labels)

        del activations_dataloader
        del activations_dataset
        gc.collect()

        feature_activations_tensor = torch.cat(feature_activations_list,dim=0).cpu()
        labels_tensor = torch.cat(prompt_labels_list)
        
        # Initial selection: all features
        selected_features = torch.arange(feature_activations_tensor.shape[1])

        # Mean absolute activation per feature  across the sentences dataset. Absolute activation to take into account the ICA method which does not include a positive threshold.
        mean_activations = feature_activations_tensor.abs().mean(dim=0)
        print(f"mean_activations : {mean_activations}")
        
        # Segment features 
        segmented_features = segment_features(mean_activations,
                                              feature_activations_tensor,
                                              labels_tensor,
                                              unique_labels,
                                              nb_classes,
        )
        
        # Additional post-selection if the method used is based on SAE
        if cfg_concept.method_name=='sae':

            n_concepts = cfg_concept.methods_args['n_concepts']
            concepts_selection_mode = cfg_concept.methods_args['features_selection']

            if concepts_selection_mode == 'truncation':
                
                logger.info(
                    f"SAE post-selection = truncation: keep the first {n_concepts} features in the hidden layer i.e. z_class = z[{n_concepts}:]"
                )                
                selected_features = torch.arange(n_concepts)
            
            elif concepts_selection_mode == 'logistic_regression':
                logger.info("SAE post-selection = logistic_regression")
                selected_features = select_top_k_sae_features(
                                    X = feature_activations_tensor,
                                    y = labels_tensor,
                                    k = n_concepts,
                                    random_state=42)
            
            else:
                raise ValueError(f"Unknown SAE features_selection mode: {concepts_selection_mode}")

        
        # Filter dead features with threshold
        dead_threshold = 1e-6
        selected_features = [j.item() for j in selected_features if mean_activations[j].item() > dead_threshold]

        # Retain only selected features in each segment
        segmented_features = [[x for x in sublist if x in selected_features] for sublist in segmented_features]

        return segmented_features, selected_features
            
# Main function for the selection of indices within z_sae that are assigned to z_class and for the features segmentation strategy
def selection_segmentation_concepts(
    config_concept: str,
    config_model: str
    ):

    """
        1) Ensure/test-split activations are cached.
        2) Load interpretation model for the chosen concept method.
        3) Post-process features : segmentation (per class) + selection of z_class for SAE.
        4) Save results to disk.

        Side effects:
        - Creates activation cache under:
            {cfg_concept.dir_dataset_activations}/{split}/{model_name}/{dataset}/layer_{hook_layer}
        - Saves segmented & selected features under:
            {cfg_concept.post_processed_features_path}/{model_name}/{dataset}/{method}/{concept_model_name}/{n_concepts}_concepts
    """

    print("\n######################################## BEGIN : Selection of z_class and Features Segmentation Strategy ########################################")

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


    # Check if  already cached a dataset of the activations, labels, attention mask....
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
    
    # # Original text—only if you actually use it
    # orig_path = os.path.join(cache_activations_dir,f'original_text.json')
    # if orig_path.exists():
    #     try:
    #         with orig_path.open("r") as f:
    #             original_text_used = json.load(f)
    #         logger.debug(f"Loaded {len(original_text_used)} original texts from cache.")
    #     except Exception as e:
    #         logger.warning(f"Could not read original_text.json: {e}")
    

    # Load interpretation model
    logger.info(f"Loading model for concept-based method={cfg_concept.method_name}")
    interp_model, _, _, method_name, concept_model_name, device_interp_model = load_interp_model(
                                                                                                    cfg_concept=cfg_concept,
                                                                                                    activations_dataset=activations_dataset,
                                                                                                    decoder=decoder,
                                                                                                    device=device
                                                                                                )

    
    '''
        'segmented_features' is a list of list where list[j] contains the features the most associated to class j. 
        'segmented_features' segments only the features from z_class.
        'selected_features' contains a subset of features to match the number of imposed concepts. It contains the indices of features in z_sae that are assigned to the subvector z_class
    '''
    segmented_features, selected_features = post_process_features(
        interp_model,
        activations_dataset,
        unique_labels,
        nb_classes,
        cfg_concept,
        device_interp_model
    )

    # Save the results
    concepts_dir_path = os.path.join(cfg_concept.post_processed_features_path,cfg_model.model_name,cfg_model.dataset_name,method_name,concept_model_name,f"{cfg_concept.methods_args['n_concepts']}_concepts")
    os.makedirs(concepts_dir_path, exist_ok=True)
   
    
    segmented_features_file_path = os.path.join(concepts_dir_path,f"segmented_features.json")
    selected_features_file_path = os.path.join(concepts_dir_path,f"selected_features.json")

    seg_sizes = [len(sub) for sub in segmented_features]
    
    with open(segmented_features_file_path, "w") as f:
        json.dump(segmented_features, f)
    with open(selected_features_file_path, "w") as f:
        json.dump(selected_features, f)

    logger.info("Post-processing (segmentation + selection) of the concepts in z_class complete.")
    logger.info(f"The segmented concepts indices are {segmented_features}, they are saved at {concepts_dir_path}")
    logger.info(f"Segments per class: {seg_sizes} | Selected features: {len(selected_features)}")

    print("######################################## END : Selection of z_class and Features Segmentation Strategy ########################################\n")