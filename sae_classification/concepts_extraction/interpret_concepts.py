import re
from sae_implementation import TrainingSAE
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os
import gc
import csv
import torch
from sklearn.mixture import GaussianMixture
import numpy as np
import joblib
from loguru import logger
from ..utils import BaselineMethodConfig, LLMLoadConfig, EvaluationConceptsConfig
from .evaluation_utils import (
    ActivationDataset, 
    cache_activations_with_labels, 
    get_sae_path,
    is_empty_dir,
    load_all_shards,
)
from .baseline_method import ConceptNet
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ..llm_classifier_tuning import process_dataset,get_hook_model
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
import json
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from collections import Counter


for pkg in [
    "wordnet",               # for WordNetLemmatizer
    "omw-1.4",               # the WordNet “Open Multilingual Wordnet” data
    "stopwords",             # for nltk.corpus.stopwords
]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        print(f"can't find corpora/{pkg}")
        nltk.download(pkg)

# compile once at module load
_WORD_RE  = re.compile(r"\b[a-zA-Z]+\b")
_NLTK_STOPS = set(stopwords.words("english"))
_MIN_LEN = 3
_LEMMATIZER = WordNetLemmatizer()

# Get features activations from z_class for the SAE
def get_sae_activations(sae,activations_dataset):

    feature_activations_list = []
    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    sae.eval()
    with torch.no_grad():
        
        for batch in tqdm(activations_dataloader, desc="Get the SAE activations", unit="batch"):
            cache = batch["cache"].to(sae.cfg.device)
            cache_sentence = cache.unsqueeze(1)
    
            # Use the SAE
            feature_acts, acts_without_process = sae.encode_with_hidden_pre(cache_sentence)
            
            feature_activations_list.append(feature_acts.cpu())
        
        feature_activations_tensor = torch.cat(feature_activations_list,dim=0)

    return feature_activations_tensor

def compute_mean_cosine_similarity(sentence_embeddings):

    # Compute cosine similarity matrix
    sentence_embeddings_norm = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    cosine_similarity_matrix = torch.mm(sentence_embeddings_norm, sentence_embeddings_norm.T)

    k = cosine_similarity_matrix.shape[0]

    triu_indices = torch.triu_indices(k, k, offset=1).to(sentence_embeddings.device)  # Indices of upper triangular elements (excluding diagonal)
    mean_cos_sim = cosine_similarity_matrix[triu_indices[0], triu_indices[1]].mean() 
    return mean_cos_sim.item()

# Weighted Average on the number of pairs of sentences per set of concept activating cluster. Each ConceptSim(j) is averaged in ConceptSim with these weights
def compute_weighted_avg_concept_sim(concept_sim_list,activating_sentences_size):

    """
    set_scores : 1-D array_like
        Mean pairwise cosine similarity of each set  (C_i).
    set_sizes  : 1-D array_like, same length
        Size of each set  (n_i).

    Return:
    
      float:  Pair-weighted (“micro”) global coherence.
    """
    scores = np.asarray(concept_sim_list, dtype=float)
    sizes  = np.asarray(activating_sentences_size,  dtype=int)

    # number of pairs in each set: w_i = n_i choose 2
    pair_counts = sizes * (sizes - 1) // 2

    # ignore empty or singleton sets (pair_count == 0)
    valid = pair_counts > 0
    if not np.any(valid):
        raise ValueError("No sets with at least two elements.")

    weighted_sum = np.sum(pair_counts[valid] * scores[valid])
    weight_total = np.sum(pair_counts[valid])

    return weighted_sum / weight_total


def calc_topic_coherence(topic_words, docs, dictionary, calc4each=False):
    """
    Computes multiple topic coherence scores (C_V, C_UCI, C_NPMI) for a given topic model.
    
    topic_words: List of topics (each topic is a list of words)
    docs: List of tokenized documents
    dictionary: Gensim dictionary built from the dataset
    calc4each: Whether to return coherence scores for each topic separately
    Return: Tuple containing overall coherence scores and per-topic coherence scores (if requested)
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    aligned_topic_words = [
        [w for w in words if w in dictionary.token2id]
        for words in topic_words
    ]

    aligned_docs = [
        [token for token in doc if token in dictionary.token2id]
        for doc in docs
    ]

    # Compute C_UCI (Pointwise Mutual Information - PMI)
    c_uci_model = CoherenceModel(topics=aligned_topic_words, texts=aligned_docs, dictionary=dictionary, coherence='c_uci',topn=10,processes=1)
    c_uci_score = c_uci_model.get_coherence()    
    print(f"c_uci_score : {c_uci_score}")
    c_uci_per_topic = c_uci_model.get_coherence_per_topic() if calc4each else None

    # Compute C_NPMI (Normalized PMI)
    c_npmi_model = CoherenceModel(topics=aligned_topic_words, texts=aligned_docs, dictionary=dictionary, coherence='c_npmi',topn=10,processes=1)
    c_npmi_score = c_npmi_model.get_coherence()
    print(f"c_npmi_score : {c_npmi_score}")
    c_npmi_per_topic = c_npmi_model.get_coherence_per_topic() if calc4each else None

    # Compute C_V (Word Co-occurrence Sliding Window)
    cv_model = CoherenceModel(topics=aligned_topic_words, texts=aligned_docs, dictionary=dictionary, coherence='c_v',topn=10,processes=1)
    cv_score = cv_model.get_coherence()
    print(f"cv_score : {cv_score}")
    cv_per_topic = cv_model.get_coherence_per_topic() if calc4each else None
    

    if calc4each:
        return (cv_score, c_uci_score, c_npmi_score), (cv_per_topic, c_uci_per_topic, c_npmi_per_topic)
    
    return (cv_score, c_uci_score, c_npmi_score)
        



def filter_sentences(sentences : list):

    """
    Tokenizes, lowercases, drops NLTK stops, filters by length,
    and lemmatizes each word.
    """
    words = []
    for sentence in sentences:
        for token in _WORD_RE.findall(sentence.lower()):
            # stop‐word & length filters
            if token in _NLTK_STOPS:
                continue
            if not (_MIN_LEN <= len(token)):
                continue

            # lemmatize
            lemma = _LEMMATIZER.lemmatize(token)
            words.append(lemma)
    return words
    
'''
Save in a .csv the specified sentences as well with:
    - the activation value of the inspected feature on that sentence
    - the index of the feature in z_class
    - the activation rate of the feature 
    - the category this feature is associated to in our segmentation strategy

'''
def save_sentences_csv(saved_sentences,saved_activation_values,feature,category_feature,activation_rate,dir_interpretability_results):

    csv_file_dir = os.path.join(dir_interpretability_results,f'activating_sentences')
    if not os.path.exists(csv_file_dir):
            os.makedirs(csv_file_dir)

    csv_file_path = os.path.join(csv_file_dir,f'category_{category_feature}.csv')

    file_exists = os.path.isfile(csv_file_path)         
    file_not_empty = file_exists and os.path.getsize(csv_file_path) > 0

    mode = "a" if file_not_empty else "w"

    with open(csv_file_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
       
        # Write the header row
        if mode=="w":
            writer.writerow(['sentence','activation strength', 'feature_id', 'sparsity'])
        
        # Write the data rows
        for sentence, activation_value in zip(saved_sentences, saved_activation_values):
            writer.writerow([sentence,activation_value, feature, activation_rate])

'''
Save in a .txt the specified sentences as well with:
    - the activation value of the inspected feature on that sentence
Each class has its own .txt file. Top activating sentences per feature are saved in the .txt file of the class the feature is the most aligned with according our features segmentation scheme.
'''
def save_sentences_txt(saved_sentences,saved_activation_values,feature,category_feature,activation_rate,dir_interpretability_results):

    txt_file_dir = os.path.join(dir_interpretability_results,f'activating_sentences')
    if not os.path.exists(txt_file_dir):
            os.makedirs(txt_file_dir)

    txt_file_path = os.path.join(txt_file_dir,f'category_{category_feature}.txt')

    file_exists = os.path.isfile(txt_file_path)         
    file_not_empty = file_exists and os.path.getsize(txt_file_path) > 0

    mode = "a" if file_not_empty else "w"

    with open(txt_file_path, mode) as file:
        file.write(f'\n####### Feature {feature} ##########\n\n')
        file.write("\n")

        for text, value in zip(saved_sentences, saved_activation_values):
                    # Write each text and value on a new line
                    file.write(f"\n{text}\t{value}\n")
       
# Main function to compute interpretabiliy metrics for each extracted concept present in z_class. In particular, the reported ConceptSim and SentenceSim presented in the paper
def concept_interpretability(config_analysis: str, config_model : str):
 
    # Force Hugging Face & SentenceTransformer to load from cache
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    cfg_concept = EvaluationConceptsConfig.autoconfig(config_analysis)
    
    supported_approaches = ["ica","concept_shap","sae"]
    assert cfg_concept.method_name in supported_approaches, f"Error: The method {cfg_concept.method_name} is not supported in analysis. Currently the only supported methods are {supported_approaches}"

    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Dataset used for interpretability
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer)

    # Get model hooked (HookedTransformer). Needed to compute the new classifier outputs with the reconstructed hidden states.
    hook_model = get_hook_model(cfg_model,tokenizer)

    # We retrieve the length of the template added at the end of the sentence
    len_template = cfg_model.len_template
    
    # Check if the acivations and labels have already been cached
    dir_dataset_activations = os.path.join(cfg_concept.dir_dataset_activations,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    # Get the labels number
    labels_dataset = dataset_tokenized["true_label"]
    n_sentences = len(labels_dataset)
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

    # Remove the template part
    for t,_ in enumerate(original_text_used):
        original_text_used[t] = original_text_used[t][:-len_template] 


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the sentence encoder and encoding of the evaluated sentences
    logger.info(f"Loading of the sentence encoder and encoding of the evaluated sentences to use the cosine similarity metric")
    sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2',device=device,local_files_only=True)
    # Generate embeddings
    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(original_text_used), batch_size)):
        batch_texts = original_text_used[i : i + batch_size]
        # Encode on GPU
        embeddings_batch = sentence_embedding_model.encode(
            batch_texts,
            convert_to_tensor=True,
            device=device
        )
        all_embeddings.append(embeddings_batch)
    
    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    del all_embeddings
    gc.collect()

  
    mean_cos_sim = compute_mean_cosine_similarity(sentence_embeddings)
    print(f"Mean Pairwise Cosine Similarity between all the embedded sentences of the dataset : {mean_cos_sim}")

    sentence_embeddings = sentence_embeddings.cpu()

    # Tokenization: Convert Sentences to Lists of Words
    list_split_words = [ filter_sentences([sentence]) for sentence in  original_text_used]

    
    # Build Dictionary 
    dictionary = Dictionary(list_split_words)
    dictionary.filter_extremes(no_below=5, no_above=0.8)  # Removes rare and common words
    dictionary.compactify()   

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

        # Activations of the concepts for the tested embedded sentences
        with torch.no_grad():
            concepts_activations = interp_model.concepts_activations(embeddings_sentences_tensor)

        # Name of the specifc concept-based method used
        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_concept.hook_layer}'
    
    elif cfg_concept.method_name=='ica':

        method_name = cfg_concept.method_name

        ica_path = os.path.join(f'{cfg_concept.path_to_baseline_methods}',f'ica.pkl')

        try : 
            interp_model = joblib.load(ica_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: '{ica_path}' not found. Ensure you have fitted the corresponding ICA before evaluation of its concepts with 'train-baseline'")

        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"]
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0) #(dataset size, embedding_dim)
        embeddings_sentences_numpy = embeddings_sentences_tensor.cpu().numpy()
        concepts_activations = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)) #shape : [batch size, num_components]

        # Name of the specifc concept-based method used
        concept_model_name=f'ica_{interp_model.components_.shape[0]}_layer_{cfg_concept.hook_layer}'


    elif cfg_concept.method_name=='sae':

        # For the SAE, we implement additional post-processing steps for the selection of features since the hidden layer dimension may be larger than the number of desired concepts
        assert  cfg_concept.methods_args['features_selection'] in ['truncation', 'logistic_regression'], f"Error: For the post-selection of the SAE features, the options supported in methods_args['features_selection'] are either 'truncation' or 'logistic_regression' "
    
        if cfg_concept.methods_args['features_selection'] == 'truncation':
            method_name = "sae_truncation"
        elif cfg_concept.methods_args['features_selection'] == 'logistic_regression':
            method_name = "sae_logistic_regression"

        # Retrieve the trained sae path
        sae_path, concept_model_name = get_sae_path(cfg_concept)

        # Load the local trained SAE
        interp_model = TrainingSAE.load_from_pretrained(sae_path,device)
        
        # Retrieve the SAE activations
        concepts_activations = get_sae_activations(interp_model,activations_dataset)
        concepts_activations = concepts_activations.squeeze(1)    

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

    # Directory where to save the interpretability metrics
    dir_to_save_metrics = os.path.join(concepts_dir_path,cfg_model.split)
    dir_interpretability_results = os.path.join(dir_to_save_metrics,"interpretability")
    if not os.path.exists(dir_interpretability_results):
        os.makedirs(dir_interpretability_results)

    interpretability_list = []
    activating_clusters_sizes_list = []
    activating_sentences_list = []

    # Cluster the 'activating' sentences for each feature in z_class
    for feature in selected_features:

        # Vector of activations of the feature across the evaluated dataset of sentences 
        activations_inspected = concepts_activations[:,feature]
        activations_inspected  = activations_inspected.cpu().numpy().reshape(-1,1)

        # 1d clustering on the sentences
        gmm = GaussianMixture(
            n_components=2,          # “inactive” vs “active”
            covariance_type="full",  # allow unequal variance
            n_init=10,               # try 10 random starts
            random_state=0
        ).fit(activations_inspected)

        # Soft assignment: posterior P(component | sample)
        proba = gmm.predict_proba(activations_inspected)        # shape (n_samples, 2)
        inactive_prob = proba[:, gmm.means_.argmin()]  # smaller mean = inactive
        active_prob   = 1 - inactive_prob

        # A scalar threshold for transparency 
        inactive_mean, active_mean = np.sort(gmm.means_.ravel())
        # threshold = 0.5 * (inactive_mean + active_mean)
        
        # Hard assignment as we need a Boolean mask
        activated = active_prob >= 0.5 

        # Save the indices of the sentences present in the concept 'activating' clsuster
        activating_cluster_sentences_indices = np.where(activated == 1)[0]
        activating_sentences_list.append(activating_cluster_sentences_indices.tolist())

        # Discard the computation of ConceptSim(j) if there is only one sentence in the activating cluster
        contains_0_or_1 = np.sum(activated) <= 1
        if contains_0_or_1:
            continue

        activating_cluster_sentences_embeddings = sentence_embeddings[torch.from_numpy(activated)]
        # print(f"activating_cluster_sentences_embeddings shape : {activating_cluster_sentences_embeddings.shape}")
        mean_cos_sim_concept = compute_mean_cosine_similarity(activating_cluster_sentences_embeddings)
        activating_cluster_size = np.sum(activated)
        # print(f"mean_cos_sim_concept : {mean_cos_sim_concept}")

        # Add the ConceptSim(j) metric of the current concept j
        interpretability_list.append(mean_cos_sim_concept)
        activating_clusters_sizes_list.append(activating_cluster_size)


    iter_concept = 0
    concepts_words = []
    concepts_sentences_ids = []
    # Save the top activating sentences for each feature in z_class (in a .txt and .csv file). Compute as well coherence scores for each concept based on the 'activating' sentences segment for the feature.
    for index_feature,feature in enumerate(selected_features):
        iter_concept+=1
        activations_inspected = concepts_activations[:,feature]

      
        activating_texts = [original_text_used[i] for i in activating_sentences_list[index_feature]]
        # print(f"len(activating_texts) : {len(activating_texts)}")
        	

	    # Retrieve the important words present in the sentences within the 'activating' cluster of the feature  
        analyzed_text = filter_sentences(activating_texts)
        concepts_words.append(analyzed_text)

        # Display the top 25 most frequent words in the 'activating' cluster of the feature (without consideration for stop words)
        cx = Counter(analyzed_text)
        most_occur = cx.most_common(25)
        print("Concept " + str(iter_concept) + " most common words:")
        print(most_occur)
        print("\n")

	    # Save the top 10 sentences within the 'activating' cluster associated with the highest activations of the concept
        _, top10_sentences = activations_inspected.topk(10)
        # We keep them if they are in the activating cluster
        saved_sentences_indices = list(set(top10_sentences.tolist()).intersection(activating_sentences_list[index_feature]))  
        # Activation strength of the feature on the sentence
        saved_sentences = [original_text_used[i] for i in saved_sentences_indices]
        saved_activation_values = activations_inspected[saved_sentences_indices].tolist()


        # Get the category the feature was segmented to
        category_feature = next((i for i, segment in enumerate(segmented_features) if feature in segment), None)
        print(f"Category {category_feature} for feature {feature}")

        # Sparsity level of the feature 
        activation_rate = len(activating_texts) / (activations_inspected.shape[0])

        # Save the sentences in the corresponding .csv for visualization after
        save_sentences_csv(saved_sentences,saved_activation_values,feature,category_feature,activation_rate,dir_interpretability_results)
        # Do the same in a .txt file
        save_sentences_txt(saved_sentences,saved_activation_values,feature,category_feature,activation_rate,dir_interpretability_results)
        

    # Compute Coherence metrics 
    (cv_score, c_uci_score, c_npmi_score) = calc_topic_coherence(concepts_words, list_split_words, dictionary, False)
    print(f"C_V: {cv_score:.4f}, C_UCI: {c_uci_score:.4f}, C_NPMI: {c_npmi_score:.4f}")

    print(f"activating_clusters_sizes_list : {activating_clusters_sizes_list}")
    weighted_avg_concept_sim = compute_weighted_avg_concept_sim(interpretability_list,activating_clusters_sizes_list)
    unweighted_avg_concept_sim = np.mean(interpretability_list)

    print(f"Unweighted Averaged Concept Cosine Similarity across all concepts : {unweighted_avg_concept_sim}")
    print(f"Weighted Averaged Concept Cosine Similarity across all concepts : {weighted_avg_concept_sim}")



    #############  Jaccard, Redundancy, Coverage ###############
    def jaccard_similarity(set1, set2):
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

    sets = [set(lst) for lst in activating_sentences_list]
    
    # Compute average pairwise Jaccard similarity
    jaccard_scores = [
        jaccard_similarity(sets[i], sets[j])
        for i, j in combinations(range(len(sets)), 2)
    ]
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
    
    # Compute redundancy
    all_elements = [x for sublist in activating_sentences_list for x in sublist]
    element_counts = Counter(all_elements)
    redundancy = sum(element_counts.values()) / len(element_counts)

    #Coverage
    coverage = len(element_counts) / len(original_text_used)
    
    print(f"avg_jaccard : {avg_jaccard}")
    print(f"redundancy : {redundancy}")
    print(f"coverage : {coverage}")
    
    ############ Compute SentenceSim(k) for k ranging from 1 to 4 ###############
    
    # Get only the top 5 most activated concepts per sentence (in case this is not already set up)
    projected_activations = concepts_activations[:,selected_features].cpu()
    topk_values, topk_indices = torch.topk(projected_activations, k=5, dim=1)
    masked_projections = torch.zeros_like(projected_activations)
    masked_projections.scatter_(1, topk_indices, topk_values)

    # Get a binary vision of the concept activations
    activation_mask = masked_projections != 0  
    # Get the number of shared concepts between each pair of sentence
    shared_counts = torch.matmul(activation_mask.float(), activation_mask.t().float())

    indices_neighbor_sentences = {}
    # For each sentence
    for i,sentence_embed in enumerate(sentence_embeddings):
        indices_neighbor_sentences[i] = {}
        # Get all sentences which share exactly k concepts with the sentence i  
        for k in range(4, 0, -1):  # from 4 down to 1
            indices = (shared_counts[i] == k).nonzero(as_tuple=True)[0]
            indices = indices[indices != i]  # remove self-match
            if indices.numel() > 0:
                indices_neighbor_sentences[i][k] = indices.tolist()

    cos_sim_by_number_share_concepts = [[] for _ in range(4)]
    # For each sentence
    for i, common_dict in indices_neighbor_sentences.items():
        sentence_embed = sentence_embeddings[i,:]
        # For each number of shared concepts
        for k, idxs in common_dict.items():
            neighbors_embeddings = sentence_embeddings[idxs,:]
            # Compute averaged cosine similarity with the sentences sharing exactly k concepts
            cos_sim = F.cosine_similarity(neighbors_embeddings, sentence_embed.unsqueeze(0), dim=1)
            mean_cos_sim_neighbors = cos_sim.mean()
            cos_sim_by_number_share_concepts[k-1].append(mean_cos_sim_neighbors.item())

    # Compute SentenceSim(k) for k from 1 to 4
    mean_cos_sim_by_number_share_concepts = [ sum(cos_sim_pairwise_for_k)/len(cos_sim_pairwise_for_k) if cos_sim_pairwise_for_k!=[] else 0 for cos_sim_pairwise_for_k in  cos_sim_by_number_share_concepts ] 
   
    print(f"SentenceSim(k) for k from 1 to 4 : {mean_cos_sim_by_number_share_concepts}")
    
    dict_interpretability_metrics = {"avg_jaccard" : avg_jaccard, "redundancy" : redundancy, "coverage" : coverage, "cv_score" : cv_score, "c_uci_score" : c_uci_score, "c_npmi_score" : c_npmi_score, "Averaged mean cosine similarity over the dataset" : mean_cos_sim ,"Weighted Averaged ConceptSim" : weighted_avg_concept_sim, "Unweighted Averaged ConceptSim" : unweighted_avg_concept_sim,'SentenceSim(k) for k from 1 to 4' : mean_cos_sim_by_number_share_concepts}


    file_to_save_intepretability_metrics = os.path.join(dir_interpretability_results,f"interpretability_metrics.json") 
    with open(file_to_save_intepretability_metrics, 'w') as file:
        json.dump(dict_interpretability_metrics, file, indent=4)

        
            
        
        
