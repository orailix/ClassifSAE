import re
from sae_implementation import TrainingSAE
from collections import Counter
import nltk
from nltk.corpus import stopwords
import os
import torch
import random
from safetensors.torch import load_file
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from loguru import logger
from ..utils import DRMethodsConfig, LLMLoadConfig, EvaluationConfig
import pickle
from .evaluation_utils import compute_loss_last_token_classif, update_metrics, count_same_match, ActivationDataset, cache_activations_with_labels
from .concept_shap import ConceptNet
from datasets import Dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ..model_training import process_dataset,get_hook_model
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
import pickle
import time
import json
from functools import partial
import gensim
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys

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

    # print(f"cosine_similarity_matrix shape : {cosine_similarity_matrix.shape}")

    k = cosine_similarity_matrix.shape[0]
    triu_indices = torch.triu_indices(k, k, offset=1)  # Indices of upper triangular elements (excluding diagonal)
    mean_cos_sim = cosine_similarity_matrix[triu_indices[0], triu_indices[1]].mean() 
    return mean_cos_sim.item()
    



def calc_topic_coherence(topic_words, docs, dictionary, calc4each=False):
    """
    Computes multiple topic coherence scores (C_V, C_UCI, C_NPMI) for a given topic model.
    
    :param topic_words: List of topics (each topic is a list of words)
    :param docs: List of tokenized documents
    :param dictionary: Gensim dictionary built from the dataset
    :param calc4each: Whether to return coherence scores for each topic separately
    :return: Tuple containing overall coherence scores and per-topic coherence scores (if requested)
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    #assert not any(len(doc) == 0 for doc in docs), "Found empty docs!"

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
            # 2) stop‐word & length filters
            if token in _NLTK_STOPS:
                continue
            if not (_MIN_LEN <= len(token)):
                continue

            # 3) lemmatize
            lemma = _LEMMATIZER.lemmatize(token)
            words.append(lemma)
    return words
    


def concept_analysis(config_analysis: str, config_model : str):
 
    # Force Hugging Face & SentenceTransformer to load from cache
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    #print(f"SentenceTransformer is searching in: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")
    
    #Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    cfg_analysis = EvaluationConfig.autoconfig(config_analysis)
    
    supported_approaches = ["ica","concept_shap","sae"]
    assert cfg_analysis.method_name in supported_approaches, f"Error: The method {cfg_analysis.method_name} is not supported in analysis. Currently the only supported methods are {supported_approaches}"

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
    dir_activations_with_labels = os.path.join(cfg_analysis.dir_acts_with_labels,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    labels_dataset = dataset_tokenized["token_labels"]
    nb_classes = len(np.unique(np.array(labels_dataset)))

    cache_activations_file_path = os.path.join(dir_activations_with_labels,f'layer_{cfg_analysis.hook_layer}.pkl')
    original_text_file_path = os.path.join(dir_activations_with_labels,f'layer_{cfg_analysis.hook_layer}.json')
    if not os.path.exists(dir_activations_with_labels):
            os.makedirs(dir_activations_with_labels)

    
    if not os.path.isfile(cache_activations_file_path):

            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
        
            activations_dataset, original_text_used = cache_activations_with_labels(hook_model,
                                                                                   dataset_tokenized,
                                                                                   data_collator,
                                                                                   tokenizer,
                                                                                   hook_name = cfg_analysis.hook_name,
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

    #Remove the template part
    for t,_ in enumerate(original_text_used):
        original_text_used[t] = original_text_used[t][:-len_template] 


    #Load the selected features
    segmented_features_dir_path = os.path.join(cfg_analysis.post_processed_features_path,"test",cfg_model.model_name,cfg_model.dataset_name,cfg_analysis.method_name)

     #Create file names
    if cfg_analysis.method_name=="sae":
        concept_model_name=f'{cfg_analysis.sae_name}_layer_{cfg_analysis.hook_layer}'
    elif cfg_analysis.method_name=="concept_shap":
        concept_model_name=f'concept_shap_{cfg_analysis.methods_args["n_concepts"]}_{cfg_analysis.methods_args["thres"]}_layer_{cfg_analysis.hook_layer}'
    elif cfg_analysis.method_name=='ica':
        concept_model_name=f'ica_{cfg_analysis.methods_args["n_components"]}_layer_{cfg_analysis.hook_layer}'
    segmented_features_file_path = os.path.join(segmented_features_dir_path,f"{concept_model_name}_segmented_features.json")
    selected_features_file_path = os.path.join(segmented_features_dir_path,f"{concept_model_name}_selected_features.json")
    
    if not os.path.isfile(selected_features_file_path):
        raise FileNotFoundError(f"File {selected_features_file_path} does not exist. Make sure to run 'select-features' before running the analysis of the extracted concepts")
    with open(selected_features_file_path, "r") as file:
        selected_features = json.load(file)
    with open(segmented_features_file_path, "r") as file:
        segmented_features = json.load(file)

    
    print(f"segmented_features : {segmented_features}")

    #----- Coherence based on cosine similarity of embedded sentences ------
    # Load the sentence embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Start the model")
    #sentence_embedding_model = SentenceTransformer('intfloat/e5-large-v2',device=device,local_files_only=True)
    sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2',device=device,local_files_only=True)
    # Generate embeddings
    print(f"Embedding of the sentences for interpretation of the concepts by cosine similarity distances")

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
        # Move back to CPU to free GPU memory
        embeddings_batch = embeddings_batch.detach().cpu()
        all_embeddings.append(embeddings_batch)

    
    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"sentence_embeddings shape : {sentence_embeddings.shape}")
    
    mean_cos_sim = compute_mean_cosine_similarity(sentence_embeddings)
    print(f"Mean Pairwise Cosine Similarity between all the embedded sentences of the dataset : {mean_cos_sim}")



    # ---- Tokenization: Convert Sentences to Lists of Words ---- #
    list_split_words = [ filter_sentences([sentence]) for sentence in  original_text_used]
    # list_split_words = [
    #     simple_preprocess(sentence, deacc=True, min_len=3)
    #     for sentence in original_text_used
    # ]
    
    # ---- Build Dictionary ---- #
    dictionary = Dictionary(list_split_words)
    dictionary.filter_extremes(no_below=5, no_above=0.8)  # Removes rare and common words
    #dictionary.id2token = {v: k for k, v in dictionary.token2id.items()}  # Fix potential Gensim bug
    dictionary.compactify()  

    if cfg_analysis.method_name=='concept_shap':

        n_concepts = cfg_analysis.methods_args['n_concepts']
        hidden_dim = cfg_analysis.methods_args['hidden_dim']
        thres = cfg_analysis.methods_args['thres']
        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(hook_model.cfg.device)
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0) #(dataset size, embedding_dim)
            
        interp_model = ConceptNet(n_concepts, embeddings_sentences_tensor, hidden_dim, thres).to(hook_model.cfg.device)
        interp_model.load_state_dict(torch.load(f'{cfg_analysis.path_to_dr_methods}_{thres}.pth'))

        concepts = interp_model.concept.T  #( n_concepts, embedding_dim)

        #Activations of the concepts for the tested embedded sentences
        with torch.no_grad():
            concepts_activations = interp_model.concepts_activations(embeddings_sentences_tensor)

        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_analysis.hook_layer}'
    

    elif cfg_analysis.method_name=='sae':

        
        #Load the local trained SAE
        tune_sae = TrainingSAE.load_from_pretrained(cfg_analysis.sae_path,cfg_analysis.methods_args['device'])

        #Retrieve the SAE activations
        concepts_activations = get_sae_activations(tune_sae,activations_dataset)
        concepts_activations = concepts_activations.squeeze(1)
        #print(f"sae_activations shape : {sae_activations.shape}")

        concept_model_name=f'{cfg_analysis.sae_name}_layer_{cfg_analysis.hook_layer}'
    
    elif cfg_analysis.method_name=='ica':

        interp_model = joblib.load(f'{cfg_analysis.path_to_dr_methods}.pkl')
        interp_model_components = interp_model.components_  # (n_concepts, embedding_dim)

        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(hook_model.cfg.device)
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0) #(dataset size, embedding_dim)
        embeddings_sentences_numpy = embeddings_sentences_tensor.cpu().numpy()
        concepts_activations = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)).to(hook_model.cfg.device) #shape : [batch size, num_components]

        concept_model_name=f'ica_{interp_model.components_.shape[0]}_layer_{cfg_analysis.hook_layer}'

    
    interpretability_list = []
    activating_sentences_list = []
    for feature in selected_features:
        activations_inspected = concepts_activations[:,feature]
        activations_inspected  = activations_inspected.cpu().numpy().reshape(-1,1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(activations_inspected)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        cluster_sizes = np.bincount(labels_kmeans)
        variances = [np.var(activations_inspected[labels_kmeans == i]) for i in range(len(centers))]
        
        # print("Cluster Centers:", centers.flatten())
        # print("Cluster Sizes:", cluster_sizes)
        labels = torch.from_numpy(labels_kmeans)
        
        
        #Discard the analysis of the feature if only one sentence in a cluster
        contains_0_or_1 = np.isin(cluster_sizes, [1]).any()
        if contains_0_or_1:
            index_activating_cluster = np.where(cluster_sizes == 1)[0]
            indices_activating_sentences = torch.nonzero((labels==index_activating_cluster), as_tuple=True)[0].tolist()
            #print(f" Supposé ne contenir que une valeur : {indices_activating_sentences}")
            activating_sentences_list.append(indices_activating_sentences)
            continue

        cos_sim_cluster = []
        for i in range(len(centers)):
            subset_embeddings = sentence_embeddings[labels==i,:]
            mean_cos_sim_concept = compute_mean_cosine_similarity(subset_embeddings)
            cos_sim_cluster.append(mean_cos_sim_concept)
        #print(f"cos_sim_cluster : {cos_sim_cluster}")
        #We take the mean pairwise cosine similarity of the cluster with the highest interpretability i.e. cluster associated to the sentences which activate the features

        index_activating_cluster = cos_sim_cluster.index(max(cos_sim_cluster))
        indices_activating_sentences = torch.nonzero((labels==index_activating_cluster), as_tuple=True)[0].tolist()
        activating_sentences_list.append(indices_activating_sentences)
        interpretability_list.append(max(cos_sim_cluster))


    iter_concept = 0
    concepts_words = []
    concepts_sentences_ids = []
    for index_feature,feature in enumerate(selected_features):
        iter_concept+=1
        activations_inspected = concepts_activations[:,feature]

        #If the number of sentences selected is above 500, we randomly select a maximum of 500 texts to prevent too much time of computation for 'calc_topic_coherence'
        if len(activating_sentences_list[index_feature]) > 500:
            sentence_indices_subset = random.sample(activating_sentences_list[index_feature], 500)
            activating_texts = [original_text_used[i] for i in sentence_indices_subset]
        else:
            activating_texts = [original_text_used[i] for i in activating_sentences_list[index_feature]]
        print(f"len(activating_texts) : {len(activating_texts)}")
        analyzed_text = filter_sentences(activating_texts)
        concepts_words.append(analyzed_text)
        cx = Counter(analyzed_text)
        most_occur = cx.most_common(25)
        print("Concept " + str(iter_concept) + " most common words:")
        print(most_occur)
        print("\n")

    # print("Size of concepts_words:", sys.getsizeof(concepts_words), "bytes")
    # print("Size of list_split_words:", sys.getsizeof(list_split_words), "bytes")
    # print("Size of dictionary:", sys.getsizeof(dictionary), "bytes")

    # ---- Compute Coherence ---- #
    (cv_score, c_uci_score, c_npmi_score) = calc_topic_coherence(concepts_words, list_split_words, dictionary, False)
    print(f"C_V: {cv_score:.4f}, C_UCI: {c_uci_score:.4f}, C_NPMI: {c_npmi_score:.4f}")
    # cv_score=0
    # c_uci_score=0
    # c_npmi_score=0
    #print(f"c_v per topic : {cv_per_topic}")

    def jaccard_similarity(set1, set2):
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

    #print(f"activating_sentences_list : {activating_sentences_list}")
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
    
    print(interpretability_list)
    print(f"Averaged Mean Pairwise Cosine Similarity of the most interpretable cluster for each concept : {np.mean(interpretability_list)}")

    projected_activations = concepts_activations[:,selected_features].cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=20, metric='manhattan')
    nbrs.fit(projected_activations)
    _, indices = nbrs.kneighbors(projected_activations)
    cos_sim_neighbors_all_sentences = []
    for i,sentence_embed in enumerate(sentence_embeddings):
        #Discard itself
        neighbor_indices = indices[i][1:]
        neighbors_embeddings = sentence_embeddings[neighbor_indices,:]
        cos_sim = F.cosine_similarity(neighbors_embeddings, sentence_embed.unsqueeze(0), dim=1)
        mean_cos_sim_neighbors = cos_sim.mean()
        cos_sim_neighbors_all_sentences.append(mean_cos_sim_neighbors.item())
    averaged_cos_sim_neighbors_all_sentences = np.mean(cos_sim_neighbors_all_sentences)
    print(f"averaged_cos_sim_neighbors_all_sentences : {averaged_cos_sim_neighbors_all_sentences}")

    # Get only the top 5 most activated concepts per sentence (in case this is not already set up)
    projected_activations = concepts_activations[:,selected_features].cpu()
    topk_values, topk_indices = torch.topk(projected_activations, k=5, dim=1)
    masked_projections = torch.zeros_like(projected_activations)
    masked_projections.scatter_(1, topk_indices, topk_values)

    #Get a binary vision of the concept activations
    activation_mask = masked_projections != 0  

    shared_counts = torch.matmul(activation_mask.float(), activation_mask.t().float())

    results = {}
    for i,sentence_embed in enumerate(sentence_embeddings):
        results[i] = {}
        for k in range(5, 0, -1):  # from 5 down to 1
            indices = (shared_counts[i] == k).nonzero(as_tuple=True)[0]
            indices = indices[indices != i]  # remove self-match
            if indices.numel() > 0:
                results[i][k] = indices.tolist()

    cos_sim_by_number_share_concepts = [[] for _ in range(5)]
    for i, common_dict in results.items():
        sentence_embed = sentence_embeddings[i,:]
        for k, idxs in common_dict.items():
            neighbors_embeddings = sentence_embeddings[idxs,:]
            cos_sim = F.cosine_similarity(neighbors_embeddings, sentence_embed.unsqueeze(0), dim=1)
            mean_cos_sim_neighbors = cos_sim.mean()
            cos_sim_by_number_share_concepts[k-1].append(mean_cos_sim_neighbors.item())

    mean_cos_sim_by_number_share_concepts = [ sum(cos_sim_pairwise_for_k)/len(cos_sim_pairwise_for_k) if cos_sim_pairwise_for_k!=[] else 0 for cos_sim_pairwise_for_k in  cos_sim_by_number_share_concepts ] 
   
    print(f"mean_cos_sim_by_number_share_concepts : {mean_cos_sim_by_number_share_concepts}")
    
        

    # inertias = []
    # K_range = range(20, 50)

    # projected_activations = concepts_activations[:,selected_features].cpu().numpy()
    # for k in K_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans.fit(projected_activations)
    #     inertias.append(kmeans.inertia_)

    # print(f"inertias : {inertias}")
    # min_index = inertias.index(min(inertias))
    # min_k = K_range[min_index]
    # print(f"min_k : {min_k}")
    # kmeans = KMeans(n_clusters=min_k, random_state=42, n_init=10)
    # labels = kmeans.fit_predict(projected_activations)

    # cos_sim_cluster = []
    # for cluster_label in range(min_k):
    #     subset_embeddings = sentence_embeddings[labels==cluster_label,:]
    #     mean_cos_sim_concept = compute_mean_cosine_similarity(subset_embeddings)
    #     cos_sim_cluster.append(mean_cos_sim_concept)
    # print(f"cos_sim_cluster : {cos_sim_cluster}")

    dict_interpretability_metrics = {"avg_jaccard" : avg_jaccard, "redundancy" : redundancy, "coverage" : coverage, "cv_score" : cv_score, "c_uci_score" : c_uci_score, "c_npmi_score" : c_npmi_score, "Averaged mean cosine similarity over the dataset" : mean_cos_sim ,"cos_sim_clusters" : np.mean(interpretability_list), 'averaged_cos_sim_neighbors_all_sentences' : averaged_cos_sim_neighbors_all_sentences, 'Averaged cosine similarity by number shared concepts' : mean_cos_sim_by_number_share_concepts}

    dir_to_save_metrics = os.path.join(cfg_analysis.metrics_reconstruction,cfg_model.model_name,cfg_model.dataset_name,cfg_analysis.method_name,concept_model_name)
    dir_interpretability_results = os.path.join(dir_to_save_metrics,"interpretability")
    if not os.path.exists(dir_interpretability_results):
        os.makedirs(dir_interpretability_results)

    file_to_save_intepretability_metrics = os.path.join(dir_interpretability_results,f"{concept_model_name}_interpretability_results.json") 
    with open(file_to_save_intepretability_metrics, 'w') as file:
        json.dump(dict_interpretability_metrics, file, indent=4)

'''
    if cfg_analysis.method_name=='concept_shap':

        n_concepts = cfg_analysis.methods_args['n_concepts']
        hidden_dim = cfg_analysis.methods_args['hidden_dim']
        thres = cfg_analysis.methods_args['thres']
        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(hook_model.cfg.device)
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0) #(dataset size, embedding_dim)
            
        interp_model = ConceptNet(n_concepts, embeddings_sentences_tensor, hidden_dim, thres).to(hook_model.cfg.device)
        interp_model.load_state_dict(torch.load(f'{cfg_analysis.path_to_dr_methods}_{thres}.pth'))

        concepts = interp_model.concept.T  #( n_concepts, embedding_dim)

        #Activations of the concepts for the tested embedded sentences
        with torch.no_grad():
            concepts_activations = interp_model.concepts_activations(embeddings_sentences_tensor)
        #print(f"concept_score_thres_prob shape : {concept_score_thres_prob.shape}")

        # iter_concept = 0
        # concepts_words = []
        # concepts_sentences_ids = []
        # for concept in concepts:
        #     iter_concept+=1
        #     distance = torch.norm(embeddings_sentences_tensor - concept, dim=1)
        #     knn = distance.topk(100, largest=False).indices
    
        #     neighbors_sentences = []
        #     neighbords_ids = []
        #     for idx in knn :
        #         neighbors_sentences.append(original_text_used[idx])
        #         neighbords_ids.append(idx)
    
        #     analyzed_text = filter_sentences(neighbors_sentences)
        #     concepts_words.append(analyzed_text)
        #     concepts_sentences_ids.append(neighbords_ids)
        #     cx = Counter(analyzed_text)
        #     most_occur = cx.most_common(25)
        #     print("Concept " + str(iter_concept) + " most common words:")
        #     print(most_occur)
        #     print("\n")

        concept_model_name=f'concept_shap_{n_concepts}_{thres}_layer_{cfg_analysis.hook_layer}'

        iter_concept = 0
        concepts_words = []
        concepts_sentences_ids = []
        for feature in selected_features:
            iter_concept+=1
            activations_inspected = concepts_activations[:,feature]
            #print(f"activations_inspected shape : {activations_inspected}")
            top_values, top_indices = torch.topk(activations_inspected,200)
            top_texts = [original_text_used[i] for i in top_indices]
            top_texts_ids = top_indices.tolist()
            analyzed_text = filter_sentences(top_texts)
            concepts_words.append(analyzed_text)
            concepts_sentences_ids.append(top_texts_ids)
            cx = Counter(analyzed_text)
            most_occur = cx.most_common(25)
            print("Concept " + str(iter_concept) + " most common words:")
            print(most_occur)
            print("\n")
    
    
    elif cfg_analysis.method_name=='sae':

        
        #Load the local trained SAE
        tune_sae = TrainingSAE.load_from_pretrained(cfg_analysis.sae_path,cfg_analysis.methods_args['device'])

        #Retrieve the SAE activations
        concepts_activations = get_sae_activations(tune_sae,activations_dataset)
        concepts_activations = concepts_activations.squeeze(1)
        #print(f"sae_activations shape : {sae_activations.shape}")

        concept_model_name=f'{cfg_analysis.sae_name}_layer_{cfg_analysis.hook_layer}'

        
        #Retrieve the most activated sentences by the kept SAE features
        concepts_words = []
        concepts_sentences_ids = []
        iter_concept = 0
        for feature in selected_features:
            iter_concept+=1
            activations_inspected = concepts_activations[:,feature]
            top_values, top_indices = torch.topk(activations_inspected,200)
            top_texts = [original_text_used[i] for i in top_indices]
            top_texts_ids = top_indices.tolist()
            analyzed_text = filter_sentences(top_texts)
            concepts_words.append(analyzed_text)
            concepts_sentences_ids.append(top_texts_ids)
            cx = Counter(analyzed_text)
            most_occur = cx.most_common(25)
            print("Concept " + str(iter_concept) + " most common words:")
            print(most_occur)
            print("\n")

    
    elif cfg_analysis.method_name=='ica':

        interp_model = joblib.load(f'{cfg_analysis.path_to_dr_methods}.pkl')
        interp_model_components = interp_model.components_  # (n_concepts, embedding_dim)

        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(hook_model.cfg.device)
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0) #(dataset size, embedding_dim)
        embeddings_sentences_numpy = embeddings_sentences_tensor.cpu().numpy()
        concepts_activations = torch.from_numpy(interp_model.transform(embeddings_sentences_numpy)).to(hook_model.cfg.device) #shape : [batch size, num_components]
        #print(f"ica activations shape : {ica_activations.shape}")

        concept_model_name=f'ica_{interp_model.components_.shape[0]}_layer_{cfg_analysis.hook_layer}'
        
        #Retrieve the most activated sentences for each component
        iter_concept = 0
        concepts_words = []
        concepts_sentences_ids = []
        for feature in selected_features:
            iter_concept+=1
            activations_inspected = concepts_activations[:,feature]
            top_values, top_indices = torch.topk(activations_inspected,200)
            top_texts = [original_text_used[i] for i in top_indices]
            top_texts_ids = top_indices.tolist()
            analyzed_text = filter_sentences(top_texts)
            concepts_words.append(analyzed_text)
            concepts_sentences_ids.append(top_texts_ids)
            cx = Counter(analyzed_text)
            most_occur = cx.most_common(25)
            print("Concept " + str(iter_concept) + " most common words:")
            print(most_occur)
            print("\n")
    
    # ---- Compute Coherence ---- #
    (cv_score, c_uci_score, c_npmi_score), (cv_per_topic, c_uci_per_topic, c_npmi_per_topic) = calc_topic_coherence(concepts_words, list_split_words, dictionary, True)
    print(f"C_V: {cv_score:.4f}, C_UCI: {c_uci_score:.4f}, C_NPMI: {c_npmi_score:.4f}")
    print(f"c_v per topic : {cv_per_topic}")


    #----- Coherence based on cosine similarity of embedded sentences ------
    # Load the sentence embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Start the model")
    #sentence_embedding_model = SentenceTransformer('intfloat/e5-large-v2',device=device,local_files_only=True)
    sentence_embedding_model = SentenceTransformer('all-MiniLM-L6-v2',device=device,local_files_only=True)
    # Generate embeddings
    print(f"Embedding of the sentences for interpretation of the concepts by cosine similarity distances")

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
        # Move back to CPU to free GPU memory
        embeddings_batch = embeddings_batch.detach().cpu()
        all_embeddings.append(embeddings_batch)

    
    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"sentence_embeddings shape : {sentence_embeddings.shape}")
    
    mean_cos_sim = compute_mean_cosine_similarity(sentence_embeddings)
    print(f"Mean Pairwise Cosine Similarity between all the embedded sentences of the dataset : {mean_cos_sim}")

    #To compare with the "similarity" between the sentences attached to a same concept, we compute the mean pairwise cosine similarity between random sentences
    k_select = len(concepts_sentences_ids[0])
    mean_cos_sim_random_concepts = []
    for p in range(10):
        random_indices = torch.randperm(sentence_embeddings.shape[0])[:k_select] 
        subset_embeddings = sentence_embeddings[random_indices,:]
        mean_cos_sim_concept = compute_mean_cosine_similarity(subset_embeddings)
        mean_cos_sim_random_concepts.append(mean_cos_sim_concept)
    mean_cos_sim_random_concept = np.mean(mean_cos_sim_random_concepts)
    print(f"Mean Pairwise Cosine Similarity between randomly selected embedded sentences : {mean_cos_sim_random_concept} \n")
        
    
    cos_sim_within_concepts = []
    for i,concept_ids in enumerate(concepts_sentences_ids):

        subset_embeddings = sentence_embeddings[concept_ids,:]
        mean_cos_sim_concept = compute_mean_cosine_similarity(subset_embeddings)
        cos_sim_within_concepts.append(mean_cos_sim_concept)
        print(f"Mean Pairwise Cosine Similarity between the embedded sentences associated to the concept {i+1} : {mean_cos_sim_concept}")

    
    print(f"Averaged Mean Pairwise Cosine Similarity between the embedded sentences associated to the concept across the concepts : {np.mean(cos_sim_within_concepts)}")
    print(f"Top Mean Pairwise Cosine Similarities :  {sorted(cos_sim_within_concepts, reverse=True)[:10]}")

    def jaccard_similarity(set1, set2):
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

    sets = [set(lst) for lst in concepts_sentences_ids]
    
    # Compute average pairwise Jaccard similarity
    jaccard_scores = [
        jaccard_similarity(sets[i], sets[j])
        for i, j in combinations(range(len(sets)), 2)
    ]
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
    
    # Compute redundancy
    all_elements = [x for sublist in concepts_sentences_ids for x in sublist]
    element_counts = Counter(all_elements)
    redundancy = sum(element_counts.values()) / len(element_counts)

    #Coverage
    coverage = len(element_counts) / len(original_text_used)
    
    print(f"avg_jaccard : {avg_jaccard}")
    print(f"redundancy : {redundancy}")
    print(f"coverage : {coverage}")
        

    interpretability_list = []
    
    for feature in selected_features:
        activations_inspected = concepts_activations[:,feature]
        activations_inspected  = activations_inspected.cpu().numpy().reshape(-1,1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(activations_inspected)
        
        # Get cluster centers
        centers = kmeans.cluster_centers_
        cluster_sizes = np.bincount(labels_kmeans)
        variances = [np.var(activations_inspected[labels_kmeans == i]) for i in range(len(centers))]
        
        print("Cluster Centers:", centers.flatten())
        print("Cluster Sizes:", cluster_sizes)
        #Discard the analysis of the feature if less than one sentence in a cluster
        contains_0_or_1 = np.isin(cluster_sizes, [0, 1]).any()
        if contains_0_or_1:
            continue

        labels = torch.from_numpy(labels_kmeans)
        cos_sim_cluster = []
        for i in range(len(centers)):
            subset_embeddings = sentence_embeddings[labels==i,:]
            mean_cos_sim_concept = compute_mean_cosine_similarity(subset_embeddings)
            cos_sim_cluster.append(mean_cos_sim_concept)
        print(f"cos_sim_cluster : {cos_sim_cluster}")
        #We take the mean pairwise cosine similarity of the cluster with the highest interpretability i.e. cluster associated to the sentences which activate the features
        interpretability_list.append(max(cos_sim_cluster))

    print(f"Averaged Mean Pairwise Cosine Similarity of the most interpretable cluster for each concept : {np.mean(interpretability_list)}")

    projected_activations = concepts_activations[:,selected_features].cpu().numpy()
    nbrs = NearestNeighbors(n_neighbors=20, metric='manhattan')
    nbrs.fit(projected_activations)
    _, indices = nbrs.kneighbors(projected_activations)
    cos_sim_neighbors_all_sentences = []
    for i,sentence_embed in enumerate(sentence_embeddings):
        #DIscard itself
        neighbor_indices = indices[i][1:]
        neighbors_embeddings = sentence_embeddings[neighbor_indices,:]
        cos_sim = F.cosine_similarity(neighbors_embeddings, sentence_embed.unsqueeze(0), dim=1)
        mean_cos_sim_neighbors = cos_sim.mean()
        cos_sim_neighbors_all_sentences.append(mean_cos_sim_neighbors.item())
    averaged_cos_sim_neighbors_all_sentences = np.mean(cos_sim_neighbors_all_sentences)
    print(f"averaged_cos_sim_neighbors_all_sentences : {averaged_cos_sim_neighbors_all_sentences}")
        
        

    # inertias = []
    # K_range = range(20, 50)

    # projected_activations = concepts_activations[:,selected_features].cpu().numpy()
    # for k in K_range:
    #     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    #     kmeans.fit(projected_activations)
    #     inertias.append(kmeans.inertia_)

    # print(f"inertias : {inertias}")
    # min_index = inertias.index(min(inertias))
    # min_k = K_range[min_index]
    # print(f"min_k : {min_k}")
    # kmeans = KMeans(n_clusters=min_k, random_state=42, n_init=10)
    # labels = kmeans.fit_predict(projected_activations)

    # cos_sim_cluster = []
    # for cluster_label in range(min_k):
    #     subset_embeddings = sentence_embeddings[labels==cluster_label,:]
    #     mean_cos_sim_concept = compute_mean_cosine_similarity(subset_embeddings)
    #     cos_sim_cluster.append(mean_cos_sim_concept)
    # print(f"cos_sim_cluster : {cos_sim_cluster}")

    dict_interpretability_metrics = {"avg_jaccard" : avg_jaccard, "redundancy" : redundancy, "coverage" : coverage, "cv_score" : cv_score, "c_uci_score" : c_uci_score, "c_npmi_score" : c_npmi_score, "cos_sim_top_sentence" : np.mean(cos_sim_within_concepts), "cos_sim_clusters" : np.mean(interpretability_list), 'averaged_cos_sim_neighbors_all_sentences' : averaged_cos_sim_neighbors_all_sentences}

    dir_to_save_metrics = os.path.join(cfg_analysis.metrics_reconstruction,cfg_model.model_name,cfg_model.dataset_name,cfg_analysis.method_name,concept_model_name)
    dir_interpretability_results = os.path.join(dir_to_save_metrics,"interpretability")
    if not os.path.exists(dir_interpretability_results):
        os.makedirs(dir_interpretability_results)

    file_to_save_intepretability_metrics = os.path.join(dir_interpretability_results,f"{concept_model_name}_interpretability_results.json") 
    with open(file_to_save_intepretability_metrics, 'w') as file:
        json.dump(dict_interpretability_metrics, file, indent=4)


'''

        
            
        
        