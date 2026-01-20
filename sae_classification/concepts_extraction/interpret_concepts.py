import re
import nltk
from typing import Sequence, Optional, Dict, List, Tuple, Iterable
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import os
import shutil
import gc
import torch
from pathlib import Path
import numpy as np, jenkspy
import joblib
from loguru import logger
from ..utils import LLMLoadConfig, EvaluationConceptsConfig
from .evaluation_utils import (
    collate_single_sample,
    is_empty_dir,
    load_all_shards,
    load_interp_model,
    create_activations_dataset
)
from .caching import max_seq_length_hook
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from ..llm_classifier_tuning import get_hook_model, get_model, _init_tokenizer, _max_seq_length, count_template_units
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
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
        logger.warning(f"NLTK corpus missing: {pkg}; attempting download")
        nltk.download(pkg,quiet=True)

# compile once at module load
_WORD_RE  = re.compile(r"\b[a-zA-Z]+\b")
_NLTK_STOPS = set(stopwords.words("english"))
_MIN_LEN = 3
_LEMMATIZER = WordNetLemmatizer()

# Get features activations from z_class for the SAE
def get_sae_activations(sae,activations_dataset,device):

    feature_activations_list = []
    activations_dataloader = DataLoader(activations_dataset,batch_size=1,collate_fn=collate_single_sample)
    sae.eval()
    with torch.no_grad():
        
        for batch in tqdm(activations_dataloader, desc="Get the SAE activations", unit="batch"):
            cache = batch["cache"].to(device)
            cache_sentence = cache.unsqueeze(1)
    
            # Use the SAE
            feature_acts, acts_without_process = sae.encode_with_hidden_pre(cache_sentence)
            
            feature_activations_list.append(feature_acts.cpu())
        
        feature_activations_tensor = torch.cat(feature_activations_list,dim=0)

    return feature_activations_tensor.squeeze(1)


def make_wordcloud(
    raw_content: str,
    width: int = 800,
    height: int = 400,
    background_color: str = "white",
    save_basepath: str | None = None,
    dpi: int = 300,
    **wordcloud_kwargs
):
  
    
    #  Parse "word:count" pairs (int or float)
    freqs: dict[str, float] = {}
    for pair in raw_content.split(","):
        if ":" not in pair:
            continue
        word, count_str = pair.split(":")
        word = word.strip()
        try:
            freqs[word] = float(count_str)
        except ValueError:
            # Skip malformed numeric fields gracefully
            continue

    if not freqs:
        raise ValueError("No valid 'word:count' pairs found.")

    # Build the WordCloud
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        colormap="Dark2",
        **wordcloud_kwargs,
    ).generate_from_frequencies(freqs)

    # Render
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor=background_color)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    # Save if requested
    if save_basepath:
        path = Path(save_basepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(path.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight")
        # Or: wc.to_file(path.with_suffix(".png"))
    
    plt.close(fig)
    return wc


def compute_mean_cosine_similarity(sentence_embeddings):

    # Compute cosine similarity matrix
    sentence_embeddings_norm = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    s = sentence_embeddings_norm.sum(dim=0)        # (D,)
    sq_norm_s = (s * s).sum()                      # scalar
    N = sentence_embeddings_norm.size(0)

    mean_cos_sim = (sq_norm_s - N) / (N * (N - 1))

    return mean_cos_sim.item()

def compute_mean_var_cosine(sentence_embeddings,d_block=256):
    sentence_embeddings_norm = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    N, D = sentence_embeddings_norm.shape
   
    s = sentence_embeddings_norm.sum(dim=0)
    mu = ((s * s).sum() - N) / (N * (N - 1))

    # Frobenius norm of G = X^T X without storing G
    fro_sq = 0.0

    for a in range(0, D, d_block):
        Xa = sentence_embeddings_norm[:, a:a + d_block]
        # diagonal block
        Ga = Xa.T @ Xa                                 
        fro_sq += (Ga * Ga).sum().item()

        #The off-diagonal blocks
        for b in range(a + d_block, D, d_block):
            Xb = sentence_embeddings_norm[:, b:b + d_block]
            Gab = Xa.T @ Xb                             
            fro_sq += 2.0 * (Gab * Gab).sum().item()    # *2 for symmetry

    m2 = (fro_sq - N) / (N * (N - 1)) 
    var = m2 - mu * mu

    return mu.item(), var.item()

# Weighted Average on the number of pairs of sentences per set of concept activating cluster. Each ConceptSim(j) is averaged in ConceptSim with these weights
def compute_weighted_avg_concept_sim(concept_sim_list,activating_sentences_size):

    # The average of ConceptSim over the learned concepts is weighted by the number of pairs of sentences labeled as 'activating' for each concept. 
    # To avoid rewarding the case with many small interpretable concepts and one large concept activating on many sentences.

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

class StreamedDocs:
    def __init__(self, docs, dictionary):
        self.docs = docs
        self.dictionary = dictionary
    def __iter__(self):
        for doc in self.docs:
            yield [t for t in doc if t in self.dictionary.token2id]


def calc_topic_coherence(
    topic_words: Sequence[Sequence[str]],
    docs: Iterable[Sequence[str]],
    dictionary: Dictionary,
    activating_cluster_sizes: Sequence[float],
    calc4each: bool = False,
    topn: int = 10,
    window_size_cv: int = 110,
    window_size_pmi: int = 10,
) -> Tuple[Tuple[float, float, float], Tuple[List[float], List[float], List[float]] | None]:
    """
    Compute topic coherence scores (C_V, C_UCI, C_NPMI).
    Uses top-N words per topic (default N=10) — a common choice in the literature.
    Window sizes are made explicit for transparency (defaults mirror common practice).
    """
    topic_words = list(topic_words)
    activating_cluster_sizes = np.asarray(activating_cluster_sizes, dtype=float)

    if len(topic_words) != len(activating_cluster_sizes):
        raise ValueError(
            f"Length mismatch: {len(topic_words)} topics but "
            f"{len(activating_cluster_sizes)} cluster sizes."
        )

    # Dedup per topic (order-preserving), filter to dictionary, then trim to topn
    def dedup_keep_order(words):
        seen = set()
        out = []
        for w in words:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    aligned_topic_words: List[List[str]] = []
    kept_indices: List[int] = []
    dropped_indices: List[int] = []

    for i, words in enumerate(topic_words):
        clean = [w for w in dedup_keep_order(words) if w in dictionary.token2id]
        if clean:
            aligned_topic_words.append(clean[:topn])
            kept_indices.append(i)
        else:
            dropped_indices.append(i)

    if not aligned_topic_words:
        raise ValueError(
            "All topics were dropped after filtering by dictionary; "
            "no coherence can be computed."
        )

    # Ensure docs are re-iterable across multiple coherence computations
    texts_stream = StreamedDocs(docs, dictionary)  # must support multiple passes

    def coherence_per_topic(measure: str, window_size: int | None = None) -> np.ndarray:
        kwargs = dict(
            topics=aligned_topic_words,
            texts=texts_stream,
            dictionary=dictionary,
            coherence=measure,
            topn=topn,
            processes=1,  # reproducible
        )
        if window_size is not None:
            kwargs["window_size"] = window_size
        model = CoherenceModel(**kwargs)
        return np.asarray(model.get_coherence_per_topic(), dtype=float)

    c_npmi_per_topic = coherence_per_topic("c_npmi", window_size_pmi)
    cv_per_topic     = coherence_per_topic("c_v",    window_size_cv)
    c_uci_per_topic  = coherence_per_topic("c_uci",  window_size_pmi)

    weights = activating_cluster_sizes[kept_indices]
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Weights must be non-negative and finite.")
    if weights.sum() == 0:
        # fallback to unweighted mean if all weights are zero
        weights = np.ones_like(weights)

    cv_score    = float(np.average(cv_per_topic, weights=weights))
    c_uci_score = float(np.average(c_uci_per_topic, weights=weights))
    c_npmi_score= float(np.average(c_npmi_per_topic, weights=weights))

    if calc4each:
        return (
            (cv_score, c_uci_score, c_npmi_score),
            (cv_per_topic.tolist(), c_uci_per_topic.tolist(), c_npmi_per_topic.tolist()),
        )

    return (cv_score, c_uci_score, c_npmi_score), None

        


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


def load_sentence_encoder(
    name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str | None = None,
) -> SentenceTransformer:
    
    kw = dict(device=device)
    
    # Try offline (cache only)
    try:
        return SentenceTransformer(name, local_files_only=True, **kw)
    except Exception:
        logger.info(f"{name} not found in cache; downloading…")

    # Fallback to online (will cache for next time)
    return SentenceTransformer(name, local_files_only=False, **kw)


def embed_sentences(
    texts: Sequence[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> torch.Tensor:
    """
    Load a SentenceTransformer offline-first, then encode `texts` in batches.
    Returns a CPU tensor of shape [N, D].
    """
    if not texts:
        return torch.empty(0, 0)

    model = load_sentence_encoder(model_name, device=device)

    # SentenceTransformer handles batching & progress bar for us
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_tensor=True,
        device=device,                      # explicitly place on device
        show_progress_bar=show_progress,
    )

    # Always return on CPU for downstream CPU ops / saving
    emb_cpu = emb.detach().cpu()
    del emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb_cpu
       


def prepare_clean_texts(
    sentences: List[str],
    *,
    bigram_min_count: int = 5,
    bigram_threshold: float = 15.0,
    trigram_threshold: float = 10.0,
) -> List[List[str]]:
    """
    Turn raw sentences into cleaned token lists:
      1) tokenize (regex over [a-zA-Z]+), lowercase
      2) mine phrases up to trigrams (2-grams then 3-grams)
      3) split anything longer than trigram back into single tokens
      4) drop stop-words / short tokens
      5) lemmatize (WordNet)
    Returns: list of token lists, one per sentence.
    """
    # Step 1 — regex tokenization
    tokenized = [_WORD_RE.findall(s.lower()) for s in sentences]

    # Step 2 — phrase mining
    bigram_phr = Phraser(Phrases(
        tokenized, min_count=bigram_min_count, threshold=bigram_threshold, delimiter="_"
    ))
    trigram_phr = Phraser(Phrases(
        bigram_phr[tokenized], threshold=trigram_threshold, delimiter="_"
    ))

    def merge(tokens: List[str]) -> List[str]:
        # apply 2- and 3-grams, then split >3-word phrases back
        toks = trigram_phr[bigram_phr[tokens]]
        out: List[str] = []
        for t in toks:
            if t.count("_") > 2:  # > 3 words? split it back for readability/visualization
                out.extend(t.split("_"))
            else:
                out.append(t)
        return out

    # Step 3 — apply phrase mappers
    with_phrases = [merge(toks) for toks in tokenized]

    # Step 4–5 — stopword/length filter + lemmatize
    clean_texts: List[List[str]] = []
    for sent in with_phrases:
        keep = []
        for tok in sent:
            if tok in _NLTK_STOPS:        # remove stops
                continue
            if len(tok) < _MIN_LEN:       # drop very short tokens
                continue
            keep.append(_LEMMATIZER.lemmatize(tok))
        clean_texts.append(keep)

    return clean_texts


def build_concept_docs_and_tfidf(
    *,
    clean_texts: List[List[str]],
    per_concept_sentences: Dict[int, List[str]],
    original_texts: List[str],
    top_k: int = 20,
    min_df: int = 2,
) -> Tuple[Dict[int, List[Tuple[str, float]]], Dict[int, str]]:
    """
    Build per-concept 'documents' by concatenating the cleaned tokens of each
    concept's 'activating sentences', then compute TF-IDF and return the top-k
    terms per concept.

    Args:
      clean_texts: output from `prepare_clean_texts` (same order as original_texts)
      per_concept_sentences: mapping concept_id -> list of *raw* sentences for that concept
      original_texts: the full list of raw sentences (same order as clean_texts)
      top_k: number of keywords to keep
      min_df: TF-IDF min doc frequency

    Returns:
      concept_terms: {concept_id: [(term, score), ... top_k]}
      concept_docs:  {concept_id: "joined tokens ..."} (no corpus doc)
    """
    # map raw sentence -> cleaned tokens for quick lookup
    sent2tokens = dict(zip(original_texts, clean_texts))

    # Per-concept documents (space-joined tokens)
    concept_docs: Dict[int, str] = {
        cid: " ".join(" ".join(sent2tokens[s]) for s in sentences if s in sent2tokens)
        for cid, sentences in per_concept_sentences.items()
    }

    # Add a pseudo-document for the whole corpus to stabilize TF/IDF weighting;
    # we’ll strip it before extracting per-concept top-k.
    full_corpus_doc = " ".join(" ".join(toks) for toks in clean_texts)
    docs_with_corpus = {**concept_docs, "__FULL_CORPUS__": full_corpus_doc}

    # Vectorize (we already preprocessed; keep tokenizer/preprocessor trivial)
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=lambda x: x,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=min_df,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(docs_with_corpus.values())
    terms = vectorizer.get_feature_names_out()

    # Strip last row (the corpus doc)
    X_concepts = X[:-1]

    # Top-k terms per concept (preserve dict insertion order to align rows)
    concept_ids = list(concept_docs.keys())
    concept_terms: Dict[int, List[Tuple[str, float]]] = {}
    for row_idx, cid in enumerate(concept_ids):
        row = X_concepts[row_idx].toarray().ravel()
        idx = row.argsort()[-top_k:][::-1]
        concept_terms[cid] = [(terms[j], float(round(row[j], 4))) for j in idx]

    return concept_terms, concept_docs

       
# Main function to compute interpretabiliy metrics for each extracted concept present in z_class. In particular, the reported ConceptSim and SentenceSim presented in the paper
def concept_interpretability(config_analysis: str, config_model : str):
 
    # # Force Hugging Face & SentenceTransformer to load from cache
    # os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # os.environ["HF_DATASETS_OFFLINE"] = "1"

    print("\n######################################## BEGIN : Computation of Interpretability Metrics on z_class Concepts ########################################")

    
    ####################################################### Models, Datasets, Concept-based models loading #######################################################
    
    # Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    # Retrieve the config of the extracting concept method (centralize for features selection and recovery accuracy, causality and interpretability measures)
    cfg_concept = EvaluationConceptsConfig.autoconfig(config_analysis)
    
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
        
        # Retrieve the length of the template added at the end of the sentence
        len_template,_,_ = count_template_units(tokenizer,cfg_model)

    else: # (Encoder only LM for classification)
        model = get_model(cfg_model,decoder,nb_classes) 
        labels_tokens_id = None
        device = model.device

        max_ctx = _max_seq_length(model)
        add_template = decoder
        eos = False
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        len_template = 1
        
    
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
    
    # Remove the template part
    for t,_ in enumerate(original_text_used):
        original_text_used[t] = original_text_used[t][:-len_template]
        
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


    ########################################################### Interpretability Metrics Calculation ###########################################################
    
    
    
    ########### Compute the activations of the extracted concepts on the investigated dataset ###########
    
    # Dataloader on the cached activations
    activations_dataloader = DataLoader(activations_dataset,batch_size=1,collate_fn=collate_single_sample)
   
    # Set up everything to load the evaluated concept-based method
    if cfg_concept.method_name=='concept_shap':

        # Activations of the concepts for the tested embedded sentences
        concepts_activations_list = []
        with torch.no_grad():
            for batch in activations_dataloader:
                cache_sentence = batch["cache"].to(device_interp_model)
                concepts_activations_batch, _, _ = interp_model.concepts_activations(cache_sentence)
                concepts_activations_list.append(concepts_activations_batch)
            concepts_activations = torch.cat(concepts_activations_list, dim=0) #(dataset size, n_concepts)
    
    elif cfg_concept.method_name=='hi_concept':

        concepts_activations_list = []
        with torch.no_grad():
            for batch in activations_dataloader:
                cache_sentence = batch["cache"].to(device_interp_model)
                concepts_activations_batch, _, _, _,_ = interp_model.concepts_activations(cache_sentence)
                concepts_activations_list.append(concepts_activations_batch)
            concepts_activations = torch.cat(concepts_activations_list, dim=0) #(dataset size, n_concepts)


    elif cfg_concept.method_name=='ica':

        concepts_activations_list = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(device_interp_model)
            concepts_activations_list.append(cache_sentence)
        concepts_activations_tensor = torch.cat(concepts_activations_list, dim=0) #(dataset size, embedding_dim)
        concepts_activations_numpy = concepts_activations_tensor.cpu().numpy()
        concepts_activations = torch.from_numpy(interp_model.transform(concepts_activations_numpy)) #shape : [batch size, num_components]


    elif cfg_concept.method_name=='sae':
 
        # Retrieve the SAE activations
        concepts_activations = get_sae_activations(interp_model,activations_dataset,device_interp_model)
    

    ########## Load the sentence encoder and encoding of the evaluated sentences ##################
    
    logger.info(f"Loading of the sentence encoder and encoding of the evaluated sentences to use the cosine similarity metric")
    sentence_embeddings = embed_sentences(
        original_text_used,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=32,
    )
    gc.collect()

    # Compute the mean and variance of the sentences pairwise cosine similarity distribution over the inspected dataset 
    mean_cos_sim, var_cos_sim = compute_mean_var_cosine(sentence_embeddings)
    logger.info(f"Mean Pairwise Cosine Similarity between all the embedded sentences of the dataset : {mean_cos_sim}")
    logger.info(f"Variance Pairwise Cosine Similarity between all the embedded sentences of the dataset : {var_cos_sim}")
   
    # Directory where to save the interpretability metrics
    dir_to_save_metrics = os.path.join(concepts_dir_path,cfg_model.split)
    dir_interpretability_results = os.path.join(dir_to_save_metrics,"interpretability")
    os.makedirs(dir_interpretability_results, exist_ok=True)
    
    
    ########## Computation of ConceptSim(j) for each feature j ##################
    
    activating_sentences_list = []
    
    concept_sim_dict = {}
    concept_size_cluster = {}

    # Cluster the 'activating' sentences for each feature in z_class
    for feature in selected_features:

        # Vector of activations of the feature across the evaluated dataset of sentences 
        activations_inspected = concepts_activations[:,feature]
        activations_inspected  = activations_inspected.cpu().numpy().reshape(-1,1)
        
        # 1D Clustering
        vals = np.asarray(activations_inspected).ravel()
        
        # In case all values are identical
        if np.allclose(vals, vals[0]):
            # no separation possible
            activating_cluster_sentences_indices = np.array([], dtype=int)
            concept_sim_dict[feature] = 0.0
            concept_size_cluster[feature] = 0
            activating_sentences_list.append([])
            continue
        
        t = jenkspy.jenks_breaks(vals.squeeze(), n_classes=2)[1]  # single split
        activated = ((vals > t).astype(bool)).squeeze()             # 1 = activation

        # Save the indices of the sentences present in the concept 'activating' cluster
        activating_cluster_sentences_indices = np.where(activated == 1)[0]
        activating_sentences_list.append(activating_cluster_sentences_indices.tolist())

        # Discard the computation of ConceptSim(j) if there is only one sentence in the activating cluster
        contains_0_or_1 = np.sum(activated) <= 1
        if contains_0_or_1:
            concept_sim_dict[feature] = 0
            concept_size_cluster[feature] = 1
            continue
        
        activating_cluster_sentences_embeddings = sentence_embeddings[torch.from_numpy(activated)]

        # Compute ConceptSim(feature)
        mean_cos_sim_concept = compute_mean_cosine_similarity(activating_cluster_sentences_embeddings)

        # Number of sentences classified as "activating sentences" for the feature
        activating_cluster_size = np.sum(activated)

        
        # Add the ConceptSim(j) metric of the current concept j
        concept_sim_dict[feature] = mean_cos_sim_concept
        concept_size_cluster[feature] = activating_cluster_size
    
    
    conceptsim_list = [concept_sim_dict[feature] for feature in concept_sim_dict.keys()]
    activating_clusters_sizes_list = [concept_size_cluster[feature] for feature in concept_sim_dict.keys()]
    # Weighted Averaged ConceptSim (Metric reported in the paper)
    weighted_avg_concept_sim = compute_weighted_avg_concept_sim(conceptsim_list,activating_clusters_sizes_list)
    
    # Unweighted Averaged ConceptSim (Flawed metric as explained in section D)
    unweighted_avg_concept_sim = np.mean(conceptsim_list)

    print(f"Unweighted Averaged Concept Cosine Similarity across all concepts : {unweighted_avg_concept_sim}")
    print(f"Weighted Averaged Concept Cosine Similarity across all concepts : {weighted_avg_concept_sim}")


    
    
    
    ############ Compute SentenceSim(k) for k ranging from 1 to 4 ###############
    
    # Get the top 5 most activated concepts per sentence
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
    
    
    ############# Compute  Jaccard, Redundancy, Coverage ###############
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
    redundancy = sum(element_counts.values()) / len(element_counts) if element_counts else 0.0

    #Coverage
    coverage = len(element_counts) / len(original_text_used)

    print(f"avg_jaccard : {avg_jaccard}")
    print(f"redundancy : {redundancy}")
    print(f"coverage : {coverage}")
    
        

    ########## Top activating sentences for each feature/concept in z_class  ##################
    
    # First, clean the activating sentences directory if it already exists
    activ_sentences_dir = os.path.join(dir_interpretability_results,f'activating_sentences')
    if os.path.exists(activ_sentences_dir):
        shutil.rmtree(activ_sentences_dir)

    activating_sentences_per_concept = {}
    top10_activating_sentences_per_concept = {}
    top10_activations_per_concept = {}
    activation_rate_concept = {}    
    top_activating_sentences_per_concept_for_tfidf = {}
   
    
    # For each feature in z_class
    for index_feature, feature in enumerate(selected_features):
        
        activations_inspected = concepts_activations[:,feature]
        
        # All activating sentences for this concept 
        activating_texts = [original_text_used[i] for i in activating_sentences_list[index_feature]]
        activating_sentences_per_concept[feature] = activating_texts
      
      
	    # Save the top 10 sentences within the 'activating' cluster associated with the highest activations of the concept
        _, top10_sentences = activations_inspected.topk(10)
        # We keep them if they are in the activating cluster
        saved_sentences_indices = list(set(top10_sentences.tolist()).intersection(activating_sentences_list[index_feature]))  
        # Activation strength of the feature on the sentence
        saved_sentences = [original_text_used[i] for i in saved_sentences_indices]
        top10_activating_sentences_per_concept[feature] = saved_sentences
        saved_activation_values = activations_inspected[saved_sentences_indices].tolist()
        top10_activations_per_concept[feature] = saved_activation_values

        # Save the top 100 sentences within the 'activating' cluster associated with the highest activations of the concept
        _, top100_sentences = activations_inspected.topk(100)
        # We keep them if they are in the activating cluster
        saved_sentences_indices = list(set(top100_sentences.tolist()).intersection(activating_sentences_list[index_feature]))  
        saved_sentences = [original_text_used[i] for i in saved_sentences_indices]
        top_activating_sentences_per_concept_for_tfidf[feature] = saved_sentences
        
        # Sparsity level of the feature 
        activation_rate = len(activating_texts) / (activations_inspected.shape[0])
        activation_rate_concept[feature] = activation_rate


  
    ##########  Generate one wordcloud per active concept in z_class for visualization (scores are based on TF-IDF) ##################
    
    # Clean/tokenize once
    clean_texts = prepare_clean_texts(original_text_used)

    # Build per-concept docs from selected sentences and compute TF-IDF
    #    (use the dict already built for TF-IDF input)
    concept_terms, concept_docs = build_concept_docs_and_tfidf(
        clean_texts=clean_texts,
        per_concept_sentences=top_activating_sentences_per_concept_for_tfidf,
        original_texts=original_text_used,
        top_k=20,
        min_df=2,
    )

    ## Save visualization for qualitative analysis of the learned concepts: Word clouds based on top n-grams in TF-IDF + Top 10 activating sentences for each concept
    common_words_dir = os.path.join(dir_interpretability_results,"top_ngrams_tf_idf")
    os.makedirs(common_words_dir, exist_ok=True)
    concepts_cards_dir = os.path.join(dir_interpretability_results,"concepts_definition")
    os.makedirs(concepts_cards_dir, exist_ok=True)
    
    # Clean the "interpretability" directories
    for entry in os.scandir(common_words_dir):
        (shutil.rmtree if entry.is_dir() else os.remove)(entry.path)
    for entry in os.scandir(concepts_cards_dir):
        (shutil.rmtree if entry.is_dir() else os.remove)(entry.path)


    # Create two "interpretability" files for each concept: 
    #   - one representing a wordcloud of the n-grams the most associated to this concept (based on TF-IDF scores)
    #   - one displaying the 10 sentences which activate the most the concept 
    for concept, keywords in concept_terms.items(): 
        
        ##### Save Wordcloud of n-grams
        content_ngrams_cloud = ""
        print(f"")
        category_concept = next((i for i, segment in enumerate(segmented_features) if concept in segment), None)
        ngrams_cloud_path = os.path.join(common_words_dir,f"Concept_{concept}_Category_{cfg_model.match_label_category[str(category_concept)]}")
        print(f"\nConcept: {concept}")
        for term, score in keywords:
            print(f"  {term}: {score}")
            content_ngrams_cloud += f" {term}:{score},"
        make_wordcloud(content_ngrams_cloud, save_basepath=ngrams_cloud_path)



        # Save a .txt file containing metrics for this concept along with the 10 sentences that most strongly activate its associated feature
        concept_file_path = os.path.join(concepts_cards_dir,f"Concept_{concept}_Category_{cfg_model.match_label_category[str(category_concept)]}.txt")
        top_activating_sentences = top10_activating_sentences_per_concept[concept]
        top_activations = top10_activations_per_concept[concept]

        # "interpretability" file for concept
        lines = []
        lines.append(f"Activation rate of the concept : {activation_rate_concept[concept]:.4f}")
        lines.append("")  
        lines.append(f"ConceptSim value for the concept : {concept_sim_dict[concept]:.4f}")
        lines.append("")  
        lines.append("Keywords associated to the concept:")
        lines.append("")  

        for term, score in keywords:
            lines.append(f"  {term}: {score}")

        lines.append("")  
        lines.append("Top activating sentences:")
        lines.append("")  
        lines.append("")  
        for text, value in zip(top_activating_sentences, top_activations):
            lines.append(f"{text}\t{value:.4f}")
            lines.append("")  

        # join into one big string
        content = "\n".join(lines)

        # write to disk
        with open(concept_file_path, "w", encoding="utf-8") as f:
            f.write(content)

       

    ################# Bonus : Compute Coherence Metrics on each concept based on the set of "activating sentences" ##############

    # Tokenization: Convert Sentences to Lists of Words
    list_split_words = [ filter_sentences([sentence]) for sentence in  original_text_used]

    # Build Dictionary 
    dictionary = Dictionary(list_split_words)
    dictionary.filter_extremes(no_below=5, no_above=0.8)  # Removes rare and common words
    dictionary.compactify()       

    concepts_words = []
    concepts_size = []
    for i, concept in enumerate(activating_sentences_per_concept.keys()):
        concept_words = filter_sentences(activating_sentences_per_concept[concept])
        concepts_words.append(concept_words)
        concepts_size.append(concept_size_cluster[concept])

    # Compute Coherence metrics 
    (cv_mean, uci_mean, npmi_mean),_ = calc_topic_coherence(concepts_words, list_split_words, dictionary,concepts_size,False)
    print(f"C_V: {cv_mean:.4f}, C_UCI: {uci_mean:.4f}, C_NPMI: {npmi_mean:.4f}")


  
    dict_interpretability_metrics = {"avg_jaccard" : avg_jaccard, "redundancy" : redundancy, "coverage" : coverage, "cv_mean" : cv_mean, "uci_mean" : uci_mean, "npmi_mean" : npmi_mean, "Averaged mean cosine similarity over the dataset" : mean_cos_sim ,"Weighted Averaged ConceptSim" : weighted_avg_concept_sim, "Unweighted Averaged ConceptSim" : unweighted_avg_concept_sim,'SentenceSim(k) for k from 1 to 4' : mean_cos_sim_by_number_share_concepts}


    file_to_save_intepretability_metrics = os.path.join(dir_interpretability_results,f"interpretability_metrics.json") 
    with open(file_to_save_intepretability_metrics, 'w') as file:
        json.dump(dict_interpretability_metrics, file, indent=4)


    print("######################################## END : Computation of Interpretability Metrics on z_class Concepts ########################################\n")
