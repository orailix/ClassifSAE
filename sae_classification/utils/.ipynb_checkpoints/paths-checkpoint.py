from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PATH_CONFIG=ROOT / "configs"
PATH_LOCAL_MODEL = ROOT / "models/local_models"
PATH_LOCAL_TOKENIZER = ROOT / "models/local_tokenizers"
PATH_LOCAL_DATASET = ROOT / "data_sae/raw"
PATH_LOCAL_DATASET_TOKENIZED = ROOT / "data_sae/tokenized"
PATH_CHECKPOINTS = ROOT / "finetuned_models"
PATH_MODEL_METRICS = ROOT / "results/model_metrics"
PATH_SAE_METRICS = ROOT / "results/sae_metrics"
PATH_CACHED_ACT = ROOT / "cached_activations"
PATH_CHECKPOINT_SAE = ROOT / "trained_sae"
PATH_SAE_ACT = ROOT / "results/sae_activations"
PATH_SAE_TOP_LOGITS = ROOT / "results/top_logits" 
PATH_NEURONS_SELECTION_METRICS = ROOT / "results/neurons_selection_metrics"
PATH_DR_METHODS = ROOT / "dr_methods"
PATH_INTERP_METHODS_METRICS = ROOT / "results" / "interpretability_methods_metrics"
PATH_INTERP_METHODS_ACTIVATIONS = ROOT / "results" / "interpretability_methods_activations"
PATH_SAVE_VIS = ROOT / 'features_interfaces'
PATH_ACTS_LABELS = ROOT / 'activations_with_label'
#PATH_SELECT_SAE_FEATURES = ROOT / "results/select_sae_features"
PATH_POST_FEATURES = ROOT / "post_processing_features"