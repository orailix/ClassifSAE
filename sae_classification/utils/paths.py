from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PATH_CONFIG=ROOT / "configs"
PATH_LOCAL_DATASET = ROOT / "local_datasets"
PATH_CHECKPOINTS = ROOT / "finetuned_models"
PATH_MODEL_METRICS = ROOT / "results/model_metrics"
PATH_SAE_METRICS = ROOT / "results/sae_metrics"
PATH_CACHED_ACT = ROOT / "cached_activations_for_sae"
PATH_CHECKPOINT_SAE = ROOT / "trained_sae"
PATH_BASELINE_METHODS = ROOT / "baseline_methods"
PATH_BASELINE_METHODS_METRICS = ROOT / "results/baseline_metrics" 
PATH_CONCEPTS_METRICS = ROOT / "results" / "concepts_metrics"
PATH_DATASET_ACTIVATIONS = ROOT / "datasets_activations_evaluation"
PATH_POST_ANALYSIS_ACTIVATIONS = ROOT / "post_analysis_activations"