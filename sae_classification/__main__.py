from typing import Optional
import typer
from loguru import logger
from .llm_classifier_tuning import fine_tuning_model, main_evaluation
from .concepts_extraction import activation_caching, sae_trainer, baseline_concept_method_train, selection_segmentation_concepts, concepts_evaluation, concept_interpretability

app = typer.Typer()


@app.command()
def tune_llm_classifier(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    fine_tuning_model(config)


@app.command()
def eval_classifier(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    main_evaluation(config)


@app.command()
def save_activations_classifier(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    activation_caching(config)

@app.command()
def train_sae(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    sae_trainer(config)

@app.command()
def train_baseline(config_baseline: Optional[str] = None,
                   config_model : Optional[str] = None):
    
    if config_baseline is None:
        raise ValueError(f"You should pass a baseline concept-based method config with the --config-baseline option.")


    baseline_concept_method_train(config_baseline,config_model)

@app.command()
def post_process_concepts(config_concept: Optional[str] = None,
                          config_classifier: Optional[str] = None):
    
    if config_concept is None:
        raise ValueError(f"You should pass a concept-based method config with the --config-concept option.")

    if config_classifier is None:
        raise ValueError(f"You should pass a llm classifier config with the --config-classifier option.")

    
    selection_segmentation_concepts(config_concept, config_classifier)


@app.command()
def evaluate_concepts(config_concept: Optional[str] = None,
                          config_classifier: Optional[str] = None):
    
    if config_concept is None:
        raise ValueError(f"You should pass a concept-based method config with the --config-concept option.")

    if config_classifier is None:
        raise ValueError(f"You should pass a llm classifier config with the --config-classifier option.")

    
    concepts_evaluation(config_concept, config_classifier)


@app.command()
def interpret_concepts(config_concept: Optional[str] = None,
                       config_classifier: Optional[str] = None):
    
    if config_concept is None:
        raise ValueError(f"You should pass a concept-based method config with the --config-concept option.")

    if config_classifier is None:
        raise ValueError(f"You should pass a llm classifier config with the --config-classifier option.")

    
    concept_interpretability(config_concept, config_classifier)

if __name__ == "__main__":
    app()