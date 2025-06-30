from typing import Optional, List
import typer
from loguru import logger
#from .end_signal import GotEndSignal
from .model_training import fine_tuning_model, main_evaluation, train_classifier
from .sae_training import activation_caching, sae_trainer, main_sae_evaluation, main_sae_evaluation_restrict ,dr_fit, dr_methods_investigation, concept_shap_train, selection_sae_features, concept_analysis, selection_features, concepts_evaluation
from .sae_application import scoring_neurons, distillation_training, sae_vis_features, NIG_features,activations_features

app = typer.Typer()

#We want either to finetune/train the model on the classification task, finetune/train the SAE on a specific layer, work on the SAE features once the SAE is trained

@app.command()
def train_model(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    fine_tuning_model(config)

@app.command()
def eval_model(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    main_evaluation(config)

@app.command()
def train_layer_classifier(
    config_model: Optional[str] = None,
    config_classifier: Optional[str] = None
):
    
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_classifier is None:
        raise ValueError(f"You should pass a SAE config with the --config_classifier option.")
  
    train_classifier(config_model,config_classifier)
    

@app.command()
def caching_activation_model(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    activation_caching(config) 

@app.command()
def train_sae(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    sae_trainer(config)

@app.command()
def eval_sae(
    config_model: Optional[str] = None,
    config_sae: Optional[str] = None
    ):
    
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_sae is None:
        raise ValueError(f"You should pass a SAE config with the --config_sae option.")

    main_sae_evaluation(config_model,config_sae)

@app.command()
def eval_sae_restrict(
    config_model: Optional[str] = None,
    config_sae: Optional[str] = None
):
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_sae is None:
        raise ValueError(f"You should pass a SAE config with the --config_sae option.")

    main_sae_evaluation_restrict(config_model,config_sae)

@app.command()
def select_sae_features(
    config_model: Optional[str] = None,
    config_sae: Optional[str] = None
    ):
    
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_sae is None:
        raise ValueError(f"You should pass a SAE config with the --config_sae option.")

    selection_sae_features(config_model,config_sae)

@app.command()
def select_features(
    config_model: Optional[str] = None,
    config_concept: Optional[str] = None
    ):
    
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_concept is None:
        raise ValueError(f"You should pass a Evaluation config with the --config_concept option.")

    selection_features(config_model,config_concept)

@app.command()
def apply_dr_methods(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    dr_fit(config)

@app.command()
def concept_shap_training(config_dr: Optional[str] = None,
                          config_model : Optional[str] = None):
    if config_dr is None:
        raise ValueError(f"You should pass a dimension reduction method config with the --config_dr option.")

    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")

    concept_shap_train(config_dr,config_model)


@app.command()
def evaluate_dr_methods(config_dr: Optional[str] = None,
                        config_model : Optional[str] = None):
    if config_dr is None:
        raise ValueError(f"You should pass a dimension reduction method config with the --config_dr option.")

    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")

    dr_methods_investigation(config_dr,config_model)


@app.command()
def evaluate_concepts(config_concept: Optional[str] = None,
                    config_model : Optional[str] = None):

    if config_concept is None:
        raise ValueError(f"You should pass an analysis method config with the --config_concept option.")

    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")
        
    concepts_evaluation(config_concept,config_model)

@app.command()
def analyse_concepts(config_concept: Optional[str] = None,
                    config_model : Optional[str] = None):

    if config_concept is None:
        raise ValueError(f"You should pass an analysis method config with the --config_analysis option.")

    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")
        
    concept_analysis(config_concept,config_model)


@app.command()
def best_unique_neurons(config: Optional[str] = None):
    if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
    scoring_neurons(config)
    
@app.command()
def selection_small_model_neurons(config: Optional[str] = None):
     if config is None:
        raise ValueError(f"You should pass a config with the --config option.")
     distillation_training(config)


@app.command()
def visualize_features(
    config_vis : Optional[str] = None,
    config_model: Optional[str] = None,
    config_sae: Optional[str] = None
):
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_sae is None:
        raise ValueError(f"You should pass a SAE config with the --config_sae option.")
    
    if config_vis is None:
        raise ValueError(f"You should pass a Vis config with the --config_vis option.")

    
    sae_vis_features(config_vis,config_model,config_sae)


@app.command()
def visu_activations_features(
    config_vis : Optional[str] = None,
    config_model: Optional[str] = None,
    config_sae: Optional[str] = None
):
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_sae is None:
        raise ValueError(f"You should pass a SAE config with the --config_sae option.")
    
    if config_vis is None:
        raise ValueError(f"You should pass a Vis config with the --config_vis option.")

    
    activations_features(config_vis,config_model,config_sae)


@app.command()
def integrated_gradient_features(
    config_vis : Optional[str] = None,
    config_model: Optional[str] = None,
    config_sae: Optional[str] = None
):
    if config_model is None:
        raise ValueError(f"You should pass a model config with the --config_model option.")   
    
    if config_sae is None:
        raise ValueError(f"You should pass a SAE config with the --config_sae option.")
    
    if config_vis is None:
        raise ValueError(f"You should pass a Vis config with the --config_vis option.")

    
    NIG_features(config_vis,config_model,config_sae)


if __name__ == "__main__":
    app()