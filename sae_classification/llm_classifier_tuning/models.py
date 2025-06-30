from ..utils import LLMTrainerConfig, LLMLoadConfig
from transformer_lens import HookedTransformer
from loguru import logger
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM,AutoTokenizer
from typing import Union


def get_model(
    cfg: Union[LLMTrainerConfig,LLMLoadConfig]
    ) -> PreTrainedModel:

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # The current writing is redundant but it enables the possibility to include more complex loading of models (quantized,...) 
    if cfg.model_path_pre_trained==cfg.model_path: # If you want only to load the pre-trained model, either just to evaluate it or to tune it

        logger.info(f"Loading pre-trained LLM classifier from {cfg.model_path_pre_trained}")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path_pre_trained)
    
    else:
        logger.info(f"Loading fine-tuned LLM classifier from {cfg.model_path}") 
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

    
    model.to(device)  
        
    return model

def get_hook_model(
    cfg :  Union[LLMLoadConfig],
    tokenizer: AutoTokenizer
    ) -> HookedTransformer:

    model = get_model(cfg)
   
    hook_model = HookedTransformer.from_pretrained_no_processing(
            cfg.model_architecture,
            hf_model=model,
            tokenizer=tokenizer,
        )

    hook_model.tokenizer = tokenizer


    return hook_model