from ..utils import LLMTrainerConfig, LLMLoadConfig
from transformer_lens import HookedTransformer
from loguru import logger
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM,AutoTokenizer, AutoModelForSequenceClassification
from typing import Union


def _load_model(cls, path, **kwargs):
    # try local-only first
    try:
        return cls.from_pretrained(path, local_files_only=True, **kwargs)
    except Exception:
        # fallback: allow download (and cache it for next time)
        return cls.from_pretrained(path, local_files_only=False, **kwargs)

def get_model(
    cfg: Union[LLMTrainerConfig,LLMLoadConfig],
    decoder: bool = True,
    nb_classes: int = -1
    ) -> PreTrainedModel:

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = cfg.model_path_pre_trained if cfg.model_path_pre_trained == cfg.model_path else cfg.model_path
    
    if path == cfg.model_path_pre_trained:
        logger.info(f"Loading pre-trained LLM classifier from {cfg.model_path_pre_trained}")
    else:
        logger.info(f"Loading fine-tuned LLM classifier from {cfg.model_path}") 
    
    if decoder:        
        model = _load_model(AutoModelForCausalLM,path)  
    else:
        model = _load_model(AutoModelForSequenceClassification,path,num_labels=nb_classes)

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
            )
    
    hook_model.tokenizer = tokenizer

    print(f"tokenizer.padding_side : {tokenizer.padding_side}")


    return hook_model