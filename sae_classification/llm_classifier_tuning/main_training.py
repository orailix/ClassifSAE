import gc
import os
import shutil
from typing import Tuple, Union

import numpy as np
import torch
import random
from loguru import logger
from transformers import AutoTokenizer
import wandb


from ..utils import LLMTrainerConfig, LLMLoadConfig
from .models import get_model
from .handle_datasets import process_dataset
from .trainer import get_trainer


def set_seed(seed: int = 42, hf: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # deterministic kernels 
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")  # set once

    # Hugging Face 
    if hf:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)          # also sets tf/torch/Np & can force determinism


def _load_tokenizer(path: str, **kwargs):
    # try local-only first
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True, **kwargs)
    except Exception:
        # fallback: allow download (and cache it for next time)
        return AutoTokenizer.from_pretrained(path, local_files_only=False, **kwargs)


def _init_tokenizer(cfg: Union[LLMTrainerConfig,LLMLoadConfig]) -> Tuple[AutoTokenizer, bool]:
    
    """
    Initialize tokenizer and infer decoder-only vs encoder-only from model_name.
    For decoder models, we pad/truncate on the left to preserve the rightmost tokens.
    """
    
    tok = _load_tokenizer(cfg.tokenizer_path)

    if "bert" in cfg.model_name.lower():
        tok.pad_token = tok.pad_token or "[PAD]"
        tok.padding_side = tok.truncation_side = "right"
        decoder = False
    else:
        tok.pad_token = tok.eos_token or "</s>"
        tok.padding_side = tok.truncation_side = "left"
        decoder = True

    logger.debug("Tokenizer initialised")
    if decoder:
        logger.info("The loaded model has a decoder-only architecture")
    else:
        logger.info("The loaded model has a encoder-only architecture")

    return tok, decoder


def _max_seq_length(model) -> int:
    """Infer maximum context length from HF model config fields."""

    for key in ("max_position_embeddings", "n_positions", "n_ctx"):
        if hasattr(model.config, key):
            return getattr(model.config, key)

    raise AttributeError("Could not determine model context length.")


def _dataset_stats(ds, split: str) -> None:
    
    lengths = [len(seq) for seq in ds["input_ids"]]
    logger.info(
        f"[{split}] max tokens: {max(lengths)} | mean tokens: {float(np.mean(lengths)):.3f}"
    )


def _rename_wandb_offline_run(run_id: str, run_name: str, root: str = "wandb") -> None:
  
    wandb_root = "wandb"
    offline_folders = os.listdir(wandb_root)

    for folder in offline_folders:
        if folder.endswith(run_id):
            original_path = os.path.join(wandb_root, folder)
            new_path = os.path.join(wandb_root, run_name)
            shutil.move(original_path, new_path)
            break      


def fine_tuning_model(config: LLMTrainerConfig) -> None:
    
    """Classification fine‑tuning according to config on a backbone LLM"""

    set_seed(seed=42, hf=True)

    cfg = LLMTrainerConfig.autoconfig(config)

    # Tokenizer + Model
    tokenizer, decoder = _init_tokenizer(cfg)
    nb_classes = len(cfg.match_label_category)
    model = get_model(cfg, decoder, nb_classes)
    model.config.pad_token_id = tokenizer.pad_token_id
    max_ctx = _max_seq_length(model)
    logger.info(f"Max context length: {max_ctx} tokens")

    # datasets
    add_template = decoder  # template prompt only for generative models
    train_ds = process_dataset(cfg, "train", tokenizer, add_template, decoder, max_ctx)
    test_ds = process_dataset(cfg, "test", tokenizer, add_template, decoder, max_ctx)

    _dataset_stats(train_ds, "train")
    _dataset_stats(test_ds, "test")

    # trainer
    trainer = get_trainer(cfg, model, train_ds, test_ds, tokenizer, decoder, cfg.imbalanced_classes)
    logger.info("Starting training…")
    trainer.train()

    # WandB rename (offline runs)
    if 'report_to' in cfg.training_args and (cfg.training_args['report_to']=='wandb'):
        # Rename the wandb folder of the run
        run = wandb.run  
        run_id = run.id
        run_name = f"{cfg.model_name}_fine_tuning_{cfg.dataset_name}_{cfg.training_args['max_steps']}"

        wandb.finish()

        _rename_wandb_offline_run(run_id, run_name, root="wandb")
       
    del model 
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Fine‑tuning complete")
