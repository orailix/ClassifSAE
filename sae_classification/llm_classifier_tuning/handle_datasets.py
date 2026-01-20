from typing import Any, Dict, Tuple, Union
import re

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from loguru import logger
from transformers import AutoTokenizer


TEMPLATE = "{example}{statement}\n\n{options}\n"


def build_prompt_template(cfg, statement: str) -> str:  
    """Return the classification prompt built from cfg and statement."""  
    example = "".join(cfg.example) if getattr(cfg, "add_example", False) else ""
    options = "\n".join([
        "OPTIONS:",
        *[f"{lbl}({cat})" for lbl, cat in cfg.match_label_category.items()],
    ])
    return TEMPLATE.format(example=example, statement=statement, options=options)


def _format_sample(sample: Dict[str, Any], cfg) -> Dict[str, Any]: 
    
    prompt = build_prompt_template(cfg, sample["text"])
    return {"prompt": prompt, "label": sample["label"]}


def _append_label(sample: Dict[str, Any]) -> Dict[str, Any]:
    # Append the ground‑truth label to the end of the prompt (decoder style).
    return {
        "prompt": f"{sample['prompt']}{sample['label']}",
        "label": sample["label"],
    }


def _tokenize_sample(
    batch: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_len: int = 512,
    decoder: bool = True,
) -> Dict[str, Any]:
    """Tokenize batch for either decoder or encoder models."""
    encodings = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    encodings["label"] = batch["label"]
    return encodings


def _select_columns(ds: Dataset, decoder: bool = True) -> Dataset:
    cols = ["input_ids", "attention_mask","label"]
    if "token_type_ids" in ds.column_names:
        cols.append("token_type_ids")
    return ds.select_columns(cols)



def count_template_units(tokenizer: AutoTokenizer, cfg) -> Tuple[int, int, int]:
    """Count the number of (characters, words, tokens) present in the prompt template. It can vary depending on the classification task we aim to solve"""
    prompt = build_prompt_template(cfg, statement="")
    char_count = len(prompt)
    word_count = len([w for w in re.split(r"[ \n]+", prompt) if w])
    token_count = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return char_count, word_count, token_count


def _load_dataset_split(cfg, split: str) -> Dataset:  

    if cfg.dataset_in_local:
        
        try:
            ds = load_from_disk(cfg.dataset_path)
        except Exception:
            
            # Special case for the dataset tweeteval
            if "tweet_eval" in cfg.official_name_repo:
                ds_name, subset = cfg.official_name_repo.split("-", 1)
                raw_ds = load_dataset(ds_name, subset)
                raw_ds = raw_ds.remove_columns(
                    [c for c in raw_ds["train"].column_names if c not in ("text", "label")]
                )
                merged_test = concatenate_datasets([raw_ds["test"], raw_ds["validation"]])
                ds = DatasetDict(train=raw_ds["train"], test=merged_test)
            else:
                ds = load_dataset(cfg.official_name_repo)

            ds.save_to_disk(cfg.dataset_path)
    
    else:
        
        # Special case for the dataset tweeteval
        if "tweet_eval" in cfg.official_name_repo:
            ds_name, subset = cfg.official_name_repo.split("-", 1)
            raw_ds = load_dataset(ds_name, subset)
            raw_ds = raw_ds.remove_columns(
                [c for c in raw_ds["train"].column_names if c not in ("text", "label")]
            )
            merged_test = concatenate_datasets([raw_ds["test"], raw_ds["validation"]])
            ds = DatasetDict(train=raw_ds["train"], test=merged_test)
        else:
            ds = load_dataset(cfg.official_name_repo)
        
    
    return ds[split]


def process_dataset(  
    cfg: Union["LLMTrainerConfig", "LLMLoadConfig"],
    split: str,
    tokenizer: AutoTokenizer,
    add_template: bool = True,
    decoder: bool = True,
    max_len: int = 512,
) -> Dataset:

    if split not in {"train", "test", "unsupervised"}:
        raise ValueError('`split` must be "train", "test" or "unsupervised".')

    ds = _load_dataset_split(cfg, split)

    # If we want to add the classification template format
    if add_template:
        ds = ds.map(lambda s: _format_sample(s, cfg), remove_columns=ds.column_names)
        ds = ds.map(_append_label)
    elif "text" in ds.column_names:
        ds = ds.rename_column("text", "prompt")

    ds = ds.map(
        lambda b: _tokenize_sample(b, tokenizer, max_len, decoder=decoder),
        batched=True,
        load_from_cache_file=False,
    )

    logger.info(f"Tokenization completed for {split} split")
    return _select_columns(ds, decoder=decoder)
