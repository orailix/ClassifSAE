from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
)

from ..utils import LLMLoadConfig
from .handle_datasets import process_dataset
from .models import get_model
from .prompt_tuning import prompt_tuning
from .main_training import _init_tokenizer, _max_seq_length


def _build_keep_ids_and_mapping(tokenizer, labels: List[int]) -> tuple[torch.Tensor, List[int]]:
    """
    Build:
      - keep_ids: tensor of vocab IDs corresponding to labels that tokenize to a single token
      - id_to_label: list mapping from index in restricted logits [0..K-1] to the original class label
    """
    keep_ids: List[int] = []
    id_to_label: List[int] = []
    for lbl in sorted(set(labels)):
        tokenized = tokenizer(str(lbl), add_special_tokens=False)["input_ids"][-1]
        keep_ids.append(tokenized)
        id_to_label.append(lbl)
    if not keep_ids:
        raise ValueError("No labels tokenize to a single token. Provide label words that are single tokens.")
    return torch.tensor(keep_ids, dtype=torch.long), id_to_label


def _collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    decoder: bool,
    keep_ids: torch.Tensor | None,
    id_to_label: List[int] | None,
    max_ctx: int,
    prompt_embeddings: torch.Tensor | None,
    device,
    eos: bool,
) -> Tuple[List[int], List[int]]:
    """Run the model and return flattened (y_true, y_pred) lists."""

    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []


    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):

            inputs = batch["input_ids"].to(device)
            attn   = batch["attention_mask"].to(device)


            # optional prompt‑tuning embedding prepend
            if decoder and prompt_embeddings is not None:
                bs, vtok = inputs.size(0), prompt_embeddings.size(0)
           
                if (inputs.size(1)+vtok) > max_ctx:
                    inputs = inputs[:, -(max_ctx-vtok):]         # keep the right-most tokens
                    attn   =   attn[:, -(max_ctx-vtok):]

                emb_b = model.get_input_embeddings()(inputs)
                emb_p = prompt_embeddings.to(device).unsqueeze(0).expand(bs, -1, -1)
                inputs_embeds = torch.cat([emb_p, emb_b], dim=1)
                attn = torch.cat([torch.ones(bs, vtok, device=device).long(), attn], dim=1)
                outputs = model(inputs_embeds=inputs_embeds, attention_mask=attn)
            else:
                outputs = model(input_ids=inputs, attention_mask=attn)

            logits = outputs.logits

            if decoder:
                # take penultimate token (−2 − eos)
                idx = -2 - int(eos)
                logits = logits[:, idx]
                if keep_ids is not None:
                    keep_ids = keep_ids.to(logits.device)
                    logits = logits.index_select(dim=1, index=keep_ids)
                    pred_idx = logits.argmax(dim=-1).tolist()
                    # Map restricted index [0..K-1] back to class label
                    preds = [id_to_label[i] for i in pred_idx]  # type: ignore[index]
                else:
                    # Unrestricted (not recommended): argmax over full vocab
                    preds = logits.argmax(dim=-1).tolist()
                label_key = "label"
            else:
                preds = logits.argmax(dim=-1).tolist()
                label_key = "labels"
        
            y_true.extend(batch[label_key].tolist())
            y_pred.extend(preds)

    return y_true, y_pred


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]: 
    """Return accuracy, prec, rec, f1 per class + global accuracy."""
    counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    match_counts = Counter(t for t, p in zip(y_true, y_pred) if t == p)

    metrics: Dict[str, float] = {}
    total_correct = sum(match_counts.values())
    metrics["accuracy"] = total_correct / len(y_true)

    for cls in sorted(counts):
        recall = match_counts[cls] / counts[cls] if counts[cls] else 0.0
        precision = (
            match_counts[cls] / pred_counts[cls] if pred_counts[cls] else 0.0
        )
        f1 = (
            2 * recall * precision / (recall + precision)
            if recall + precision
            else 0.0
        )
        metrics.update(
            {
                f"recall_{cls}": recall,
                f"precision_{cls}": precision,
                f"f1_{cls}": f1,
            }
        )

    logger.info(f"\nAverage accuracy of the model on the evaluated dataset of {len(y_true)} samples : {metrics['accuracy']}\n")

    return metrics


# Prompt-tuning for LLM backbones if we don't fine-tune them directly in order to align them with the classification task
def load_or_create_prompt(
    cfg: LLMLoadConfig,
    model,
    tokenizer,
    train_ds,
    device,
    max_ctx: int,
    keep_token_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if not (cfg.prompt_tuning):
        return None

    vtok = cfg.prompt_tuning_params["vtok"]
    steps = cfg.prompt_tuning_params["total_steps"]
    eos = cfg.task_args['eos']

    save_dir = Path(cfg.prompt_embeddings_dir) / cfg.model_name / cfg.dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"embeddings_prompt_{vtok}_{steps}.pt"
    save_path = save_dir / file_name

    if not save_path.exists():
        logger.info("Prompt embeddings not found – running prompt‑tuning …")
        prompt_tuning(
            cfg,
            model,
            tokenizer,
            train_ds,
            max_ctx=max_ctx,
            eos=eos,
            device=device,
            save_dir=save_dir,
            file_name=file_name,
            vtok=vtok,
            total_steps=steps,
            keep_token_ids=keep_token_ids,
        )
    else:
        logger.info(f"Loaded existing prompt embeddings → {save_path}")

    return torch.load(save_path)




def main_evaluation(config: LLMLoadConfig) -> None:
    """Evaluate a pre‑trained / fine‑tuned model on a classification dataset."""

    cfg = LLMLoadConfig.autoconfig(config)
    tokenizer, decoder = _init_tokenizer(cfg)

    # model & datasets
    num_classes = len(cfg.match_label_category)
    model = get_model(cfg, decoder, num_classes)
    max_ctx = _max_seq_length(model)
    device = model.device

    add_template = decoder
    split = cfg.split
    dataset = process_dataset(cfg, split, tokenizer, add_template, decoder, max_ctx)

    # optionally load train set for prompt‑tuning
    train_ds = (
        process_dataset(cfg, "train", tokenizer, add_template, decoder, max_ctx)
        if decoder and cfg.prompt_tuning
        else None
    )


    # Collator and decoder label restriction
    collator: DataCollatorForLanguageModeling | DataCollatorWithPadding
    if decoder:
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        unique_labels = list(set(dataset["label"]))
        keep_ids, id_to_label = _build_keep_ids_and_mapping(tokenizer, unique_labels)
    else:
        collator = DataCollatorWithPadding(tokenizer)
        keep_ids = None
        id_to_label = None

    prompt_emb = (
        load_or_create_prompt(cfg, model, tokenizer, train_ds, device, max_ctx, keep_token_ids=keep_ids) if decoder else None
    )

    eos = cfg.task_args['eos']
    batch_size = cfg.task_args['batch_size']
    proportion_to_evaluate = cfg.task_args['proportion_to_evaluate']

    # Number of sentences to select among the evaluation dataset
    n_selected = int(proportion_to_evaluate * len(dataset))
    evaluated_dataset = dataset.select(range(n_selected))

    dataloader = DataLoader(evaluated_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    y_true, y_pred = _collect_predictions(
        model,
        dataloader,
        decoder=decoder,
        keep_ids=keep_ids,
        id_to_label=id_to_label,
        max_ctx=max_ctx,
        prompt_embeddings=prompt_emb,
        device=device,
        eos=eos,
    )

    metrics = compute_metrics(y_true, y_pred)

    # Save metrics and the labels predicted by the classifier LLM
    out_dir = Path(cfg.dir_to_save_metrics) / split
    pred_dir = out_dir / "predicted_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / f"{cfg.model_name}_{cfg.dataset_name}.json").write_text(
        json.dumps(metrics, indent=4)
    )
    torch.save(torch.tensor(y_pred), pred_dir / f"{cfg.model_name}_{cfg.dataset_name}.pt")

    logger.info(f"Saved metrics and predictions to {out_dir}")
