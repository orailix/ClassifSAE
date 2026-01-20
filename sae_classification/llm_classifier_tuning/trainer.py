from collections import Counter
from typing import Dict, Optional, Sequence
import os

import numpy as np
import torch
from datasets import Dataset
from loguru import logger
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from ..utils import LLMTrainerConfig



def _class_weights(label_counts: Dict[int, int], device: torch.device) -> torch.Tensor:
    """Return inverse‑frequency weights.

    *N* is the total number of samples, *C* the number of classes, *n_i* the
    samples of class *i*.

    ``w_i = N / (C · n_i)``
    """
    classes = sorted(label_counts)  # sort by category number
    counts = torch.tensor([label_counts[c] for c in classes], dtype=torch.float)
    weights = counts.sum() / (len(classes) * counts)
    return weights.to(device)


def _labels_to_single_token_ids(
    tokenizer: AutoTokenizer,
    labels: Sequence[int],
) -> torch.Tensor:
    """
    Return a tensor of token IDs such that each label l in `labels` is represented
    by exactly one tokenizer ID. Labels whose string form does not map to a single
    token are skipped.

    This is more robust than checking tokenizer.get_vocab() membership, which
    may not reflect actual tokenization behavior for some vocabularies.
    """
    keep_ids = []
    for lbl in sorted(set(labels)):
        tokenized = tokenizer(str(lbl), add_special_tokens=False)["input_ids"]
        if len(tokenized) == 1:
            keep_ids.append(tokenized[0])
        else:
            logger.warning(
                f"Label {lbl} does not tokenize to a single token; skipping for decoder classification."
            )
    if not keep_ids:
        raise ValueError("No labels tokenize to a single token. Provide label words that are single tokens.")
    return torch.tensor(keep_ids, dtype=torch.long)

def loss_last_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eos: bool = False,
    class_weights: Optional[torch.Tensor] = None,
    keep_token_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    """
        Decoder-only loss: predict the final class token.

        - Assumes inputs/labels are language modeling style, where `labels` equals
        input_ids (for MLM=False).
        - We compute loss on the penultimate position (-2 - eos), predicting the
        last actual token (-1 - eos), where `eos=True` indicates an EOS token
        is present at the end.

        If `keep_token_ids` is provided, restrict logits to these vocab IDs and
        map the ground-truth token IDs to class indices [0..K-1] accordingly.
    """
    
    vocab_size = logits.size(-1)
    target_index = -2 - int(eos)

    # shape: (B, V)
    pred_logits = logits[:, target_index].contiguous()
    
    # Ground-truth token IDs at the next position
    true_labels = labels[:, (target_index+1)]

    if keep_token_ids is not None:
        
        # Ensure device match
        keep_token_ids = keep_token_ids.to(pred_logits.device)

        # Restrict logits to the candidate class token IDs
        pred_logits = pred_logits.index_select(dim=1, index=keep_token_ids)

        # Build a vectorized mapping from token_id -> index in [0..K-1]
        token_to_idx = torch.full(
            (vocab_size,), -1, dtype=torch.long, device=pred_logits.device
        )
        token_to_idx[keep_token_ids] = torch.arange(
            keep_token_ids.numel(), device=pred_logits.device
        )
        class_targets = token_to_idx[true_labels]
        if (class_targets < 0).any():
            raise ValueError(
                "Ground-truth tokens not in keep_token_ids; check label-token mapping."
            )
        true_targets = class_targets
    else:
        # Unrestricted vocabulary (rare path; typically use keep_token_ids)
        true_targets = true_labels

    if class_weights is not None and class_weights.device != pred_logits.device:
        class_weights = class_weights.to(pred_logits.device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    return loss_fn(pred_logits, true_targets)


# -----------------------------------------------------------------------------
# Custom Trainer classes
# -----------------------------------------------------------------------------

class _BaseTrainer(Trainer):
    """Common for both encoder and decoder trainers."""

    def __init__(
        self,
        *args,
        eos: bool = False,
        tokenizer = None,
        class_weights: Optional[torch.Tensor] = None,
        keep_token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._eos = eos
        self._class_weights = class_weights
        self._keep_token_ids = keep_token_ids
        self._tokenizer = tokenizer


class DecoderTrainer(_BaseTrainer):
    """Compute loss on the classifier token for decoder models."""

    def compute_loss(self, model, inputs, return_outputs=False): 
        outputs = model(**inputs,use_cache=False)
        loss = loss_last_token(
            outputs.logits,
            inputs.labels,
            eos=self._eos,
            class_weights=self._class_weights,
            keep_token_ids=self._keep_token_ids
        )
        return (loss, outputs) if return_outputs else loss


class EncoderTrainer(_BaseTrainer):
    """Standard CE loss on logits with optional balancing."""

    def compute_loss(self, model, inputs, return_outputs=False):  
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        loss_fn = torch.nn.CrossEntropyLoss(weight=self._class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss



def get_trainer(
    cfg: LLMTrainerConfig,
    model: PreTrainedModel,
    train_ds: Dataset,
    eval_ds: Dataset,
    tokenizer: AutoTokenizer,
    decoder: bool,
    imbalanced_classes: bool = False,
) -> Trainer:
    

    if cfg.training_args.get("report_to") == "wandb":
        os.environ.update(
            {
                "WANDB_MODE": "offline",
                "WANDB_PROJECT": f"{cfg.model_name}_ft_{cfg.dataset_name}",
                "WANDB_RUN_NAME": f"{cfg.model_name}_{cfg.dataset_name}_{cfg.training_args['max_steps']}",
            }
        )

    # class count
    label_col = "label"
    class_weights: Optional[torch.Tensor] = None
    keep_token_ids: Optional[torch.Tensor] = None

    if imbalanced_classes:
        label_array = np.asarray(train_ds[label_col])
        label_counts = dict(Counter(label_array))
        class_weights = _class_weights(label_counts, model.device)

    # trainer for decoder‑only architecture
    if decoder:
        # Build a robust keep_token_ids by actual tokenization (single-token labels only)
        unique_labels = sorted(set(train_ds[label_col]))
        keep_token_ids = _labels_to_single_token_ids(tokenizer, unique_labels)
        
        # For LM collator, labels come from input_ids; remove the scalar label column
        train_ds = train_ds.remove_columns("label")
        eval_ds = eval_ds.remove_columns("label")


    # collator & training arguments
    collator = (
        DataCollatorForLanguageModeling(tokenizer, mlm=False)
        if decoder
        else DataCollatorWithPadding(tokenizer)
    )

    training_args = TrainingArguments(**cfg.training_args)
    model.gradient_checkpointing_enable()
    logger.info(f"Training set size: {len(train_ds)}")

    # trainer
    trainer_cls = DecoderTrainer if decoder else EncoderTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        eos=cfg.eos,
        class_weights=class_weights,
        keep_token_ids=keep_token_ids,
    )

    return trainer
