from pathlib import Path
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import DataCollatorForLanguageModeling

from ..utils import LLMTrainerConfig
from .trainer import loss_last_token
from .main_training import set_seed


def prompt_tuning(
    cfg: LLMTrainerConfig,
    model: torch.nn.Module,
    tokenizer,
    train_ds,
    device,
    max_ctx: int,
    eos:bool,
    save_dir: str | Path,
    file_name: str,
    vtok: int,
    total_steps: int,
    seed: int = 42,
    batch_size: int = 2,
    lr_max: float = 5e-3,
    lr_min: float = 1e-4,
    keep_token_ids: torch.Tensor | None = None,
) -> Path:
    
    """
        Run soft‑prompt tuning and return the saved embeddings path.

        - Optimizes a learnable prompt of length `vtok` prepended to inputs.
        - Loss computed on the final class token via loss_last_token.
        - If `keep_token_ids` is provided (decoder classification), restricts loss to class tokens.
    """

    set_seed(seed)

    model.eval()  
    model.requires_grad_(False)

    prompt = nn.Embedding(vtok, model.config.hidden_size).to(device)

    optimizer = torch.optim.AdamW(prompt.parameters(), lr=lr_max)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=lr_min / lr_max,
        total_iters=total_steps,
    )

    # dataloader
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    loader_iter = iter(loader)
    
    if keep_token_ids is not None:
        keep_token_ids = keep_token_ids.to(device)


    # training
    logger.info(f"Starting soft prompt‑tuning for {cfg.model_name} – {total_steps} steps")
    pbar = tqdm(total=total_steps, unit="step", desc="Prompt‑tuning")

    for step in range(total_steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        inputs = batch["input_ids"].to(device)
        attn   = batch["attention_mask"].to(device)
                       
        # If virtual tokens exceed context, trim right‑most of the base sequence
        if (inputs.size(1)+vtok) > max_ctx:
            trim = max_ctx - vtok
            inputs = inputs[:, -trim:]         # keep the right-most tokens
            attn   =   attn[:, -trim:]

        bs = inputs.size(0)
        emb_backbone = model.get_input_embeddings()(inputs)
        emb_prompt   = prompt.weight.unsqueeze(0).expand(bs, -1, -1)
        embeddings   = torch.cat([emb_prompt, emb_backbone], dim=1)

        prefix_mask = torch.ones(bs, vtok, dtype=torch.long, device=device)
        attn        = torch.cat([prefix_mask, attn], dim=1)

        outputs = model(inputs_embeds=embeddings, attention_mask=attn, use_cache=False)
        logits  = outputs.logits

        # labels == inputs for LM-style loss
        loss = loss_last_token(
            logits,
            inputs, 
            eos=eos,
            keep_token_ids=keep_token_ids
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.update(1)
        pbar.set_postfix(step=step + 1, loss=loss.item())

    pbar.close()
    logger.info(f"Prompt‑tuning finished – final loss: {loss.item():.4f}")

    # ------------------------------------------------------------- save prompt
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file_name
    torch.save(prompt.weight.cpu(), save_path)
    logger.info("Saved prompt embeddings → %s", save_path)

    return save_path