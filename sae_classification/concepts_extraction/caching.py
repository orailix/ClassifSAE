from pathlib import Path
import math
import time
import os
import numpy as np
import torch
import contextlib
from loguru import logger
from transformers import PreTrainedModel, DataCollatorWithPadding
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

from ..utils import LLMLoadConfig
from ..llm_classifier_tuning import (
    process_dataset,
    get_hook_model,
    get_model,
    _init_tokenizer,
    _max_seq_length
)

from sae_implementation import (
    CacheActivationsRunnerConfig,
    SentenceCacheActivationsRunner,
)



def adjust_training_tokens(cfg: LLMLoadConfig, dataset_len: int) -> None:

    """Cap training_tokens so we only cache unique sentences"""
    tokens = cfg.task_args["training_tokens"]
    if tokens > dataset_len:
        logger.warning(f"training_tokens ({tokens}) > dataset size ({dataset_len}) – clamping.")
        cfg.task_args["training_tokens"] = dataset_len

        dir_name = Path(cfg.task_args["new_cached_activations_path"]).parent
        cfg.task_args["new_cached_activations_path"] = str(dir_name / str(dataset_len))


def load_predictions(cfg: LLMLoadConfig) -> torch.Tensor:
    pred_path = (
        Path(cfg.dir_to_save_metrics)
        / cfg.split
        / "predicted_labels"
        / f"{cfg.model_name}_{cfg.dataset_name}.pt"
    )
    if not pred_path.exists():
        raise FileNotFoundError(
            f"File {pred_path} does not exist. Make sure to run 'eval-model' before caching the activations of the model so that it also includes the associated predicted labels"
        )
    logger.info(f"Loaded predicted labels from {pred_path}")
    return torch.load(pred_path, map_location="cpu", weights_only=True)  


def max_seq_length_hook(hook_model):
    return hook_model.cfg.n_ctx



def get_buffer(dataset_tokenized: Dataset, dataset_idx_last_token:int, n_batches_in_buffer: int, store_batch_size_prompts: int, d_in: int, save_label: bool, model: PreTrainedModel, hook_layer:int, max_ctx: int,data_collator ):

    # Initialize an empty tensor of the maximum required size.
    # There is a dimension for the number of cached layers (currently 1), and a dimension for the context (1, as we cache one token per sentence).
    context_size=1
    num_layers=1
    batch_size = store_batch_size_prompts
    buffer_size = batch_size * n_batches_in_buffer
    new_buffer = torch.zeros(
        (buffer_size, context_size, num_layers, d_in)
    )

    # If save the labels, initialize an empty tensor to store them
    buffer_labels = None
    if save_label:
        buffer_labels = torch.zeros((buffer_size),dtype=new_buffer.dtype)

    '''
    We check the remaining sentences of the dataset can match buffer_size, which might not be the case if we reach the end of the dataset.
    We want all our buffers to be of same size for simplicity, therefore in the event when the buffer size provided does not divide the length of the sentences dataset, we fill the last buffer with duplicate hidden states of sentences. 
    This is not ideal, preferably the size of the buffer specified shall divide the length of the dataset so that we only cache once different samples.
    '''
    
    remaining_prompts = len(dataset_tokenized) - dataset_idx_last_token
    if remaining_prompts > buffer_size :
        selected_dataset = dataset_tokenized.select(
            list(range(dataset_idx_last_token, dataset_idx_last_token + buffer_size))
        )
        dataset_idx_last_token += buffer_size
    else: 
        # Last buffer: concatenate tail with random samples to complete buffer size
        end_of_dataset = dataset_tokenized.select(
            list(range(dataset_idx_last_token, len(dataset_tokenized)))
        )
        # Calculate the number of additional samples needed to complete the buffer size
        n_elements_to_complete = buffer_size - remaining_prompts
        # Select random indices from the full dataset (with replacement if necessary)
        rand_indices = np.random.choice(len(dataset_tokenized), n_elements_to_complete, replace=True)
        # Create the completion_dataset by selecting the randomly chosen indices
        completion_dataset = dataset_tokenized.select(rand_indices)
        selected_dataset = concatenate_datasets([end_of_dataset,completion_dataset])
    
    selected_dataset.cleanup_cache_files()
    selected_dataset = selected_dataset.with_format("python")

    model.eval()

    dataloader_sentence_hidden_states = DataLoader(
        selected_dataset, batch_size=store_batch_size_prompts, collate_fn=data_collator
    )

    with torch.no_grad():
            
        for num_batch, batch in enumerate(dataloader_sentence_hidden_states):

            batch_tokens = batch["input_ids"].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            predicted_labels = batch['predicted_labels'].to(model.device)

           
            autocast_if_enabled = contextlib.nullcontext()

            with autocast_if_enabled:

                # Ensure sequences fit the model context window
                if batch_tokens.shape[1] > max_ctx:
                    attention_mask = attention_mask[:, :max_ctx]
                    batch_tokens = batch_tokens[:, :max_ctx]

                out = model(
                    input_ids=batch_tokens,
                    attention_mask=attention_mask,
                    output_hidden_states=True,        
                    return_dict=True
                )

                layerwise_activations = out.hidden_states[hook_layer]

            n_prompts, _ = batch_tokens.shape
    
            # Allocate per-batch tensor on model/device dtype to avoid silent casts
            stacked_activations = torch.zeros((n_prompts, context_size, num_layers, d_in),
                                              dtype=layerwise_activations.dtype,
                                              device=layerwise_activations.device,
            )
        
            # It assumes that the activation comes from the residual stream
            # Get the embeddings of the [CLS] token
            stacked_activations[:, :, 0] = layerwise_activations[:,0,:].unsqueeze(1)

            new_buffer[
                num_batch*batch_size : num_batch*batch_size + stacked_activations.shape[0], ...
            ] = stacked_activations.cpu()

            # Save labels if requested; keep dtype consistent with new_buffer
            if save_label  and buffer_labels is not None:
                buffer_labels[
                    num_batch*batch_size : num_batch*batch_size + stacked_activations.shape[0]
                ] = predicted_labels.cpu()
    

    # Keep buffer_size x num_layers x d_in
    new_buffer = new_buffer.reshape(-1, num_layers, d_in)
                
    if save_label  and buffer_labels is not None:

        buffer_labels = buffer_labels.to(dtype=new_buffer.dtype).unsqueeze(1).unsqueeze(2)

        # Concatenate the predicted label with the corresponding cached sentence representation
        new_buffer = torch.cat((new_buffer, buffer_labels), dim=2)
    
    # Shuffle
    perm = torch.randperm(new_buffer.shape[0], device=new_buffer.device)
    new_buffer = new_buffer[perm]


    return new_buffer, dataset_idx_last_token


def cache_activations_encoder(cfg,tokenizer):
    
    # model & datasets
    num_classes = len(cfg.match_label_category)
    model = get_model(cfg, False, num_classes)
    max_ctx = _max_seq_length(model)
    
    split = cfg.split
    ds = process_dataset(cfg, split, tokenizer, False, False, max_ctx)
    adjust_training_tokens(cfg, len(ds))

    preds = load_predictions(cfg)
    if len(ds) != preds.shape[0]:
        raise ValueError(
            f"Dataset length ({len(ds)}) is not equal to predictions length ({preds.shape[0]})."
        )
    ds = ds.add_column("predicted_labels", preds.tolist())

    
    buffer_size = cfg.task_args['n_batches_in_buffer'] * cfg.task_args['store_batch_size_prompts']
    n_buffers = math.ceil(cfg.task_args['training_tokens'] / buffer_size)
    total = buffer_size * n_buffers
    logger.info(
        f"Caching {total} activations ({cfg.task_args['training_tokens']} unique). Buffer size = {buffer_size}"
    )

    start_time = time.time()

    data_collator = DataCollatorWithPadding(tokenizer)

    # Global index to track where we are in the dataset when filling a new buffer
    dataset_idx_last_token = 0 

    # Ensure output directory exists and is empty
    new_cached_activations_path = cfg.task_args['new_cached_activations_path']
    assert new_cached_activations_path is not None
    if os.path.exists(new_cached_activations_path):
        if len(os.listdir(new_cached_activations_path)) > 0:
            raise Exception(
                f"Activations directory ({new_cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )
    else:
        os.makedirs(new_cached_activations_path)

    # Cache one activation per sentence
    tokens_per_buffer = cfg.task_args["store_batch_size_prompts"] * cfg.task_args["n_batches_in_buffer"]


    for i in tqdm(range(n_buffers), desc="Caching activations"):
        try:

            buffer, dataset_idx_last_token = get_buffer(dataset_tokenized=ds,
                                                        dataset_idx_last_token=dataset_idx_last_token,
                                                        n_batches_in_buffer=cfg.task_args['n_batches_in_buffer'],
                                                        store_batch_size_prompts=cfg.task_args['store_batch_size_prompts'],
                                                        d_in=cfg.task_args['d_in'],
                                                        save_label=cfg.task_args['save_label'],
                                                        model=model, 
                                                        hook_layer=cfg.task_args['hook_layer'],
                                                        max_ctx=max_ctx,
                                                        data_collator=data_collator)
            
            save_file({"activations": buffer},  os.path.join(new_cached_activations_path, f"{i}.safetensors"))

            del buffer

        except StopIteration:
                logger.warning(
                    f"Warning: Ran out of samples while filling the buffer at batch {i} before reaching {n_buffers} batches. No more caching will occur."
                )
                break

    end_time = time.time()
    elapsed = end_time - start_time

    logger.success(
        f"Caching complete in {elapsed:.2f} seconds ({(cfg.task_args['training_tokens'] / (elapsed * 1e6)):.2f} M tokens/s)",
    )



def cache_activations_decoder(cfg,tokenizer):


    model = get_hook_model(cfg, tokenizer)  # returns a HookedTransformer
    max_ctx = max_seq_length_hook(model)
    

    # dataset 
    ds = process_dataset(cfg, split=cfg.split, tokenizer=tokenizer, add_template=True, decoder=True, max_len=max_ctx)
    adjust_training_tokens(cfg, len(ds))

    preds = load_predictions(cfg)
    if len(ds) != preds.shape[0]:
        raise ValueError(
            f"Dataset length ({len(ds)}) is not equal to predictions length ({preds.shape[0]})."
        )
    ds = ds.add_column("predicted_labels", preds.tolist())

    # runner setup
    cache_cfg = CacheActivationsRunnerConfig(**cfg.task_args)
    
    buffer_size = cfg.task_args['n_batches_in_buffer']*cfg.task_args['store_batch_size_prompts']
    total = buffer_size * math.ceil(cfg.task_args['training_tokens']/ buffer_size)
    logger.info(
        f"Caching {total} activations ({cfg.task_args['training_tokens']} unique). Buffer size = {buffer_size}"
    ) 

    # Ensure output directory exists and is empty
    new_cached_activations_path = cfg.task_args['new_cached_activations_path']
    assert new_cached_activations_path is not None
    if os.path.exists(new_cached_activations_path):
        if len(os.listdir(new_cached_activations_path)) > 0:
            raise Exception(
                f"Activations directory ({new_cached_activations_path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
            )
    else:
        os.makedirs(new_cached_activations_path)

    runner = SentenceCacheActivationsRunner(cache_cfg, model, ds)
    logger.info(f"Runner initialised:\n{runner}")


    # run
    start_time = time.time()
    runner.run()
    end_time = time.time()
    elapsed = end_time - start_time

    logger.success(
        f"Caching complete in {elapsed:.1f} seconds ({(cfg.task_args['training_tokens'] / (elapsed * 1e6)):.2f} M tokens/s)",
    )



# Call the caching procedure of the LLM classifier activations implemented in `sae_implementation`
def cache_activations(config: LLMLoadConfig) -> None: 

    cfg = LLMLoadConfig.autoconfig(config)
    tokenizer, decoder = _init_tokenizer(cfg)

    if decoder:
        cache_activations_decoder(cfg,tokenizer)
    else:
        cache_activations_encoder(cfg,tokenizer)

