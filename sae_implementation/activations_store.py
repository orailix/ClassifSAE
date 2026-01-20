# =============================================================================
# This file is adapted from:
#   SAELens (v 3.13.0) (https://github.com/jbloomAus/SAELens/blob/v3.13.0/sae_lens/training/activations_store.py)
#   License: MIT (see https://github.com/orailix/ClassifSAE/blob/main/SAELens_License/LICENSE)
#
#
# NOTES:
#   • We adapted the caching activation procedure in `get_buffer` so that we only cache a single hidden state per sentence. 
#     For a decoder-only model, it corresponds to the hidden state associated with the token preceding the classification label. 
#   • We enabled joint caching of the LLM-predicted label for each sentence by concatenating the predicted label index with the cached activation ('self.save_label' option)
#
# =============================================================================


import contextlib
import os
from typing import Any, Iterator, Literal, cast
import numpy as np
import torch
from datasets import concatenate_datasets, DatasetDict
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule
from transformers import DataCollatorForLanguageModeling
from loguru import logger

from .config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)



class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    cached_activations_path: str | None
    tokens_column: Literal["input_ids"]
    hook_name: str
    hook_layer: int
    _dataloader: Iterator[Any] | None = None
    _storage_buffer: torch.Tensor | None = None
    device: torch.device

    @classmethod
    def from_config(
        cls,
        cfg: LanguageModelSAERunnerConfig | CacheActivationsRunnerConfig,
        model: HookedRootModule | None = None,
        dataset_to_cache : HfDataset | None = None
    ) -> "ActivationsStore":
        cached_activations_path = cfg.cached_activations_path
        # set cached_activations_path to None if we're not using cached activations
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None


        return cls(
            model=model,
            dataset=dataset_to_cache,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=cfg.normalize_activations,
            device=torch.device(cfg.act_store_device),
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            prompt_embeddings_path=cfg.prompt_embeddings_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            eos=cfg.eos,
            prompt_tuning=cfg.prompt_tuning,
            save_label=cfg.save_label
        )


    def __init__(
        self,
        model: HookedRootModule | None,
        dataset: HfDataset | None,
        hook_name: str,
        hook_layer: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        cached_activations_path: str | None = None,
        prompt_embeddings_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        autocast_lm: bool = False,
        eos: bool = False,
        save_label : bool = False,
        prompt_tuning: bool = False
    ):
        self.model = model
       
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.dataset = dataset
        
        if isinstance(self.dataset, DatasetDict):
            if "train" in self.dataset:
                self.dataset = self.dataset["train"]
            else:
                raise ValueError(
                "DatasetDict provided but no 'train' split found. "
                "Pass a specific split (e.g., dataset['validation'])."
                )

        self.hook_name = hook_name
        self.hook_layer = hook_layer
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.half_buffer_size = n_batches_in_buffer // 2
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.cached_activations_path = cached_activations_path
        self.prompt_embeddings_path = prompt_embeddings_path
        self.autocast_lm = autocast_lm
        self.eos = eos
        self.prompt_tuning = prompt_tuning
        self.save_label = save_label

        # Global index into dataset to know where to resume when filling a new buffer.
        self.dataset_idx_last_token = 0 

        if self.model is not None :
            # We only cache the hidden state associated with the token preceding the class-generating token
            # emitted by the decoder-only model for classification.
            self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.model.tokenizer,mlm=False)

        self.n_dataset_processed = 0
        self.estimated_norm_scaling_factor = 1.0

        if self.dataset is not None:
            # Verify dataset is tokenized and offers 'input_ids'
            if len(self.dataset) == 0:
                raise ValueError("Dataset is empty.")
            sample = self.dataset[0]
            if "input_ids" in sample:
                self.is_dataset_tokenized = True
                self.tokens_column = "input_ids"
            else:
                raise ValueError("Dataset must provide 'input_ids' (tokenized) column.")

            if self.is_dataset_tokenized and hasattr(self.dataset, "set_format"):
                self.dataset.set_format(type="torch", columns=[self.tokens_column])
            else:
                print(
                    "Error: Dataset is not tokenized. This is not expected"
                )
            
            

        self.check_cached_activations_against_config()

   
  
    def check_cached_activations_against_config(self):

        if self.cached_activations_path is not None:  
            assert os.path.exists(
                self.cached_activations_path
            ), f"Cache directory {self.cached_activations_path} does not exist. Check dataset/model/hook names."

            self.next_cache_idx = 0  # which file to open next
            self.next_idx_within_buffer = 0  # where to start reading from in that file

          

    def apply_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return activations * self.estimated_norm_scaling_factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        return activations / self.estimated_norm_scaling_factor

    def get_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return (self.d_in**0.5) / activations.norm(dim=-1).mean()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):

        norms_per_batch = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            acts = self.next_batch(return_label_if_any=False)
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(self.d_in) / mean_norm

        return scaling_factor

    def reset_input_dataset(self):
        """
        Resets the input dataset iterator to the beginning.
        """
        self.dataset_idx_last_token = 0

    @property
    def storage_buffer(self) -> torch.Tensor:
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.half_buffer_size)

        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader

    
    
    @torch.no_grad()
    def get_buffer(
        self, n_batches_in_buffer: int
    ) -> torch.Tensor:
        """
        Loads the next n_batches_in_buffer batches of activations into a tensor and returns half of it.
        """
        if self.dataset is None and self.cached_activations_path is None:
            raise ValueError("Either dataset or cached_activations_path must be provided.")

        batch_size = self.store_batch_size_prompts
        d_in = self.d_in
        buffer_size = batch_size * n_batches_in_buffer
        num_layers = 1
        

        # If activations are cached on disk and we want to fetch them for SAE training:
        if self.cached_activations_path is not None:
                
            # Initialize an empty tensor.
            # There is a dimension for the number of cached layers, currently we only cache for a single layer.
            # Add an extra channel to concatenate the label predicted by the model with its vector representation of the sentence at the studied layer.
            expected_last_dim = d_in + 1 if self.save_label else d_in
            new_buffer = torch.zeros(
                (buffer_size, num_layers, expected_last_dim),
                dtype=self.dtype,
                device=self.device,
            )
            n_tokens_filled = 0

            while n_tokens_filled < buffer_size:

                # For an ActivationStore object, next_cache_idx keeps in memory which buffer files have not been provided yet in the training. 
                # When they have been provided all at least once, it resets the index and the cycle begins again.
                # It is useful in practice if the number of training steps is greater than the number of cached states. 
                cache_file = os.path.join(self.cached_activations_path, f"{self.next_cache_idx}.safetensors")

                # If the expected cache file does not exist
                if not os.path.exists(cache_file):
                    if self.next_cache_idx == 0:
                        raise ValueError(f"The activation directory {self.cached_activations_path} is empty")
                    else:
                        # Reset to beginning; we will repeat cached activations as needed
                        self.next_cache_idx = 0
                        self.next_idx_within_buffer = 0
                        cache_file = os.path.join(self.cached_activations_path, f"{self.next_cache_idx}.safetensors")
                    
                full = self.load_buffer(cache_file)

                # Validate last dimension matches configuration
                if full.size(-1) != expected_last_dim:
                    raise ValueError(
                        f"Cached buffer last dim {full.size(-1)} does not match expected {expected_last_dim}. "
                        f"Ensure caches were generated with save_label={self.save_label} and d_in={self.d_in}."
                    )

                file_len = full.shape[0]
                start = self.next_idx_within_buffer
                
                if start >= file_len:
                    # At EOF for this file; move to next and continue
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0
                    continue

                remaining_in_file = file_len - start
                remaining_in_buffer = buffer_size - n_tokens_filled
                take = min(remaining_in_file, remaining_in_buffer)

                chunk = full[start : start + take, ...]
                new_buffer[n_tokens_filled : n_tokens_filled + take, ...] = chunk
                
                # Advance pointers
                n_tokens_filled += take
                self.next_idx_within_buffer = start + take
                
                if self.next_idx_within_buffer >= file_len:
                    # Move to next file and reset within-file index
                    self.next_cache_idx += 1
                    self.next_idx_within_buffer = 0

            return new_buffer


        
        # In the context of caching new hidden states for the first time, we start here.
        
        # Initialize an empty tensor of the maximum required size.
        # There is a dimension for the number of cached layers (currently 1)
        # There is a dimension for the "context" (currently 1 since we only cache the sentence-level token embedding per sentence).
        context_size=1
        new_buffer = torch.zeros(
            (buffer_size, context_size, num_layers, d_in),
            dtype=self.dtype, 
            device=self.device,
        )

        # If we save the labels,  initialize a buffer to store them.
        buffer_labels = None
        if self.save_label:
            buffer_labels = torch.zeros((buffer_size),dtype=self.dtype,device=self.device)

        '''
        We check the remaining sentences of the dataset can match buffer_size, which might not be the case if we reach the end of the dataset.
        We want all our buffers to be of same size for simplicity, therefore in the event when the buffer size provided does not divide the length of the sentences dataset, we fill the last buffer with duplicate hidden states of sentences. 
        This is not ideal, preferably the size of the buffer specified shall divide the length of the dataset so that we only cache once different samples.
        '''
        remaining_prompts = len(self.dataset) - self.dataset_idx_last_token
        if remaining_prompts > buffer_size :
            selected_dataset = self.dataset.select(
                list(range(self.dataset_idx_last_token, self.dataset_idx_last_token + buffer_size))
            )
            self.dataset_idx_last_token += buffer_size
        else: 
            # Last buffer: concatenate tail of dataset with random samples to complete the buffer
            end_of_dataset = self.dataset.select(
                list(range(self.dataset_idx_last_token, len(self.dataset)))
            )
            # Calculate the number of additional samples needed to complete the buffer size
            n_elements_to_complete = buffer_size - remaining_prompts
            # Select random indices from the full dataset (with replacement if necessary)
            rand_indices = np.random.choice(len(self.dataset), n_elements_to_complete, replace=True)
            # Create the completion_dataset by selecting the randomly chosen indices
            completion_dataset = self.dataset.select(rand_indices)
            selected_dataset = concatenate_datasets([end_of_dataset,completion_dataset])
        
        selected_dataset.cleanup_cache_files()
        selected_dataset = selected_dataset.with_format("python")

        # Create DataLoader over selected sentences
        dataloader_sentence_hidden_states = DataLoader(selected_dataset, batch_size=self.store_batch_size_prompts, collate_fn=self.data_collator)

        # Load prompt-tuning embeddings (optional)
        if self.prompt_embeddings_path is not None:
            try:
                prompt_embeddings = torch.load(self.prompt_embeddings_path, map_location=self.model.cfg.device) 
                logger.info(
                    "Prompt embeddings loaded from {}. Using them for activations caching.",
                    self.prompt_embeddings_path,
                )
            except (FileNotFoundError, IOError):
                prompt_embeddings = None
                logger.warning(
                    "Cannot open {}. Skipping prompt embeddings for activations caching.",
                    self.prompt_embeddings_path,
                )
        else:
            prompt_embeddings = None

        self.model.eval()

        with torch.no_grad():
            
            for num_batch, batch in enumerate(dataloader_sentence_hidden_states):

                batch_tokens = batch[self.tokens_column].to(self.model.cfg.device)
                attention_mask = batch['attention_mask'].to(self.model.cfg.device)
                predicted_labels = batch['predicted_labels'].to(self.model.cfg.device)

                # Setup autocast if using
                if self.autocast_lm:
                    autocast_if_enabled = torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=self.autocast_lm,
                    )
                else:
                    autocast_if_enabled = contextlib.nullcontext()

                with autocast_if_enabled:
                    
                    # If prompt-tuning was done to align the model with the classification task, we
                    # add the prompt embeddings at the beginning of the tokenized sentence via a hook.
                    if prompt_embeddings is not None:
                        vtok = prompt_embeddings.size(0)
                        emb = self.model.embed(batch_tokens)
                        bs = batch_tokens.size(0)
                        if (emb.shape[1]+vtok) > self.model.cfg.n_ctx:
                            # Trim to fit within context window when prompt tokens are added.
                            emb = emb[:,-(self.model.cfg.n_ctx-vtok):] 
                            attention_mask = attention_mask[:,-(self.model.cfg.n_ctx-vtok):]
                       
                        P = prompt_embeddings.unsqueeze(0).expand(bs, -1, -1)
                        new_emb = torch.cat([P, emb], dim=1) 

                        def override_embed(resid_pre, hook):        # resid_pre is (B, L, d_model)
                            return new_emb                       

                        B, L, _ = new_emb.shape         # (batch, seq_len, d_model)
                        dummy_tokens  = torch.zeros(B, L, dtype=torch.long, device=self.model.cfg.device)
                        prefix_mask = torch.ones(bs, vtok, device=self.model.cfg.device).long()
                        attention_mask   = torch.cat([prefix_mask, attention_mask], dim=1) 

                    
                        # We directly add the prompt embeddings at the beginning of the tokenized sentence after the encoding matrix through a hook
                        with self.model.hooks(
                                fwd_hooks=[("hook_embed", override_embed)]
                        ):
                            layerwise_activations = self.model.run_with_cache(
                                dummy_tokens,
                                attention_mask=attention_mask,
                                names_filter=[self.hook_name],
                                stop_at_layer=self.hook_layer + 1,
                                prepend_bos=False,
                                **self.model_kwargs
                            )[1]

                
                        
                    else:
   
                        # Run inference on the LLM classifier while caching the inspected layer.
                        layerwise_activations = self.model.run_with_cache(
                            batch_tokens,
                            attention_mask=attention_mask,
                            names_filter=[self.hook_name],
                            stop_at_layer=self.hook_layer + 1,
                            prepend_bos=self.prepend_bos,
                            **self.model_kwargs,
                        )[1]
                
                if self.hook_name not in layerwise_activations:
                    raise KeyError(f"Hook {self.hook_name} not found in cached activations")
                
                n_prompts, _ = batch_tokens.shape
        
                # Allocate per-batch buffer on correct device/dtype
                stacked_activations = torch.zeros(
                    (n_prompts, context_size, 1, self.d_in),
                    dtype=self.dtype,
                    device=self.device,
                )
                
                # It assumes that the activation does not come from a head dimension                
                # Pick the activation of the token preceding the class token.
                # Compute the last valid token and account for eos if configured.
                picked = layerwise_activations[self.hook_name][:,(-2-int(self.eos)),:].unsqueeze(1) # (B, 1, d_in)
                stacked_activations[:, :, 0] = picked.to(stacked_activations.dtype)
                
                new_buffer[
                    num_batch*batch_size : num_batch*batch_size + stacked_activations.shape[0], ...
                ] = stacked_activations

                # Save labels for the classifier component of ClassifSAE
                if self.save_label:
                    buffer_labels[num_batch*batch_size : num_batch*batch_size + stacked_activations.shape[0]] = predicted_labels.to(buffer_labels.dtype)
        

        # Keep the dimensions regarding the buffer size,  number of layers (1) and the dimension of the residual stream
        new_buffer = new_buffer.reshape(-1, num_layers, d_in)

        #  No normalization during caching except when explicitly requested.
        if self.normalize_activations == "expected_average_only_in":
            new_buffer = self.apply_norm_scaling_factor(new_buffer)
                    
        if self.save_label:
            # Keep labels in same dtype as activations to allow concatenation.
            buffer_labels = buffer_labels.to(dtype=self.dtype).unsqueeze(1).unsqueeze(2)
            # Concatenate the predicted label with the corresponding cached sentence representation
            new_buffer = torch.cat((new_buffer, buffer_labels), dim=2)
            
        # Shuffle buffer with index tensor on the same device for portability (CPU/GPU)
        perm = torch.randperm(new_buffer.shape[0], device=new_buffer.device)
        new_buffer = new_buffer[perm]

        return new_buffer

    def save_buffer(self, buffer: torch.Tensor, path: str):
        """
        Used by cached activations runner to save a buffer to disk.
        For reuse by later workflows.
        """
        save_file({"activations": buffer}, path)

    def load_buffer(self, path: str) -> torch.Tensor:

        with safe_open(path, framework="pt", device=str(self.device)) as f: 
            buffer = f.get_tensor("activations")
        return buffer

    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.train_batch_size_tokens
    
        try:
            new_samples = self.get_buffer(
                self.half_buffer_size
            )
        except StopIteration:
            print(
                "Warning: All samples in the training dataset have been exhausted, we are now beginning a new epoch with the same samples."
            )
            self._storage_buffer = (
                None  # dump the current buffer so samples do not leak between epochs
            )
            try:
                new_samples = self.get_buffer(self.half_buffer_size)
            except StopIteration:
                raise ValueError(
                    "We were unable to fill up the buffer directly after starting a new epoch. This could indicate that there are less samples in the dataset than are required to fill up the buffer. Consider reducing batch_size or n_batches_in_buffer. "
                )

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [new_samples, self.storage_buffer],
            dim=0,
        )

        # Shuffle once here (no need to reshuffle in the DataLoader)
        perm = torch.randperm(mixing_buffer.shape[0], device=mixing_buffer.device)
        mixing_buffer = mixing_buffer[perm]

        # 2.  put 50 % in storage
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        dataloader = iter(
            DataLoader(
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=False,
            )
        )

        return dataloader

    def next_batch(self,return_label_if_any=False):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        Set return_label_if_any=True to keep the concatenated class labels (if present).
        """
        try:
            # Try to get the next batch
            next_batch = next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            next_batch = next(self.dataloader)

        if self.save_label and not return_label_if_any:
            next_batch = next_batch[:,:,:-1] #remove the labels
        return next_batch

    def state_dict(self) -> dict[str, torch.Tensor]:
        result = {
            "n_dataset_processed": torch.tensor(self.n_dataset_processed),
        }
        if self._storage_buffer is not None:  # first time might be None
            result["storage_buffer"] = self._storage_buffer
        return result

    def save(self, file_path: str):
        save_file(self.state_dict(), file_path)

    