import argparse
import re
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Mapping

import einops
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from .sae import SAE

from .activations_store import ActivationsStore


# Everything by default is false so the user can just set the ones they want to true
@dataclass
class EvalConfig:
    batch_size_prompts: int | None = None

    # Reconstruction metrics
    n_eval_reconstruction_batches: int = 10
    compute_kl: bool = False
    compute_ce_loss: bool = False

    # Sparsity and variance metrics
    n_eval_sparsity_variance_batches: int = 1
    compute_l2_norms: bool = False
    compute_sparsity_metrics: bool = False
    compute_variance_metrics: bool = False


def get_eval_everything_config(
    batch_size_prompts: int | None = None,
    n_eval_reconstruction_batches: int = 10,
    n_eval_sparsity_variance_batches: int = 1,
) -> EvalConfig:
    """
    Returns an EvalConfig object with all metrics set to True, so that when passed to run_evals all available metrics will be run.
    """
    return EvalConfig(
        batch_size_prompts=batch_size_prompts,
        n_eval_reconstruction_batches=n_eval_reconstruction_batches,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
    )


@torch.no_grad()
def run_evals(
    sae: SAE,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    eval_config: EvalConfig = EvalConfig(),
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:

    hook_name = sae.cfg.hook_name
    actual_batch_size = (
        eval_config.batch_size_prompts or activation_store.store_batch_size_prompts
    )

    metrics = {}

    if eval_config.compute_kl or eval_config.compute_ce_loss:
        assert eval_config.n_eval_reconstruction_batches > 0
        metrics |= get_downstream_reconstruction_metrics(
            sae,
            model,
            activation_store,
            compute_kl=eval_config.compute_kl,
            compute_ce_loss=eval_config.compute_ce_loss,
            n_batches=eval_config.n_eval_reconstruction_batches,
            eval_batch_size_prompts=actual_batch_size,
        )

        activation_store.reset_input_dataset()

    if (
        eval_config.compute_l2_norms
        or eval_config.compute_sparsity_metrics
        or eval_config.compute_variance_metrics
    ):
        assert eval_config.n_eval_sparsity_variance_batches > 0
        metrics |= get_sparsity_and_variance_metrics(
            sae,
            model,
            activation_store,
            compute_l2_norms=eval_config.compute_l2_norms,
            compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
            compute_variance_metrics=eval_config.compute_variance_metrics,
            n_batches=eval_config.n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=actual_batch_size,
            model_kwargs=model_kwargs,
        )

    if len(metrics) == 0:
        raise ValueError(
            "No metrics were computed, please set at least one metric to True."
        )

    total_tokens_evaluated = (
        activation_store.context_size
        * eval_config.n_eval_reconstruction_batches
        * actual_batch_size
    )
    metrics["metrics/total_tokens_evaluated"] = total_tokens_evaluated

    return metrics


def get_downstream_reconstruction_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    n_batches: int,
    eval_batch_size_prompts: int,
):
    metrics_dict = {}
    if compute_kl:
        metrics_dict["kl_div_with_sae"] = []
        metrics_dict["kl_div_with_ablation"] = []
    if compute_ce_loss:
        metrics_dict["ce_loss_with_sae"] = []
        metrics_dict["ce_loss_without_sae"] = []
        metrics_dict["ce_loss_with_ablation"] = []

    
    for _ in range(n_batches):
        #Customization so that we select 'eval_batch_size_prompts' sentences at each eval batch
        sentence_tokens_to_evaluate = activation_store.dataset.shuffle(seed=random.randint(0, 10000)).select(range(eval_batch_size_prompts)) #The dataset is already tokenized
        #batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        batch_tokens_list = sentence_tokens_to_evaluate[activation_store.tokens_column]
        
        max_length = max(tensor.size(0) for tensor in batch_tokens_list)
        padded_tensors = []
        for tensor in batch_tokens_list:
            padding_length = max_length - tensor.size(0)
            padded_tensor = torch.cat([torch.full((padding_length,), activation_store.model.tokenizer.pad_token_id), tensor], dim=0)
            padded_tensors.append(padded_tensor)

        batch_tokens = torch.stack(padded_tensors, dim=0)

        #Manually fixing the issue of sentence longer than the context windows size since it is not automatically handled by transformer-lens and it causes conflict with the positional embedding that it dones on a vector of size the context attention windows
        if batch_tokens.shape[1] > model.cfg.n_ctx:
            batch_tokens = batch_tokens[:,-model.cfg.n_ctx:] #It should not be an issue since we want to predict for almost the last token, so it looses almost nothing of the context it could have seen otherwise
           
        
        for metric_name, metric_value in get_recons_loss(
            sae,
            model,
            batch_tokens,
            activation_store,
            compute_kl=compute_kl,
            compute_ce_loss=compute_ce_loss,
        ).items():
            metrics_dict[metric_name].append(metric_value)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metrics_dict.items():
        metrics[f"metrics/{metric_name}"] = torch.stack(metric_values).mean().item()

    if compute_kl:
        metrics["metrics/kl_div_score"] = (
            metrics["metrics/kl_div_with_ablation"] - metrics["metrics/kl_div_with_sae"]
        ) / metrics["metrics/kl_div_with_ablation"]

    if compute_ce_loss:
        metrics["metrics/ce_loss_score"] = (
            metrics["metrics/ce_loss_with_ablation"]
            - metrics["metrics/ce_loss_with_sae"]
        ) / (
            metrics["metrics/ce_loss_with_ablation"]
            - metrics["metrics/ce_loss_without_sae"]
        )

    return metrics


def get_sparsity_and_variance_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int,
    compute_l2_norms: bool,
    compute_sparsity_metrics: bool,
    compute_variance_metrics: bool,
    eval_batch_size_prompts: int,
    model_kwargs: Mapping[str, Any],
):

    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    metric_dict = {}
    if compute_l2_norms:
        metric_dict["l2_norm_in"] = []
        metric_dict["l2_norm_out"] = []
        metric_dict["l2_ratio"] = []
    if compute_sparsity_metrics:
        metric_dict["l0"] = []
        metric_dict["l1"] = []
    if compute_variance_metrics:
        metric_dict["explained_variance"] = []
        metric_dict["mse"] = []

    for _ in range(n_batches):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

        # get cache
        _, cache = model.run_with_cache(
            batch_tokens,
            prepend_bos=False,
            names_filter=[hook_name],
            **model_kwargs,
        )

        # we would include hook z, except that we now have base SAE's
        # which will do their own reshaping for hook z.
        has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
        if any(substring in hook_name for substring in has_head_dim_key_substrings):
            original_act = cache[hook_name].flatten(-2, -1)
        else:
            original_act = cache[hook_name]

        # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
        if activation_store.normalize_activations == "expected_average_only_in":
            original_act = activation_store.apply_norm_scaling_factor(original_act)

        # send the (maybe normalised) activations into the SAE
        sae_feature_activations = sae.encode(original_act.to(sae.device))
        sae_out = sae.decode(sae_feature_activations).to(original_act.device)
        del cache

        if activation_store.normalize_activations == "expected_average_only_in":
            sae_out = activation_store.unscale(sae_out)

        flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
        flattened_sae_feature_acts = einops.rearrange(
            sae_feature_activations, "b ctx d -> (b ctx) d"
        )
        flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

        if compute_l2_norms:
            l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
            l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
            l2_norm_in_for_div = l2_norm_in.clone()
            l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
            l2_norm_ratio = l2_norm_out / l2_norm_in_for_div
            metric_dict["l2_norm_in"].append(l2_norm_in)
            metric_dict["l2_norm_out"].append(l2_norm_out)
            metric_dict["l2_ratio"].append(l2_norm_ratio)

        if compute_sparsity_metrics:
            l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
            l1 = flattened_sae_feature_acts.sum(dim=-1)
            metric_dict["l0"].append(l0)
            metric_dict["l1"].append(l1)

        if compute_variance_metrics:
            resid_sum_of_squares = (
                (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
            )
            total_sum_of_squares = (
                (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
            )
            explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares
            metric_dict["explained_variance"].append(explained_variance)
            metric_dict["mse"].append(resid_sum_of_squares)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metric_dict.items():
        metrics[f"metrics/{metric_name}"] = torch.stack(metric_values).mean().item()

    return metrics


@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:
    hook_name = sae.cfg.hook_name
    #head_index = sae.cfg.hook_head_index

    
    # original_logits, original_ce_loss = model(
    #     batch_tokens, return_type="both", **model_kwargs
    # )


    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = model.tokenizer.get_vocab()
    #Vocabulary that matches token ids to tokens
    reverse_vocab = {v: k for k, v in vocab.items()}
    unique_labels = np.unique(np.array(activation_store.dataset["token_labels"]))
    keys_labels = set(unique_labels)
    labels_tokens_id = {int(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}
    


    #We extract the ground truth to compute the cross-entropy later on
    ground_truth = batch_tokens[:,-2]
    ##Based on the reverse vocab, we create the vectors of ground truth because we only extratced the id tokens
    max_key = max((key for key in reverse_vocab.keys() if key in ground_truth))
    lookup_tensor = torch.zeros(max_key + 1,dtype=torch.int64)
    for key, value in reverse_vocab.items() :
        if key in ground_truth:
            lookup_tensor[key] = int(value)
    ground_truth_label = lookup_tensor[ground_truth]
    # vectors_ground_truth = F.one_hot(ground_truth_label, num_classes=unique_labels.size+1)
    # print(f"vectors_ground_truth {vectors_ground_truth}")
    
    original_logits = model(
        batch_tokens, return_type="logits", **model_kwargs
    )
        

    metrics = {}

    def create_classif_logits(all_logits,labels_tokens_id,unique_labels):
        all_logits = all_logits[:,-3,:]  #New shape [batch_size, voacab_size]
        classif_all_logits = torch.zeros((all_logits.size(0),unique_labels.size+1))
        sum_logits = torch.sum(all_logits,dim=1)
        for key, value in labels_tokens_id.items():
            classif_all_logits[:,key] = all_logits[:,value]
            sum_logits -= all_logits[:,value]
        classif_all_logits[:,-1] = sum_logits

        return classif_all_logits

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        activations = sae.decode(sae.encode(activations)).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)

        return activations.to(original_device)

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(
            activations.dtype
        )

        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        return new_activations.to(original_device)

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):

        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        new_activations = sae.decode(sae.encode(activations[:, :, head_index])).to(
            activations.dtype
        )
        activations[:, :, head_index] = new_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.unscale(activations)
        return activations.to(original_device)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

    def single_head_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations[:, :, head_index] = torch.zeros_like(activations[:, :, head_index])
        return activations.to(original_device)

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook

    # recons_logits, recons_ce_loss = model.run_with_hooks(
    #     batch_tokens,
    #     return_type="both",
    #     fwd_hooks=[(hook_name, partial(replacement_hook))],
    #     **model_kwargs,
    # )

    # zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
    #     batch_tokens,
    #     return_type="both",
    #     fwd_hooks=[(hook_name, zero_ablate_hook)],
    #     **model_kwargs,
    # )

    recons_logits = model.run_with_hooks(
        batch_tokens,
        return_type="logits",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        **model_kwargs,
    )

    zero_abl_logits = model.run_with_hooks(
        batch_tokens,
        return_type="logits",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        **model_kwargs,
    )

    
    classif_original_logits = create_classif_logits(original_logits,labels_tokens_id,unique_labels)
    classif_recons_logits = create_classif_logits(recons_logits,labels_tokens_id,unique_labels)
    classif_zero_abl_logits = create_classif_logits(zero_abl_logits,labels_tokens_id,unique_labels)
    loss_ce = torch.nn.CrossEntropyLoss(reduction="mean")
    original_ce_loss =  loss_ce(classif_original_logits,ground_truth_label)
    recons_ce_loss =  loss_ce(classif_recons_logits,ground_truth_label)
    zero_abl_ce_loss =  loss_ce(classif_zero_abl_logits,ground_truth_label)
       

    def kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        log_original_probs = torch.log(original_probs)
        new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        log_new_probs = torch.log(new_probs)
        kl_div = original_probs * (log_original_probs - log_new_probs)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    if compute_kl:
        recons_kl_div = kl(original_logits, recons_logits)
        zero_abl_kl_div = kl(original_logits, zero_abl_logits)
        metrics["kl_div_with_sae"] = recons_kl_div
        metrics["kl_div_with_ablation"] = zero_abl_kl_div

    if compute_ce_loss:
        metrics["ce_loss_with_sae"] = recons_ce_loss
        metrics["ce_loss_without_sae"] = original_ce_loss
        metrics["ce_loss_with_ablation"] = zero_abl_ce_loss

    return metrics



