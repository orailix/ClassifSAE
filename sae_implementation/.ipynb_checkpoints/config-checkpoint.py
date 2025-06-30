import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, cast

import torch
import wandb
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
}

HfDataset = DatasetDict | Dataset | IterableDatasetDict | IterableDataset

def _default_cached_activations_path(
    dataset_path: str,
    model_name: str,
    hook_name: str,
    hook_head_index: int | None,
) -> str:
    path = f"activations/{dataset_path.replace('/', '_')}/{model_name.replace('/', '_')}/{hook_name}"
    if hook_head_index is not None:
        path += f"_{hook_head_index}"
    return path



@dataclass
class CacheActivationsRunnerConfig:
    """
    Configuration for caching activations of an LLM.
    """

    # Data Generating Function (Model + Training Distibuion)
    model_class_name: str = "HookedTransformer"
    hook_name: str = "blocks.{layer}.hook_mlp_out"
    hook_layer: int = 0
    
    new_cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_head_index}"
    )
    cached_activations_path: Optional[str] = None
    # SAE Parameters
    d_in: int = 512

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    training_tokens: int = 2_000_000
    store_batch_size_prompts: int = 32
    train_batch_size_tokens: int = 4096
    normalize_activations: str = "none"  # should always be none for activation caching

    # Misc
    device: str = "cpu"
    act_store_device: str = "with_model"  # will be set by post init if with_model
    seed: int = 42
    dtype: str = "float32"
    prepend_bos: bool = True
    autocast_lm: bool = False  # autocast lm during activation fetching

    #To know where to locate the token containing the class prediction
    is_eos : bool = False
    #To know if we have to shape the forward pass so that it handles the added prompt embeddings
    prompt_tuning : bool = False

    #Wether we also cache the labels, in that case the labels are concatenated to the embedding vector
    save_label : bool = False
    
    # Activation caching stuff
    shuffle_every_n_buffers: int = 10
    n_shuffles_with_last_section: int = 10
    n_shuffles_in_entire_dir: int = 10
    n_shuffles_final: int = 100
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Autofill cached_activations_path unless the user overrode it
        if self.new_cached_activations_path is None:
            self.new_cached_activations_path = _default_cached_activations_path(
                self.dataset_path,
                self.model_name,
                self.hook_name,
                self.hook_head_index,
            )

        if self.act_store_device == "with_model":
            self.act_store_device = self.device




@dataclass
class LanguageModelSAERunnerConfig:
    """
    Configuration for training a sparse autoencoder on a language model.

    Args:
        model_class_name (str): The name of the class of the model to use. This should be either `HookedTransformer` or `HookedMamba`.
        hook_name (str): The name of the hook to use. This should be a valid TransformerLens hook.
        hook_eval (str): NOT CURRENTLY IN USE. The name of the hook to use for evaluation.
        hook_layer (int): The index of the layer to hook. Used to stop forward passes early and speed up processing.
        
        
        use_cached_activations (bool): Whether to use cached activations. This is useful when doing sweeps over the same activations.
        cached_activations_path (str, optional): The path to the cached activations.
        d_in (int): The input dimension of the SAE.
        d_sae (int, optional): The output dimension of the SAE. If None, defaults to `d_in * expansion_factor`.
        b_dec_init_method (str): The method to use to initialize the decoder bias. Zeros is likely fine.
        expansion_factor (int): The expansion factor. Larger is better but more computationally expensive.
        activation_fn (str): The activation function to use. Relu is standard.
        normalize_sae_decoder (bool): Whether to normalize the SAE decoder. Unit normed decoder weights used to be preferred.
       
       
        from_pretrained_path (str, optional): The path to a pretrained SAE. We can finetune an existing SAE if needed.
        apply_b_dec_to_input (bool): Whether to apply the decoder bias to the input. Not currently advised.
        decoder_orthogonal_init (bool): Whether to use orthogonal initialization for the decoder. Not currently advised.
        decoder_heuristic_init (bool): Whether to use heuristic initialization for the decoder. See Anthropic April Update.
        init_encoder_as_decoder_transpose (bool): Whether to initialize the encoder as the transpose of the decoder. See Anthropic April Update.
        n_batches_in_buffer (int): The number of batches in the buffer. When not using cached activations, a buffer in ram is used. The larger it is, the better shuffled the activations will be.
        training_tokens (int): The number of training tokens.
       
        store_batch_size_prompts (int): The batch size for storing activations. This controls how many prompts are in the batch of the language model when generating actiations.
        train_batch_size_tokens (int): The batch size for training. This controls the batch size of the SAE Training loop.
        normalize_activations (str): Activation Normalization Strategy. Either none, expected_average_only_in (estimate the average activation norm and divide activations by it -> this can be folded post training and set to None), or constant_norm_rescale (at runtime set activation norm to sqrt(d_in) and then scale up the SAE output).
        device (str): The device to use. Usually cuda.
        act_store_device (str): The device to use for the activation store. CPU is advised in order to save vram.
        seed (int): The seed to use.
        dtype (str): The data type to use.
        prepend_bos (bool): Whether to prepend the beginning of sequence token. You should use whatever the model was trained with.
        autocast (bool): Whether to use autocast during training. Saves vram.
        autocast_lm (bool): Whether to use autocast during activation fetching.

        
        is_eos : bool  - To know where to locate the token containing the class prediction
        prompt_tuning : bool
        save_label bool - Wether we also cache the labels, in that case the labels are concatenated to the embedding vector
        len_epoch : int - The number of training steps that can be considered as one epoch
        
        compile_llm (bool): Whether to compile the LLM.
        llm_compilation_mode (str): The compilation mode to use for the LLM.
        
        compile_sae (bool): Whether to compile the SAE.
        sae_compilation_mode (str): The compilation mode to use for the SAE.
        
        train_batch_size_tokens (int): The batch size for training.
        adam_beta1 (float): The beta1 parameter for Adam.
        adam_beta2 (float): The beta2 parameter for Adam.
        mse_loss_normalization (str): The normalization to use for the MSE loss.
        l1_coefficient (float): The L1 coefficient.
        lp_norm (float): The Lp norm.
        
        scale_sparsity_penalty_by_decoder_norm (bool): Whether to scale the sparsity penalty by the decoder norm.
        
        l1_warm_up_steps (int): The number of warm-up steps for the L1 loss.
        lr (float): The learning rate.
        lr_scheduler_name (str): The name of the learning rate scheduler to use.
        lr_warm_up_steps (int): The number of warm-up steps for the learning rate.
        lr_end (float): The end learning rate for the cosine annealing scheduler.
        lr_decay_steps (int): The number of decay steps for the learning rate.
        n_restart_cycles (int): The number of restart cycles for the cosine annealing warm restarts scheduler.
        
        
        use_ghost_grads (bool): Whether to use ghost gradients.
        coef_mse (float) : coefficient for the mse loss of the hidden state reconstruction
        penalty_activation (float): coefficient for the features sparsity activation
        penalty_vcr (float): coefficient for the vcr penalty
        penalty_decoder_columns (float) : coefficient for the decoder columns similarity
        num_classifier_features (int) : number of features within the SAE hidden layer used as inputs for the learned classifier
        percentile_max (float) : Frequency activation threshold for all features 
        causal_alpha (float): scalar to tune the weight of the causal loss
        percent_mask (float): percentage of elements masked for the second forward pass to evaluate causality

        
        feature_sampling_window (int): The feature sampling window.
        dead_feature_window (int): The dead feature window.
        dead_feature_threshold (float): The dead feature threshold.
        n_eval_batches (int): The number of evaluation batches.
        eval_batch_size_prompts (int): The batch size for evaluation.
        log_to_wandb (bool): Whether to log to Weights & Biases.
        log_activations_store_to_wandb (bool): NOT CURRENTLY USED. Whether to log the activations store to Weights & Biases.
        log_optimizer_state_to_wandb (bool): NOT CURRENTLY USED. Whether to log the optimizer state to Weights & Biases.
        wandb_project (str): The Weights & Biases project to log to.
        wandb_id (str): The Weights & Biases ID.
        run_name (str): The name of the run.
        wandb_entity (str): The Weights & Biases entity.
        wandb_log_frequency (int): The frequency to log to Weights & Biases.
        eval_every_n_wandb_logs (int): The frequency to evaluate.
        resume (bool): Whether to resume training.
        n_checkpoints (int): The number of checkpoints.
        checkpoint_path (str): The path to save checkpoints.
        verbose (bool): Whether to print verbose output.
        model_kwargs (dict[str, Any]): Additional keyword arguments for the model.
        model_from_pretrained_kwargs (dict[str, Any]): Additional keyword arguments for the model from pretrained.

        Parameters of the classifier
        classifier_input_dim (int)
        classifier_hidden_dim (int)
        classifier_nb_hidden_layer (int)
        classifier_output_dim (int)
        classifier_dropout (float)
        
    """

    # Data Generating Function (Model + Training Distibuion)
    model_class_name: str = "HookedTransformer"
    hook_name: str = "blocks.0.hook_mlp_out"
    hook_eval: str = "NOT_IN_USE"
    hook_layer: int = 0
        
    use_cached_activations: bool = False
    cached_activations_path: Optional[str] = (
        None  # Defaults to "activations/{dataset}/{model}/{full_hook_name}_{hook_head_index}"
    )

    # SAE Parameters
    architecture: Literal["standard", "gated"] = "standard"
    d_in: int = 512
    d_sae: Optional[int] = None
    b_dec_init_method: str = "geometric_median"
    expansion_factor: int = 4
    activation_fn: str = "relu"  # relu, tanh-relu, topk
    activation_fn_kwargs: dict[str, Any] = field(default_factory=dict)  # for topk
    normalize_sae_decoder: bool = True
   
    from_pretrained_path: Optional[str] = None
    apply_b_dec_to_input: bool = True
    decoder_orthogonal_init: bool = False
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False

    # Activation Store Parameters
    n_batches_in_buffer: int = 20
    training_tokens: int = 2_000_000
 
    store_batch_size_prompts: int = 32
    train_batch_size_tokens: int = 4096
    normalize_activations: str = (
        "none"  # none, expected_average_only_in (Anthropic April Update), constant_norm_rescale (Anthropic Feb Update)
    )

    # Misc
    device: str = "cpu"
    act_store_device: str = "with_model"  # will be set by post init if with_model
    seed: int = 42
    dtype: str = "float32"  # type: ignore #
    prepend_bos: bool = True

   
    autocast: bool = False  # autocast to autocast_dtype during training
    autocast_lm: bool = False  # autocast lm during activation fetching
    compile_llm: bool = False  # use torch.compile on the LLM
    llm_compilation_mode: str | None = None  # which torch.compile mode to use
    compile_sae: bool = False  # use torch.compile on the SAE
    sae_compilation_mode: str | None = None

    is_eos : bool = False
    prompt_tuning : bool = False
    save_label : bool = False

    len_epoch : int = 10000

    # Training Parameters

    ## Batch size
    train_batch_size_tokens: int = 4096

    ## Adam
    adam_beta1: float = 0
    adam_beta2: float = 0.999

    ## Loss Function
    mse_loss_normalization: Optional[str] = None
    l1_coefficient: float = 1e-3
    lp_norm: float = 1
    scale_sparsity_penalty_by_decoder_norm: bool = False
    l1_warm_up_steps: int = 0
    coef_mse: float = 1.0
    penalty_activation: float = 0.0
    penalty_vcr: float = 0.0
    penalty_decoder_columns: float = 0.0 
    num_classifier_features: int = 80
    causal_alpha: float = 0.0
    percentile_max: float = 0.0
    percent_mask: float = 30.0

    ## Learning Rate Schedule
    lr: float = 3e-4
    lr_scheduler_name: str = (
        "constant"  # constant, cosineannealing, cosineannealingwarmrestarts
    )
    lr_warm_up_steps: int = 0
    lr_end: Optional[float] = None  # only used for cosine annealing, default is lr / 10
    lr_decay_steps: int = 0
    n_restart_cycles: int = 1  # used only for cosineannealingwarmrestarts


    # Resampling protocol args
    use_ghost_grads: bool = False  # want to change this to true on some timeline.
    feature_sampling_window: int = 2000
    dead_feature_window: int = 1000  # unless this window is larger feature sampling,

    dead_feature_threshold: float = 1e-8

    context_size = 1 #We only use one token per sentence, the one preceding the answer

    # Evals
    n_eval_batches: int = 10
    eval_batch_size_prompts: int | None = None  # useful if evals cause OOM

    # WANDB
    log_to_wandb: bool = True
    log_activations_store_to_wandb: bool = False
    log_optimizer_state_to_wandb: bool = False
    wandb_project: str = "sae_training_classification"
    wandb_id: Optional[str] = None
    run_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100  # logs every 1000 steps.

    # Misc
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    verbose: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    model_from_pretrained_kwargs: dict[str, Any] = field(default_factory=dict)

    classifier_input_dim : int = 90
    classifier_hidden_dim : int = 256
    classifier_nb_hidden_layer : int = 2
    classifier_output_dim : int = 4
    classifier_dropout : float = 0.1

    def __post_init__(self):


        if self.use_cached_activations and self.cached_activations_path is None:
            self.cached_activations_path = _default_cached_activations_path(
                self.dataset_path,
                self.model_name,
                self.hook_name,
                self.hook_head_index,
            )

        if self.expansion_factor != 0:
            self.d_sae = self.d_in * self.expansion_factor
        else:
            self.expansion_factor = self.d_sae / self.d_in

        print(f"self.d_sae : {self.d_sae}")
        print(f"self.expansion_factor : {self.expansion_factor}")
        
        self.tokens_per_buffer = (
            self.train_batch_size_tokens * self.context_size * self.n_batches_in_buffer
        )

        if self.run_name is None:
            self.run_name = f"{self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.training_tokens:3.3e}"

        if self.b_dec_init_method not in ["geometric_median", "mean", "zeros"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )

        if self.normalize_sae_decoder and self.decoder_heuristic_init:
            raise ValueError(
                "You can't normalize the decoder and use heuristic initialization."
            )

        if self.normalize_sae_decoder and self.scale_sparsity_penalty_by_decoder_norm:
            raise ValueError(
                "Weighting loss by decoder norm makes no sense if you are normalizing the decoder weight norms to 1"
            )

        if self.normalize_activations not in [
            "none",
            "expected_average_only_in",
            "constant_norm_rescale",
            "layer_norm",
        ]:
            raise ValueError(
                f"normalize_activations must be none, expected_average_only_in, or constant_norm_rescale. Got {self.normalize_activations}"
            )

        if self.act_store_device == "with_model":
            self.act_store_device = self.device

        if self.lr_end is None:
            self.lr_end = self.lr / 10

        unique_id = self.wandb_id
        if unique_id is None:
            unique_id = cast(
                Any, wandb
            ).util.generate_id()  # not sure why this type is erroring
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

    
        if self.verbose:
            print(
                f"Run name: {self.d_sae}-L1-{self.l1_coefficient}-LR-{self.lr}-Tokens-{self.training_tokens:3.3e}"
            )
            # Print out some useful info:
            n_tokens_per_buffer = (
                self.store_batch_size_prompts
                * self.context_size
                * self.n_batches_in_buffer
            )
            print(f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 10 ** 6}")
            n_contexts_per_buffer = (
                self.store_batch_size_prompts * self.n_batches_in_buffer
            )
            print(
                f"Lower bound: n_contexts_per_buffer (millions): {n_contexts_per_buffer / 10 ** 6}"
            )

            total_training_steps = (
                self.training_tokens 
            ) // self.train_batch_size_tokens
            print(f"Total training steps: {total_training_steps}")

            total_wandb_updates = total_training_steps // self.wandb_log_frequency
            print(f"Total wandb updates: {total_wandb_updates}")

            # how many times will we sample dead neurons?
            n_feature_window_samples = (
                total_training_steps // self.feature_sampling_window
            )
            print(
                f"n_tokens_per_feature_sampling_window (millions): {(self.feature_sampling_window * self.context_size * self.train_batch_size_tokens) / 10 ** 6}"  #Quand on fait les stats de sparsity, se sera sur tous ces tokens
            )
            print(
                f"n_tokens_per_dead_feature_window (millions): {(self.dead_feature_window * self.context_size * self.train_batch_size_tokens) / 10 ** 6}"
            )
            print(
                f"We will reset the sparsity calculation {n_feature_window_samples} times."
            )
            # print("Number tokens in dead feature calculation window: ", self.dead_feature_window * self.train_batch_size_tokens)
            print(
                f"Number tokens in sparsity calculation window: {self.feature_sampling_window * self.train_batch_size_tokens:.2e}"
            )

        if self.use_ghost_grads:
            print("Using Ghost Grads.")

    @property
    def total_training_tokens(self) -> int:
        return self.training_tokens 

    @property
    def total_training_steps(self) -> int:
        return self.total_training_tokens // self.train_batch_size_tokens

    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            # TEMP
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "dtype": self.dtype,
            "device": self.device,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "activation_fn_str": self.activation_fn,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "prepend_bos": self.prepend_bos,
            "normalize_activations": self.normalize_activations,
            "activation_fn_kwargs": self.activation_fn_kwargs,
        }

    def get_training_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            **self.get_base_sae_cfg_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "penalty_activation": self.penalty_activation,
            "coef_mse": self.coef_mse,
            "penalty_vcr": self.penalty_vcr,
            "penalty_decoder_columns": self.penalty_decoder_columns,
            "num_classifier_features": self.num_classifier_features,
            "causal_alpha": self.causal_alpha,
            "percentile_max": self.percentile_max,
            "percent_mask": self.percent_mask,
            "is_eos" : self.is_eos,
            "prompt_tuning" : self.prompt_tuning,
            "save_label" : self.save_label,
            "len_epoch" : self.len_epoch,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "normalize_activations": self.normalize_activations,
            "classifier_input_dim": self.classifier_input_dim,
            "classifier_hidden_dim" : self.classifier_hidden_dim,
            "classifier_nb_hidden_layer" : self.classifier_nb_hidden_layer,
            "classifier_output_dim" : self.classifier_output_dim,
            "classifier_dropout": self.classifier_dropout
        }

    def to_dict(self) -> dict[str, Any]:

        cfg_dict = {
            **self.__dict__,
            # some args may not be serializable by default
            "dtype": str(self.dtype),
            "device": str(self.device),
            "act_store_device": str(self.act_store_device),
        }

        return cfg_dict

    def to_json(self, path: str) -> None:

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        with open(path + "cfg.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "LanguageModelSAERunnerConfig":
        with open(path + "cfg.json", "r") as f:
            cfg = json.load(f)
        return cls(**cfg)

