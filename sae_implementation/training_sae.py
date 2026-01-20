# =============================================================================
# This file is adapted from:
#   SAELens (v 3.13.0) (https://github.com/jbloomAus/SAELens/blob/v3.13.0/sae_lens/training/training_sae.py)
#   License: MIT (see https://github.com/orailix/ClassifSAE/blob/main/SAELens_License/LICENSE)
#
#
# NOTES:
#   • We enabled joint caching of the LLM-predicted label for each sentence by concatenating the predicted label index with the cached activation ('self.save_label' option)
#   • Modification of `training_forward_pass` to pass predicted label information when needed as well as the current epoch number.
#   • In `training_forward_pass` function, we define additional loss terms in the final training objective (classifier head loss and activation rate sparsity loss) to encourage ClassifSAE to learn  
#      decorrelated classification relevant features in z_class, they are activated based on the weights defined by the user.
#    
# =============================================================================

import json
import os
from typing import Any, Optional
from dataclasses import dataclass, fields

import torch
import einops
from jaxtyping import Float
from torch import nn
import torch.nn.functional as F
import math

from .config import LanguageModelSAERunnerConfig, DTYPE_MAP
from .sae import SAE, SAEConfig, SAE_CFG_PATH, SAE_WEIGHTS_PATH
from .toolkit.pretrained_sae_loaders import (
    handle_config_defaulting,
    read_sae_from_disk,
)

SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


def sparse_efficient_outer_product_sum(vectors):
    """
    Compute the sum of outer products V_i @ V_i^T for sparse vectors efficiently.
    
    Args:
    vectors (torch.Tensor): A 2D tensor of shape (N, D) where N is the number of vectors
                             and D is the vector dimension
    
    Returns:
    torch.Tensor: The sum of outer products V_i @ V_i^T with shape (D, D)
    """
    # Ensure the input requires gradients
    vectors = vectors.clone().requires_grad_()
    
    # Compute sum of outer products using efficient matrix multiplication
    result = torch.matmul(vectors.t(), vectors)
    
    return result


@dataclass
class TrainStepOutput:
    sae_in: torch.Tensor
    sae_out: torch.Tensor
    feature_acts: torch.Tensor
    loss: torch.Tensor  # we need to call backwards on this
    mse_loss: float
    l1_loss: float
    ghost_grad_loss: float
    auxiliary_reconstruction_loss: float = 0.0
    activation_rate_sparsity_loss: float = 0.0
    vcr_loss: float = 0.0
    decoder_columns_similarity_loss: float = 0.0
    classifier_loss: float = 0.0


@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig):

    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    lmbda_mse: float
    lmbda_activation_rate: float
    lmbda_vcr: float
    lmbda_decoder_columns_similarity: float
    lmbda_classifier: float
    num_classifier_features: int
    nb_classes: int
    feature_activation_rate: float
    normalize_sae_decoder: bool
    decoder_orthogonal_init: bool
    mse_loss_normalization: Optional[str]
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False
    seed: int

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":

       
        return cls(
            # base config
            architecture=cfg.architecture,
            d_in=cfg.d_in,
            d_sae=cfg.d_sae, 
            dtype=cfg.dtype,
            device=cfg.device,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            activation_fn_str=cfg.activation_fn,
            activation_fn_kwargs=cfg.activation_fn_kwargs,
            apply_b_dec_to_input=cfg.apply_b_dec_to_input,
            prepend_bos=cfg.prepend_bos,
            # Training cfg
            l1_coefficient=cfg.l1_coefficient,
            lp_norm=cfg.lp_norm,
            use_ghost_grads=cfg.use_ghost_grads,
            lmbda_mse=cfg.lmbda_mse,
            lmbda_activation_rate=cfg.lmbda_activation_rate,
            lmbda_vcr=cfg.lmbda_vcr,
            lmbda_decoder_columns_similarity=cfg.lmbda_decoder_columns_similarity, 
            lmbda_classifier=cfg.lmbda_classifier,
            num_classifier_features=cfg.num_classifier_features,
            nb_classes=cfg.nb_classes,
            feature_activation_rate=cfg.feature_activation_rate,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            normalize_activations=cfg.normalize_activations,
            seed=cfg.seed,
        )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAEConfig":
        # remove any keys that are not in the dataclass
        # since we sometimes enhance the config with the whole LM runner config
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }
        return TrainingSAEConfig(**valid_config_dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "l1_coefficient": self.l1_coefficient,
            "lp_norm": self.lp_norm,
            "use_ghost_grads": self.use_ghost_grads,
            "lmbda_mse": self.lmbda_mse,
            "lmbda_activation_rate": self.lmbda_activation_rate,
            "lmbda_vcr": self.lmbda_vcr,
            "lmbda_decoder_columns_similarity": self.lmbda_decoder_columns_similarity,
            "lmbda_classifier": self.lmbda_classifier,
            "feature_activation_rate":self.feature_activation_rate,
            "num_classifier_features": self.num_classifier_features,
            "nb_classes": self.nb_classes,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
            "seed": self.seed,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific parameters.
    def get_base_sae_cfg_dict(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "d_in": self.d_in,
            "d_sae": self.d_sae,
            "activation_fn_str": self.activation_fn_str,
            "activation_fn_kwargs": self.activation_fn_kwargs,
            "apply_b_dec_to_input": self.apply_b_dec_to_input,
            "dtype": self.dtype,
            "hook_name": self.hook_name,
            "hook_layer": self.hook_layer,
            "device": self.device,
            "prepend_bos": self.prepend_bos,
            "normalize_activations": self.normalize_activations,
            "num_classifier_features": self.num_classifier_features,
            "nb_classes": self.nb_classes,
            
        }




class TrainingSAE(SAE):
    """
    A SAE used for training. This class provides a `training_forward_pass` method which calculates
    losses used for training.
    """

    cfg: TrainingSAEConfig
    use_error_term: bool
    dtype: torch.dtype
    device: torch.device
   

    def __init__(self, cfg: TrainingSAEConfig ,use_error_term: bool = False):

        base_sae_cfg = SAEConfig.from_dict(cfg.get_base_sae_cfg_dict())
        super().__init__(base_sae_cfg)
        self.cfg = cfg  

        self.epoch = 0

        self.encode_with_hidden_pre_fn = (
            self.encode_with_hidden_pre
            if cfg.architecture != "gated"
            else self.encode_with_hidden_pre_gated
        )

        self.check_cfg_compatibility()

        self.use_error_term = use_error_term

        self.initialize_weights_complex()

        self.mse_loss_fn = self._get_mse_loss_fn()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "TrainingSAE":
        return cls(TrainingSAEConfig.from_dict(config_dict))

    def check_cfg_compatibility(self):
        if self.cfg.architecture == "gated":
            assert (
                self.cfg.use_ghost_grads is False
            ), "Gated SAEs do not support ghost grads"
            assert self.use_error_term is False, "Gated SAEs do not support error terms"

    def encode_standard(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Calcuate SAE features from inputs
        """
        feature_acts, _ = self.encode_with_hidden_pre_fn(x)
        return feature_acts

    def encode_with_hidden_pre(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:

        x = x.to(self.dtype)
        x = self.hook_sae_input(x)
        x = self.run_time_activation_norm_fn_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)

        # "... d_in, d_in d_sae -> ... d_sae",
        hidden_pre = self.hook_sae_acts_pre(sae_in @ self.W_enc + self.b_enc)
     
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))

        return feature_acts, hidden_pre

    def encode_with_hidden_pre_gated(
        self, x: Float[torch.Tensor, "... d_in"]
    ) -> tuple[Float[torch.Tensor, "... d_sae"], Float[torch.Tensor, "... d_sae"]]:

        x = x.to(self.dtype)
        x = self.hook_sae_input(x)
        x = self.run_time_activation_norm_fn_in(x)

        # apply b_dec_to_input if using that method.
        sae_in = x - (self.b_dec * self.cfg.apply_b_dec_to_input)

        # Gating path with Heaviside step function
        gating_pre_activation = sae_in @ self.W_enc + self.b_gate 
        active_features = (gating_pre_activation > 0).to(self.dtype)

        # Magnitude path with weight sharing
        magnitude_pre_activation = sae_in @ (self.W_enc * self.r_mag.exp()) + self.b_mag
        # magnitude_pre_activation_noised = magnitude_pre_activation + (
        #     torch.randn_like(magnitude_pre_activation) * self.cfg.noise_scale * self.training
        # )
        feature_magnitudes = self.activation_fn(
            magnitude_pre_activation
        )  # magnitude_pre_activation_noised)

        # Return both the gated feature activations and the magnitude pre-activations
        return (
            active_features * feature_magnitudes,
            magnitude_pre_activation,
        )  # magnitude_pre_activation_noised

    def forward(
        self,
        x: Float[torch.Tensor, "... d_in"],
    ) -> Float[torch.Tensor, "... d_in"]:

        feature_acts, _ = self.encode_with_hidden_pre_fn(x)
        sae_out = self.decode(feature_acts)

        return sae_out


    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
        rank_dead_neurons: Optional[torch.Tensor] = None,
        epoch_number: int = 0,
        total_epochs: int = 50,
        labels: torch.Tensor = None
    ) -> TrainStepOutput:


        # Ensure labels are indeed long
        labels = labels.long()
        
        # Do a forward pass to retrieve the hidden layer SAE activations, before and after the application of the activation function
        feature_acts, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
      
        # Split SAE features into classification-related and context-related parts
        z_class = feature_acts[:,:self.cfg.num_classifier_features]
        z_ctx = feature_acts[:,self.cfg.num_classifier_features:]

        # SAE DECODING
        if epoch_number < total_epochs // 2:
            sae_out = self.decode( torch.cat([z_class,z_ctx], dim=-1))
        else:
            sae_out = self.decode( torch.cat([z_class.detach(),z_ctx], dim=-1))


        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        # We rescale the mse loss based on its weight in the final training objective
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()


        # GHOST GRADS
        if self.cfg.use_ghost_grads and self.training and dead_neuron_mask is not None:

            # first half of second forward pass
            _, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
            ghost_grad_loss = self.calculate_ghost_grad_loss(
                x=sae_in,
                sae_out=sae_out,
                per_item_mse_loss=per_item_mse_loss,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
                rank_dead_neurons=rank_dead_neurons,
            )
        else:
            ghost_grad_loss = 0.

        
       
        if self.cfg.lmbda_classifier > 0 :
            
            '''
            Supervised learning component of ClassifSAE. A linear classifier is trained to predict the same labels as the LLM classifier based on the SAE features z_class as input. 
            z_class is assigned as the truncation of the first 'self.cfg.num_classifier_features' dimensions in the SAE hidden layer. 
            '''
            if labels is not None:
                
                batch_dim, sae_hidden_dim = feature_acts.shape            
        
                logits = z_class @ self.classifier_weight.t() + self.classifier_bias
                ce_loss = nn.CrossEntropyLoss()
                classifier_loss = ce_loss(logits, labels)
                classifier_loss = classifier_loss 
                
            else:
                classifier_loss = torch.tensor(0.0)   
           
        else:
            classifier_loss = torch.tensor(0.0)


        if self.cfg.lmbda_activation_rate > 0 :
           
            batch_dim, d_sae = feature_acts.shape

            # Determine the number of elements to exclude per column
            exclude_count = int(batch_dim * self.cfg.feature_activation_rate)

            with torch.no_grad():
                _, idxs = torch.topk(feature_acts, exclude_count, dim=0)
                mask = torch.ones_like(feature_acts, dtype=torch.bool) 
                mask.scatter_(0, idxs, False)  # place 'False' at top-k positions


            masked_features_acts = mask * feature_acts
                
            # Compute the sum of the remaining elements for each column
            sum_mask_features_acts = (masked_features_acts).sum(dim=0)

            # We only penalize feature activation rate above the frequency threshold imposed by the user
            activation_rate_sparsity_loss = sum_mask_features_acts.sum() / batch_dim


        else:
            activation_rate_sparsity_loss = torch.tensor(0.0)
            
        if self.cfg.lmbda_vcr > 0 :

            batch_dim, _ = feature_acts.shape
            #Compute the mean vector
            batch_means = feature_acts[:,:self.cfg.num_classifier_features].mean(dim=0)

            #Centered vectors
            z_vector = feature_acts[:,:self.cfg.num_classifier_features]-batch_means
            
            #Compute the covariance matrix
            cov_matrix = 1/(batch_dim-1) * sparse_efficient_outer_product_sum(z_vector)
            
            # Compute the sum of squares of non-diagonal elements
            squared_off_diagonal = (cov_matrix.triu(diagonal=1) ** 2).sum() + \
                                   (cov_matrix.tril(diagonal=-1) ** 2).sum()

                        
            #regularized standard deviation
            rsd = torch.sqrt(torch.diagonal(cov_matrix) + 1e-6)

            variance_terme = torch.relu(1 - rsd).mean()

            vcr_loss = self.cfg.lmbda_vcr * (1/self.cfg.num_classifier_features) * (squared_off_diagonal+variance_terme)
        else:
            vcr_loss = torch.tensor(0.0)

            
        if self.cfg.lmbda_decoder_columns_similarity > 0:

            d_in, d_sae = self.W_enc.shape

            dot_products_dec = torch.matmul(self.W_dec[:self.cfg.num_classifier_features,:], self.W_dec[:self.cfg.num_classifier_features,:].T)  # Shape: (n_features, n_features)
            
            # Extract the upper triangular part without the diagonal
            upper_triangle_dec = torch.triu(dot_products_dec, diagonal=1).abs()
            
            # Compute the sum of unique dot products (excluding the diagonal)
            sum_cossim_dec = torch.sum(upper_triangle_dec)

            decoder_columns_similarity_loss = self.cfg.lmbda_decoder_columns_similarity * (1/(d_in *d_sae)) * sum_cossim_dec
        else:
            decoder_columns_similarity_loss = torch.tensor(0.0)


        if self.cfg.architecture == "gated":
            # Gated SAE Loss Calculation

            # Shared variables
            sae_in_centered = (
                sae_in - self.b_dec * self.cfg.apply_b_dec_to_input
            )
            pi_gate = sae_in_centered @ self.W_enc + self.b_gate
            pi_gate_act = torch.relu(pi_gate)

            # SFN sparsity loss - summed over the feature dimension and averaged over the batch
            l1_loss = (
                current_l1_coefficient
                * torch.sum(pi_gate_act * self.W_dec.norm(dim=1), dim=-1).mean()
            )

            # Auxiliary reconstruction loss - summed over the feature dimension and averaged over the batch
            via_gate_reconstruction = pi_gate_act @ self.W_dec + self.b_dec
            aux_reconstruction_loss = torch.sum(
                (via_gate_reconstruction - sae_in) ** 2, dim=-1
            ).mean()
            
            # Multi-objective loss balancing
            loss = self.cfg.lmbda_mse * (mse_loss + ghost_grad_loss) + aux_reconstruction_loss +  self.cfg.lmbda_activation_rate * activation_rate_sparsity_loss +  self.cfg.lmbda_classifier * classifier_loss

            
        else:
            
            # default SAE sparsity loss
            weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
            sparsity = weighted_feature_acts.norm(
                p=self.cfg.lp_norm, dim=-1
            )  # sum over the feature dimension


            # If we impose a L1 penalization. 
            # This is not necessary if we already use the TopK activation function as we directly enforce the L0 sparsity within the SAE hidden layer.
            l1_loss = (current_l1_coefficient * sparsity).mean() 

            aux_reconstruction_loss = torch.tensor(0.0)
            
            # Multi-objective loss balancing
            loss = self.cfg.lmbda_mse * (mse_loss + ghost_grad_loss) +  self.cfg.lmbda_activation_rate * activation_rate_sparsity_loss +  self.cfg.lmbda_classifier * classifier_loss


        self.epoch = epoch_number

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=loss,
            mse_loss=mse_loss.item(),
            l1_loss=l1_loss.item(),
            ghost_grad_loss=(
                ghost_grad_loss.item()
                if isinstance(ghost_grad_loss, torch.Tensor)
                else ghost_grad_loss
            ),
            auxiliary_reconstruction_loss=aux_reconstruction_loss.item(),
            activation_rate_sparsity_loss = activation_rate_sparsity_loss.item(),
            vcr_loss = vcr_loss.item(),
            decoder_columns_similarity_loss = decoder_columns_similarity_loss.item(),
            classifier_loss = classifier_loss.item()
        )

    def calculate_ghost_grad_loss(
        self,
        x: torch.Tensor,
        sae_out: torch.Tensor,
        per_item_mse_loss: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor,
        rank_dead_neurons: torch.Tensor,
    ) -> torch.Tensor:
        
        # 1.
        residual = x - sae_out
        l2_norm_residual = torch.norm(residual, dim=-1)

        
        # 2.
        # ghost grads use an exponentional activation function, ignoring whatever
        # the activation function is in the SAE. The forward pass uses the dead neurons only.
        feature_acts_dead_neurons_only = torch.exp(hidden_pre[:, dead_neuron_mask])
        ghost_out = feature_acts_dead_neurons_only @ self.W_dec[dead_neuron_mask, :]
        l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
        norm_scaling_factor = l2_norm_residual / (1e-6 + l2_norm_ghost_out * 2)
        ghost_out = ghost_out * norm_scaling_factor[:, None].detach()

        # 3. There is some fairly complex rescaling here to make sure that the loss
        # is comparable to the original loss. This is because the ghost grads are
        # only calculated for the dead neurons, so we need to rescale the loss to
        # make sure that the loss is comparable to the original loss.
        # There have been methodological improvements that are not implemented here yet
        # see here: https://www.lesswrong.com/posts/C5KAZQib3bzzpeyrg/full-post-progress-update-1-from-the-gdm-mech-interp-team#Improving_ghost_grads
        per_item_mse_loss_ghost_resid = self.mse_loss_fn(ghost_out, residual.detach())
        mse_rescaling_factor = (
            per_item_mse_loss / (per_item_mse_loss_ghost_resid + 1e-6)
        ).detach()
        per_item_mse_loss_ghost_resid = (
            mse_rescaling_factor * per_item_mse_loss_ghost_resid
        )

        return per_item_mse_loss_ghost_resid.mean()

    @torch.no_grad()
    def _get_mse_loss_fn(self) -> Any:

        def standard_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return torch.nn.functional.mse_loss(preds, target, reduction="none")

        def batch_norm_mse_loss_fn(
            preds: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            target_centered = target - target.mean(dim=0, keepdim=True)
            normalization = target_centered.norm(dim=-1, keepdim=True)
            return torch.nn.functional.mse_loss(preds, target, reduction="none") / (
                normalization + 1e-6
            )

        if self.cfg.mse_loss_normalization == "dense_batch":
            return batch_norm_mse_loss_fn
        else:
            return standard_mse_loss_fn

    @classmethod
    def load_from_pretrained(
        cls,
        path: str,
        device: str = "cpu",
        dtype: str | None = None,
    ) -> "TrainingSAE":

        # get the config
        config_path = os.path.join(path, SAE_CFG_PATH)
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        cfg_dict = handle_config_defaulting(cfg_dict)
        cfg_dict["device"] = device
        if dtype is not None:
            cfg_dict["dtype"] = dtype

        weight_path = os.path.join(path, SAE_WEIGHTS_PATH)
        cfg_dict, state_dict = read_sae_from_disk(
            cfg_dict=cfg_dict,
            weight_path=weight_path,
            device=device,
            dtype=DTYPE_MAP[cfg_dict["dtype"]],
        )
        
        sae_cfg = TrainingSAEConfig.from_dict(cfg_dict)

        sae = cls(sae_cfg)
        sae.load_state_dict(state_dict)

        return sae

    def initialize_weights_complex(self):
        """ """

        if self.cfg.decoder_orthogonal_init:
            self.W_dec.data = nn.init.orthogonal_(self.W_dec.data.T).T

        elif self.cfg.decoder_heuristic_init:
            self.W_dec = nn.Parameter(
                torch.rand(
                    self.cfg.d_sae, self.cfg.d_in, dtype=self.dtype, device=self.device
                )
            )
            self.initialize_decoder_norm_constant_norm()

        # Then we initialize the encoder weights (either as the transpose of decoder or not)
        if self.cfg.init_encoder_as_decoder_transpose:
            self.W_enc.data = self.W_dec.data.T.clone().contiguous()
        else:
            self.W_enc = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(
                        self.cfg.d_in,
                        self.cfg.d_sae,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            )

        if self.cfg.normalize_sae_decoder:
            with torch.no_grad():
                # Anthropic normalize this to have unit columns
                self.set_decoder_norm_to_unit_norm()

    ## Initialization Methods
    @torch.no_grad()
    def initialize_b_dec_with_precalculated(self, origin: torch.Tensor):
        out = torch.tensor(origin, dtype=self.dtype, device=self.device)
        self.b_dec.data = out

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, all_activations: torch.Tensor):
        previous_b_dec = self.b_dec.clone().cpu()
        out = all_activations.mean(dim=0)

        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)

        print("Reinitializing b_dec with mean of activations")
        print(
            f"Previous distances: {previous_distances.median(0).values.mean().item()}"
        )
        print(f"New distances: {distances.median(0).values.mean().item()}")

        self.b_dec.data = out.to(self.dtype).to(self.device)

    ## Training Utils
    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def initialize_decoder_norm_constant_norm(self, norm: float = 0.1):
        """
        A heuristic proceedure inspired by:
        https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """

        # ensure W_dec norms at unit norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        # assert self.W_dec.grad is not None

        # if W_dec is frozen then no grad has been accumulated
        if self.W_dec.grad is None:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
