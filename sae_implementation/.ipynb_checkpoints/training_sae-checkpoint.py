import json
import logging
import os
from typing import Any, cast, Optional
from dataclasses import dataclass, fields

import torch
from transformer_lens.hook_points import HookedRootModule

from .config import LanguageModelSAERunnerConfig, DTYPE_MAP
from .sae import SAE, SAEConfig, SAE_CFG_PATH, SAE_WEIGHTS_PATH
from .toolkit.pretrained_sae_loaders import (
    handle_config_defaulting,
    read_sae_from_disk,
)


import einops
from jaxtyping import Float
from torch import nn
import torch.nn.functional as F
import torchsort 
import math


import time


SPARSITY_PATH = "sparsity.safetensors"
SAE_WEIGHTS_PATH = "sae_weights.safetensors"
SAE_CFG_PATH = "cfg.json"


import torch




def pearson_correlation(x, y):
    """
    Computes the mean Pearson correlation coefficient across a batch of vectors.

    Parameters:
        x (torch.Tensor): First tensor of shape (batch_size, dimension).
        y (torch.Tensor): Second tensor of shape (batch_size, dimension).

    Returns:
        float: Mean Pearson correlation coefficient across the batch.
    """
    if x.size() != y.size():
        raise ValueError("Tensors x and y must have the same shape.")

    # Calculate means along dimensions
    mean_x = torch.mean(x, dim=1, keepdim=True)
    mean_y = torch.mean(y, dim=1, keepdim=True)

    # Center the tensors
    x_centered = x - mean_x
    y_centered = y - mean_y

    # Compute the numerator and denominator for Pearson correlation
    numerator = torch.sum(x_centered * y_centered, dim=1)
    denominator = torch.sqrt(torch.sum(x_centered**2, dim=1) * torch.sum(y_centered**2, dim=1))

    # Avoid division by zero
    if torch.any(denominator == 0):
        raise ValueError("Division by zero: One of the rows has zero variance.")

    correlations = numerator / denominator

    return torch.mean(correlations)


def exp_decay(epoch, decay_rate=0.05, min_weight=0.0):
    return max(min_weight, math.exp(-decay_rate * epoch))

def exponential_increase(epoch, k=0.05, max_weight=1.0, initial_weight=0.0):
    return min(max_weight, initial_weight + (max_weight - initial_weight)*(1 - math.exp(-k * epoch)))

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


# Alternative version with sparse tensor conversion
def sparse_tensor_outer_product_sum(vectors):
    """
    Compute the sum of outer products using sparse tensor conversion.
    
    Args:
    vectors (torch.Tensor): A 2D tensor of shape (N, D)
    
    Returns:
    torch.Tensor: The sum of outer products V_i @ V_i^T
    """
    # Threshold for sparsity
    threshold = 1e-8
    
    # Create a sparse representation
    sparse_indices = (vectors.abs() > threshold).nonzero().t()
    sparse_values = vectors[sparse_indices[0], sparse_indices[1]]
    
    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_indices, 
        sparse_values, 
        size=vectors.shape
    )
    
    # Compute sum of outer products
    result = torch.matmul(sparse_tensor.t(), sparse_tensor).to_dense()
    
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
    feature_sparsity_loss: float = 0.0
    vcr_loss: float = 0.0
    decoder_columns_similarity_loss: float = 0.0
    causal_loss: float = 0.0



@dataclass(kw_only=True)
class TrainingSAEConfig(SAEConfig):

    # Sparsity Loss Calculations
    l1_coefficient: float
    lp_norm: float
    use_ghost_grads: bool
    coef_mse: float
    penalty_activation: float
    penalty_vcr: float
    penalty_decoder_columns: float
    num_classifier_features: int
    causal_alpha: float
    percentile_max: float
    percent_mask: float
    normalize_sae_decoder: bool
    decoder_orthogonal_init: bool
    mse_loss_normalization: Optional[str]
    decoder_heuristic_init: bool = False
    init_encoder_as_decoder_transpose: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False

    @classmethod
    def from_sae_runner_config(
        cls, cfg: LanguageModelSAERunnerConfig
    ) -> "TrainingSAEConfig":

        return cls(
            # base config
            architecture=cfg.architecture,
            d_in=cfg.d_in,
            d_sae=cfg.d_sae,  # type: ignore
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
            penalty_activation=cfg.penalty_activation,
            coef_mse=cfg.coef_mse,
            penalty_vcr=cfg.penalty_vcr,
            penalty_decoder_columns=cfg.penalty_decoder_columns, 
            num_classifier_features=cfg.num_classifier_features,
            causal_alpha=cfg.causal_alpha,
            percentile_max=cfg.percentile_max,
            percent_mask=cfg.percent_mask,
            normalize_sae_decoder=cfg.normalize_sae_decoder,
            decoder_orthogonal_init=cfg.decoder_orthogonal_init,
            mse_loss_normalization=cfg.mse_loss_normalization,
            decoder_heuristic_init=cfg.decoder_heuristic_init,
            init_encoder_as_decoder_transpose=cfg.init_encoder_as_decoder_transpose,
            scale_sparsity_penalty_by_decoder_norm=cfg.scale_sparsity_penalty_by_decoder_norm,
            normalize_activations=cfg.normalize_activations,
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
            "coef_mse": self.coef_mse,
            "penalty_activation": self.penalty_activation,
            "penalty_vcr": self.penalty_vcr,
            "penalty_decoder_columns": self.penalty_decoder_columns,
            "num_classifier_features": self.num_classifier_features,
            "causal_alpha": self.causal_alpha,
            "percentile_max":self.percentile_max,
            "percent_mask": self.percent_mask,
            "normalize_sae_decoder": self.normalize_sae_decoder,
            "decoder_orthogonal_init": self.decoder_orthogonal_init,
            "init_encoder_as_decoder_transpose": self.init_encoder_as_decoder_transpose,
            "mse_loss_normalization": self.mse_loss_normalization,
            "decoder_heuristic_init": self.decoder_heuristic_init,
            "scale_sparsity_penalty_by_decoder_norm": self.scale_sparsity_penalty_by_decoder_norm,
            "normalize_activations": self.normalize_activations,
        }

    # this needs to exist so we can initialize the parent sae cfg without the training specific
    # parameters. Maybe there's a cleaner way to do this
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
            "classifier_input_dim": self.classifier_input_dim,
            "classifier_hidden_dim" : self.classifier_hidden_dim,
            "classifier_nb_hidden_layer" : self.classifier_nb_hidden_layer,
            "classifier_output_dim" : self.classifier_output_dim,
            "classifier_dropout": self.classifier_dropout,
            "num_classifier_features": self.num_classifier_features,
            
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
    #llm_model: HookedRootModule | None 

    def __init__(self, cfg: TrainingSAEConfig ,use_error_term: bool = False):

        base_sae_cfg = SAEConfig.from_dict(cfg.get_base_sae_cfg_dict())
        super().__init__(base_sae_cfg)
        self.cfg = cfg  # type: ignore

        #self.classifier = SAE_Classifier(input_size=self.cfg.d_sae,hidden_size=self.cfg.topk,output_size=)
        #self.llm_model = llm_model
        

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
        labels: torch.Tensor = None
    ) -> TrainStepOutput:


        #Randomly add noise on sae_in : 
        noise = torch.randn_like(sae_in) * (1 / sae_in.shape[1])
        sae_in += noise
        
        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        feature_acts, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)

        # if epoch_number <= 45:

        #     num_elements = feature_acts.numel()
        #     top_k = int(num_elements * 0.005)
        
        #     # Handle edge case: if top_k is 0, just return a mask of all False
        #     if top_k == 0:
        #         return torch.zeros_like(feature_acts, dtype=torch.bool)
        
        #     # Flatten and get top_k values
        #     x_flat = feature_acts.view(-1)
        #     top_values, _ = torch.topk(x_flat, top_k)
        
        #     # Get the smallest value among the top_k
        #     threshold = top_values[-1]
        
        #     # Create mask
        #     mask = (feature_acts >= threshold)
    
        #     feature_acts = feature_acts * mask
        # else:

        #     # Get the top p values and their indices along each column (dim=0)
        #     top_values, top_indices = torch.topk(feature_acts, 20, dim=0)
        
        #     # Create a mask of the same shape as the input tensor
        #     mask = torch.zeros_like(feature_acts, dtype=torch.bool)
        
        #     # Scatter True into the mask at the top indices
        #     mask.scatter_(0, top_indices, True)
    
        #     feature_acts = feature_acts * mask


        # percentile = 0.05
        # batch_dim, vector_dim = feature_acts.shape
        # p = int(batch_dim * percentile)  # Number of top values to exclude per column
        # #Get the top p values and their indices along each column (dim=0)
        # top_values, top_indices = torch.topk(feature_acts, p, dim=0)
    
        # # Create a mask of the same shape as the input tensor
        # mask = torch.zeros_like(feature_acts, dtype=torch.bool)
    
        # # Scatter True into the mask at the top indices
        # mask.scatter_(0, top_indices, True)

        # feature_acts = feature_acts * mask
    
        sae_out = self.decode(feature_acts)

        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        # if epoch_number>30:
        #    self.W_dec.requires_grad = False
        #    self.b_dec.requires_grad = False

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
            #print(f"ghost_grad_loss : {ghost_grad_loss}")
        else:
            ghost_grad_loss = 0.

        if self.cfg.causal_alpha > 0 :

            # self.llm_model.eval()
            # output_logits = self.llm_model(input=sae_out.unsqueeze(1),start_at_layer=self.cfg.hook_layer,return_type='logits') #skips the embedding then runs the rest of the model
            # #output_logits  : batch_size,1,vocab_size

            # with torch.no_grad():
               
            #     #mask a given % of the activated features
            #     mask = torch.ones_like(feature_acts)
            #     #Random tensor
            #     random_tensor = torch.rand_like(feature_acts)
                
            #     # Calculate the threshold for the given percentage
            #     p = self.cfg.percent_mask
            #     threshold = torch.quantile(torch.where(feature_acts != 0, random_tensor, float('inf')), 
            #                                q=1 - p/100, dim=1, keepdim=True)
                
            #     # Create the mask: 1 where random_tensor > threshold, 0 otherwise
            #     # This effectively zeros out p% of non-zero values
            #     mask = torch.where(feature_acts != 0, (random_tensor > threshold).float(), mask)
              

            # altered_feature_acts = mask * feature_acts
            # altered_sae_out = self.decode(altered_feature_acts)
            # altered_output_logits = self.llm_model(input=altered_sae_out.unsqueeze(1),start_at_layer=self.cfg.hook_layer,return_type='logits') #skips the embedding then runs the rest of the model

           
            # # causal_loss = - self.cfg.causal_alpha * F.mse_loss(output_logits, altered_output_logits)
            # batch_dim, vector_dim = feature_acts.shape
            # # Compute the Jacobian of the hidden layer w.r.t. the input
            # hidden_derivative = feature_acts * (1 - feature_acts)  # Derivative of sigmoid activation
            # jacobian = torch.matmul(hidden_derivative, self.W_enc.T)  # Jacobian for each input
            # frobenius_norm = torch.sum(jacobian ** 2)
            # causal_loss = (1/batch_dim) * (1/vector_dim) * frobenius_norm

            # batch_dim, vector_dim = feature_acts.shape

            # non_zero_mask = (feature_acts != 0).float()  # Create a mask for non-zero elements
            # nonzero_count = torch.sum(non_zero_mask, dim=1, keepdim=True)  # Count of non-zero elements per vector
            
            # # Avoid division by zero
            # nonzero_count = torch.clamp(nonzero_count, min=1.0)
    
            # # Compute mean and variance of non-zero elements
            # masked_representation = feature_acts * non_zero_mask
            # mean = torch.sum(masked_representation, dim=1, keepdim=True) / nonzero_count
            # variance = torch.sum(non_zero_mask * (masked_representation - mean) ** 2, dim=1) / nonzero_count.squeeze(1)
    
            # # Penalize low variance (negative of mean variance)
            # spread_loss = (1/batch_dim) * (1/vector_dim) * (-torch.mean(variance))
        

            # # causal_loss = spread_loss
            # correlation = pearson_correlation(sae_in, sae_out)
            # causal_loss = - correlation

            if labels is not None:
                classifier_logits_preds = feature_acts[:,:self.cfg.num_classifier_features] @ self.classifier_weight.t() + self.classifier_bias
                #classifier_loss = self.classifier.classification_loss(feature_acts[:,:90],labels)
               
                
                if epoch_number > 30000000:

                   # Freeze the classifier_weight and classifier_bias parameters
                    self.classifier_weight.requires_grad = False
                    self.classifier_bias.requires_grad = False
                    
                    with torch.no_grad():
                   
                        # #mask a given % of the activated features
                        # mask = torch.ones_like(feature_acts)
                        # #Random tensor
                        # random_tensor = torch.rand_like(feature_acts)
                        
                        # # Calculate the threshold for the given percentage
                        # p = self.cfg.percent_mask
                        # print(f"p : {p}")
                        # threshold = torch.quantile(torch.where(feature_acts != 0, random_tensor, float('inf')), 
                        #                            q=1 - p/100, dim=1, keepdim=True)
                        
                        # # Create the mask: 1 where random_tensor > threshold, 0 otherwise
                        # # This effectively zeros out p% of non-zero values
                        # random_mask = torch.where(feature_acts != 0, (random_tensor > threshold).float(), mask)

                        p = self.cfg.percent_mask
                        #print(f"p : {p}")
                        # # Flatten the tensor to work with indices
                        # flat_feature_acts = feature_acts.flatten()
                        
                        # # Get the indices of non-zero elements
                        # non_zero_indices = torch.nonzero(flat_feature_acts).squeeze()
                        
                        # # Number of non-zero elements to zero out
                        # num_to_zero = int(len(non_zero_indices) * p / 100)
                        
                        # # Randomly select indices to zero out
                        # indices_to_zero = torch.randperm(len(non_zero_indices))[:num_to_zero]
                        
                        # # Create a mask for the selected indices
                        # random_mask = torch.ones_like(flat_feature_acts)
                        # random_mask[non_zero_indices[indices_to_zero]] = 0
                        
                        # # Reshape the mask to the original tensor shape and apply it
                        # altered_feature_acts = feature_acts * random_mask.reshape(feature_acts.shape)

                        num_to_zero = int(90 * p / 100)
                        masks=[]
                        k=10

                        for _ in range(k):
                            # Create a new column mask for each variation
                            column_mask = torch.ones(90).to(feature_acts.device)
                            columns_to_zero = torch.randperm(90)[:num_to_zero]
                            column_mask[columns_to_zero] = 0
                            masks.append(column_mask)

                        # Stack masks into a single tensor for efficient broadcasting
                        masks = torch.stack(masks)  # Shape: (k, d)


                    altereds_feature_acts = feature_acts[:,:90].unsqueeze(0) * masks.unsqueeze(1)  # Shape: (k, bs, d)
                    altered_classifier_logits_preds = altereds_feature_acts @ self.classifier_weight.t() + self.classifier_bias # Shape: (k, bs, num_classes)
                    # print(f"non zero altered_feature_acts : {torch.count_nonzero(altereds_feature_acts)/k}")
                    # print(f"non zero feature_acts : {torch.count_nonzero(feature_acts)}")
                    classifier_logits_preds = classifier_logits_preds.unsqueeze(0)
                    mse_losses_logits = F.mse_loss(classifier_logits_preds, altered_classifier_logits_preds, reduction='none') 
                    mse_logits_diff = mse_losses_logits.mean()
                    # probs_feature_acts = F.softmax(classifier_logits_preds, dim=-1)
                    # probs_altered_feature_acts = F.softmax(altered_classifier_logits_preds, dim=-1)
                    #mse_logits_diff = F.mse_loss(classifier_logits_preds, altered_classifier_logits_preds)
                    #print(f"mse_logits_diff : {mse_logits_diff}")

                    causal_loss = - self.cfg.causal_alpha * mse_logits_diff
            
                else:
                    ce_loss = nn.CrossEntropyLoss()
                    classifier_loss = ce_loss(classifier_logits_preds, labels)
                    # print(f"loss_classifier : {loss_classifier}")
                    causal_loss = self.cfg.causal_alpha * classifier_loss
            else:
                causal_loss = torch.tensor(0.0)


            
            
           
        else:
            causal_loss = torch.tensor(0.0)


        if self.cfg.penalty_activation > 0:

            batch_dim, vector_dim = feature_acts.shape

            # Determine the number of elements to exclude per column
            exclude_count = int(batch_dim * self.cfg.percentile_max)

            with torch.no_grad():
                _, idxs = torch.topk(feature_acts, exclude_count, dim=0)
                mask = torch.ones_like(feature_acts, dtype=torch.bool) 
                mask.scatter_(0, idxs, False)  # place 'False' at top-k positions

            masked_features_acts = mask * feature_acts
            # Compute the sum of the remaining elements for each column
            sum_mask_features_acts = (masked_features_acts).sum(dim=0)

            feature_sparsity_loss = self.cfg.penalty_activation * (1/batch_dim) * ( (sum_mask_features_acts.sum()))
        else:
            feature_sparsity_loss = torch.tensor(0.0)
            
        if self.cfg.penalty_vcr > 0 :

            batch_dim, vector_dim = feature_acts.shape
            #Compute the mean vector
            batch_means = feature_acts[:,:self.cfg.num_classifier_features].mean(dim=0)

            #Centered vectors
            z_vector = feature_acts[:,:self.cfg.num_classifier_features]-batch_means

            # # # non_zero_count = torch.count_nonzero(z_vector)
            # # # print(f"Number of non zeros : {non_zero_count}")
            
            #Compute the covariance matrix
            cov_matrix = 1/(batch_dim-1) * sparse_efficient_outer_product_sum(z_vector)
            
            # Compute the sum of squares of non-diagonal elements
            squared_off_diagonal = (cov_matrix.triu(diagonal=1) ** 2).sum() + \
                                   (cov_matrix.tril(diagonal=-1) ** 2).sum()

                        
            #regularized standard deviation
            rsd = torch.sqrt(torch.diagonal(cov_matrix) + 1e-6)

            variance_terme = torch.relu(1 - rsd).mean()

            vcr_loss = self.cfg.penalty_vcr * (1/self.cfg.num_classifier_features) * (squared_off_diagonal+variance_terme)
        else:
            vcr_loss = torch.tensor(0.0)

            
        if self.cfg.penalty_decoder_columns > 0:

            d_in, d_sae = self.W_enc.shape

            dot_products_dec = torch.matmul(self.W_dec[:self.cfg.num_classifier_features,:], self.W_dec[:self.cfg.num_classifier_features,:].T)  # Shape: (n_features, n_features)
            
            # Extract the upper triangular part without the diagonal
            upper_triangle_dec = torch.triu(dot_products_dec, diagonal=1).abs()
            
            # Compute the sum of unique dot products (excluding the diagonal)
            sum_cossim_dec = torch.sum(upper_triangle_dec)

            decoder_columns_similarity_loss = self.cfg.penalty_decoder_columns * (1/(d_in *d_sae)) * sum_cossim_dec
        else:
            decoder_columns_similarity_loss = torch.tensor(0.0)

            
            
        '''
        if self.cfg.penalty_activation > 0 and epoch_number >= 0 :

            batch_dim, vector_dim = feature_acts.shape
            #Compute the mean vector
            batch_means = feature_acts.mean(dim=0)

            #Centered vectors
            z_vector = feature_acts-batch_means

            # # # non_zero_count = torch.count_nonzero(z_vector)
            # # # print(f"Number of non zeros : {non_zero_count}")
            
            #Compute the covariance matrix
            cov_matrix = 1/(batch_dim-1) * sparse_efficient_outer_product_sum(z_vector)
            
            # Compute the sum of squares of non-diagonal elements
            squared_off_diagonal = (cov_matrix.triu(diagonal=1) ** 2).sum() + \
                                   (cov_matrix.tril(diagonal=-1) ** 2).sum()

                        
            #regularized standard deviation
            rsd = torch.sqrt(torch.diagonal(cov_matrix) + 1e-6)

            variance_terme = torch.relu(1 - rsd).mean()
            
            #vcr_loss = (1/batch_dim) * (1/vector_dim) * (squared_off_diagonal)#+variance_terme)

            # batch_dim, vector_dim = feature_acts.shape

           
            vcr_loss = (1/batch_dim) * (1/vector_dim) * (squared_off_diagonal+variance_terme)

            # Compute the dot product matrix
            dot_products = torch.matmul(feature_acts.T, feature_acts)  # Shape: (n_features, n_features)
            
            # Extract the upper triangular part without the diagonal
            upper_triangle = torch.triu(dot_products, diagonal=1)**2
            
            # Compute the sum of unique dot products (excluding the diagonal)
            sum_cossim = torch.sum(upper_triangle)

            cos_loss = (1/batch_dim) * (1/vector_dim) * sum_cossim

            #cosine_similarity_loss = vcr_loss #+ 0.1*cos_loss 

            # Compute the dot product matrix        
            d_in, d_sae = self.W_enc.shape
            dot_products_enc = torch.matmul(self.W_enc.T, self.W_enc)  # Shape: (n_features, n_features)
            
            # Extract the upper triangular part without the diagonal
            upper_triangle_enc = torch.triu(dot_products_enc, diagonal=1)
            
            # Compute the sum of unique dot products (excluding the diagonal)
            sum_cossim_enc = torch.sum(upper_triangle_enc) 

            cos_loss_enc = 1/(d_sae * d_in) * sum_cossim_enc

            dot_products_dec = torch.matmul(self.W_dec, self.W_dec.T)  # Shape: (n_features, n_features)
            
            # Extract the upper triangular part without the diagonal
            upper_triangle_dec = torch.triu(dot_products_dec, diagonal=1).abs()
            
            # Compute the sum of unique dot products (excluding the diagonal)
            sum_cossim_dec = torch.sum(upper_triangle_dec)

            cos_loss_dec = 1/(d_in *d_sae) * sum_cossim_dec

            #cosine_similarity_loss = 0.1*cos_loss + 5*vcr_loss#+ 5*vcr_loss #+ 0.05*cos_loss #+ 10*vcr_loss#100*vcr_loss + cos_loss

            # MoG sparsity loss
            #mog_loss = (1/batch_dim) * (1/vector_dim) * self.mog_loss(hidden_pre)

            
            # Step 1: Compute the threshold for the top 10% in each column
            alpha = 50.0
            # p = int(batch_dim * percentile)  # Number of top values to exclude per column
            # thresholds, _ = torch.topk(feature_acts, p, dim=0, largest=True, sorted=True)
            
            # # The threshold is the smallest value in the top p%
            # threshold = thresholds[-1, :]
            
            # # Step 2: Create a mask for values below the threshold
            # mask = feature_acts < threshold
            
            # # Step 3: Compute the sum excluding top p% values for each column
            # mask_feature_acts = torch.where(mask, feature_acts, torch.zeros_like(feature_acts))
            # sum_mask_features_acts = (mask_feature_acts**2).sum()
            # #print(f"sum_mask_features_acts : {sum_mask_features_acts}")
            # over_token_loss = (1/batch_dim) * (1/vector_dim) * sum_mask_features_acts
    
            # masked_cols = []
            # for col_idx in range(vector_dim):
            #     col_vals = feature_acts[:, col_idx].unsqueeze(0)  # shape: (batch_size,)
    
            #     ranks = torchsort.soft_rank(
            #         col_vals,
            #         regularization="l2",
            #         regularization_strength=0.1
            #     )

            #     boundary = (batch_dim - exclude_count)  # approximate cutoff
    
            #     mask = torch.sigmoid(alpha * (boundary - soft_ranks))
    
            #     trimmed_col = col_vals * mask
            #     masked_cols.append(trimmed_col)

            # # Stack columns back together
            # trimmed_feature_acts = torch.stack(masked_cols, dim=1)  # shape: (batch_size, d)
        
    
            # Determine the number of elements to exclude per column
            exclude_count = int(batch_dim * self.cfg.percentile_max)
            
            # # Sort the tensor along the rows (dim=0) for each column
            # sorted_feature_acts, _ = torch.sort(feature_acts, dim=0, descending=True)
            # print(f"sorted_feature_acts grad : {sorted_feature_acts.grad}")
            
            # # Exclude the top `exclude_count` rows from each column
            # trimmed_feature_acts = sorted_feature_acts[exclude_count:, :]


            with torch.no_grad():
                _, idxs = torch.topk(feature_acts, exclude_count, dim=0)
                mask = torch.ones_like(feature_acts, dtype=torch.bool) 
                mask.scatter_(0, idxs, False)  # place 'False' at top-k positions
                
            masked_features_acts = mask * feature_acts
            # Compute the sum of the remaining elements for each column
            sum_mask_features_acts = (masked_features_acts).sum(dim=0)

            # with torch.no_grad():
            #     nonzero_counts = (sum_mask_features_acts != 0).sum(dim=0)
            #     # Find the highest count of non-zero values among the columns
            #     highest_nonzero_count = nonzero_counts.max().item()

            # alpha = 50.0  # a large constant controlling "steepness"
            # soft_nonzero = torch.sigmoid(alpha * torch.abs(masked_features_acts))
            # soft_count_nonzero = soft_nonzero.sum()  # This is now a differentiable "approx" 
            #                                  # to the number of non-zero elements.

            over_token_loss = (1/batch_dim) * ( (sum_mask_features_acts.sum())) + cos_loss_dec #+ vcr_loss # + 100000*cos_loss_enc + 100000*cos_loss_dec )
            #print(f"sum_mask_features_acts.sum() : {sum_mask_features_acts.sum()}")
            # print(f"cos_loss_enc : {100000*cos_loss_enc}")
            #print(f"cos_loss_dec : {cos_loss_dec}")
            #print(f"nonzero_counts : {nonzero_counts}")
            #print(f"over_token_loss : {over_token_loss}")
            # print(f"vcr_loss : {vcr_loss}")
            # print(f"cos_loss_enc : {cos_loss_enc}")
            # print(f"cos_loss_dec : {cos_loss_dec}")
            # print(f"cos_loss : {cos_loss}")
            cosine_sim_loss = over_token_loss #+ cos_loss_dec #+ 0.1*cos_loss_enc + 0.1*vcr_loss# + (1e-2)*cos_loss#+ 10*cos_loss_dec #+  100*cos_loss_enc # + 0.001*cos_loss
            cosine_similarity_loss = self.cfg.penalize_cossim * cosine_sim_loss
        
        else:
            cosine_similarity_loss = torch.tensor(0.0)

        '''

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

            loss = mse_loss + l1_loss + aux_reconstruction_loss + feature_sparsity_loss + vcr_loss + decoder_columns_similarity_loss
        else:
            # default SAE sparsity loss
            weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
            sparsity = weighted_feature_acts.norm(
                p=self.cfg.lp_norm, dim=-1
            )  # sum over the feature dimension

            l1_loss = (current_l1_coefficient * sparsity).mean() 


            if causal_loss > 0:
                # loss = exp_decay(epoch_number)*mse_loss + l1_loss + ghost_grad_loss + exponential_increase(epoch_number)*cosine_similarity_loss + causal_loss
                loss = exp_decay(epoch_number)*self.cfg.coef_mse*mse_loss + l1_loss + ghost_grad_loss + feature_sparsity_loss + vcr_loss + decoder_columns_similarity_loss + exponential_increase(epoch_number)*causal_loss
            else:
                loss = self.cfg.coef_mse*mse_loss + l1_loss + ghost_grad_loss + feature_sparsity_loss + vcr_loss + decoder_columns_similarity_loss  
                  

            aux_reconstruction_loss = torch.tensor(0.0)

    
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
            feature_sparsity_loss = feature_sparsity_loss.item(),
            vcr_loss = vcr_loss.item(),
            decoder_columns_similarity_loss = decoder_columns_similarity_loss.item(),
            causal_loss = causal_loss.item()
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

        # with torch.no_grad():
        #     assert "k" in self.cfg.activation_fn_kwargs, "Provide top_k"
        #     k = self.cfg.activation_fn_kwargs.get("k", 5)  # Default k to 5 if not provided
        #     k_aux = 2*k
        
        # _, indices_topk_dead_neurons = torch.topk(rank_dead_neurons, k_aux)
        
        # 1.
        residual = x - sae_out
        l2_norm_residual = torch.norm(residual, dim=-1)

        # sparse_representation_dead_neurons = hidden_pre[:, indices_topk_dead_neurons]
        # ghost_out = sparse_representation_dead_neurons @ self.W_dec[indices_topk_dead_neurons,:]

        # aux_loss = torch.sum(
        #         (gap_error - residual) ** 2, dim=-1
        #     ).mean()

        
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
        #return 0.01 * aux_loss

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
        # TODO: Parameterise this as a function of m and n

        # ensure W_dec norms at unit norm
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data *= norm  # will break tests but do this for now.

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_in) shape
        """
        assert self.W_dec.grad is not None  # keep pyright happy

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
