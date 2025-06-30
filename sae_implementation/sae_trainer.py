# =============================================================================
# This file is adapted from:
#   SAELens (v 3.13.0) (https://github.com/jbloomAus/SAELens/blob/v3.13.0/sae_lens/training/sae_trainer.py)
#   License: MIT (see https://github.com/m-lebail/Concept_Interpretability_LLM/tree/main/SAELens_License/LICENSE)
#
#
# NOTES:
#   • We only did minor modifications here. 
#     We essentially just modified the call to `_train_step` to pass predicted label information when needed and adjusted the `_build_train_step_log_dict` function to display our model’s additional specific training losses in wandb.
# =============================================================================


import os
import json
import contextlib
import logging
import signal
from dataclasses import dataclass
from typing import Any, cast, Optional
from types import SimpleNamespace

import torch
import wandb
import random
from safetensors.torch import save_file
from torch.optim import Adam
from tqdm import tqdm
from torch.amp import GradScaler
from transformer_lens.hook_points import HookedRootModule


from .config import LanguageModelSAERunnerConfig, HfDataset
from .activations_store import ActivationsStore
from .training_sae import TrainingSAE, TrainStepOutput, TrainingSAEConfig, SPARSITY_PATH, SAE_WEIGHTS_PATH,SAE_CFG_PATH
from .optim import L1Scheduler, get_lr_scheduler




class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):
    raise InterruptedException()

# Fix the seed regarding many libraries
def set_seed(seed: int):
    # Python built-in RNG
    random.seed(seed)
    # OS 
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch CPU RNG
    torch.manual_seed(seed)
    # PyTorch GPU RNG (for single-GPU & multi-GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN backend: deterministic, no auto-benchmarking
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def _log_feature_sparsity(
    feature_sparsity: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    return torch.log10(feature_sparsity + eps).detach().cpu()



@dataclass
class TrainSAEOutput:
    sae: TrainingSAE
    checkpoint_path: str
    log_feature_sparsities: torch.Tensor


class SAETrainer:
    """
    Core SAE class used for inference. For training, see TrainingSAE.
    """

    def __init__(
        self,
        model: HookedRootModule | None,
        sae: TrainingSAE,
        activation_store: ActivationsStore,
        save_checkpoint_fn,  
        cfg: LanguageModelSAERunnerConfig,
    ) -> None:

        self.model = model
        self.sae = sae
        self.activation_store = activation_store
        self.save_checkpoint = save_checkpoint_fn
        self.cfg = cfg

        self.n_training_steps: int = 0
        self.n_training_tokens: int = 0


        self.checkpoint_thresholds = []
        if self.cfg.n_checkpoints > 0:
            self.checkpoint_thresholds = list(
                range(
                    0,
                    cfg.total_training_tokens,
                    cfg.total_training_tokens // self.cfg.n_checkpoints,
                )
            )[1:]

        self.act_freq_scores = torch.zeros(
            cast(int, cfg.d_sae),
            device=cfg.device,
        )
        self.n_forward_passes_since_fired = torch.zeros(
            cast(int, cfg.d_sae),
            device=cfg.device,
        )
        self.n_frac_active_tokens = 0
        # we don't train the scaling factor (initially)
        # set requires grad to false for the scaling factor
        for name, param in self.sae.named_parameters():
            if "scaling_factor" in name:
                param.requires_grad = False

 
        self.optimizer = Adam(
            sae.parameters(),
            lr=cfg.lr,
            betas=(
                cfg.adam_beta1,
                cfg.adam_beta2,
            ),
        )
        
        
        assert cfg.lr_end is not None  # this is set in config post-init
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            lr=cfg.lr,
            optimizer=self.optimizer,
            warm_up_steps=cfg.lr_warm_up_steps,
            decay_steps=cfg.lr_decay_steps,
            training_steps=self.cfg.total_training_steps,
            lr_end=cfg.lr_end,
            num_cycles=cfg.n_restart_cycles,
        )


        self.l1_scheduler = L1Scheduler(
            l1_warm_up_steps=cfg.l1_warm_up_steps, 
            total_steps=cfg.total_training_steps,
            final_l1_coefficient=cfg.l1_coefficient,
        )

        # Setup autocast if using
        self.scaler = GradScaler(device='cuda',enabled=self.cfg.autocast)


        if self.cfg.autocast:
            self.autocast_if_enabled = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.cfg.autocast,
            )
        else:
            self.autocast_if_enabled = contextlib.nullcontext()


    @property
    def feature_sparsity(self) -> torch.Tensor:
        return self.act_freq_scores / self.n_frac_active_tokens

    @property
    def log_feature_sparsity(self) -> torch.Tensor:
        return _log_feature_sparsity(self.feature_sparsity)

    @property
    def current_l1_coefficient(self) -> float:
        return self.l1_scheduler.current_l1_coefficient

    @property
    def dead_neurons(self) -> torch.Tensor:
        return (self.n_forward_passes_since_fired > self.cfg.dead_feature_window).bool()

    def fit(self) -> TrainingSAE:

        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")

        self._estimate_norm_scaling_factor_if_needed()

        # We count the number of training tokens corresponding to one epoch (based on the length of the training sentences dataset)
        threshold_training_tokens = 0
        number_equivalent_epoch = 0
        total_equivalent_epochs = int(self.cfg.total_training_tokens/self.cfg.len_epoch)
       
        # Train loop
        while self.n_training_tokens < self.cfg.total_training_tokens:
            # Do a training step.
            # We retrieve both the hidden state of the inspected layer concatenated with the classifier's predicted label if it was cached
            layer_acts = self.activation_store.next_batch(return_label_if_any=True)[:, 0, :].to(self.sae.device)

            self.n_training_tokens += self.cfg.train_batch_size_tokens
            threshold_training_tokens +=  self.cfg.train_batch_size_tokens

            if threshold_training_tokens >= self.cfg.len_epoch:
                threshold_training_tokens -= self.cfg.len_epoch
                number_equivalent_epoch += 1

            
            if self.cfg.save_label:
                labels = layer_acts[:,-1].to(dtype=torch.long)
                layer_acts = layer_acts[:,:-1]
                step_output = self._train_step(sae=self.sae, sae_in=layer_acts ,epoch_number=number_equivalent_epoch,total_equivalent_epochs=total_equivalent_epochs,labels=labels)
            else:
                step_output = self._train_step(sae=self.sae, sae_in=layer_acts, epoch_number=number_equivalent_epoch,total_equivalent_epochs=total_equivalent_epochs)

            if self.cfg.log_to_wandb:
                self._log_train_step(step_output)

            self._checkpoint_if_needed()
            self.n_training_steps += 1
            self._update_pbar(step_output,pbar)


        # fold the estimated norm scaling factor into the sae weights
        if self.activation_store.estimated_norm_scaling_factor is not None:
            self.sae.fold_activation_norm_scaling_factor(
                self.activation_store.estimated_norm_scaling_factor
            )

        # save final sae group to checkpoints folder
        self.save_checkpoint(
            trainer=self,
            checkpoint_name=f"final_{self.n_training_tokens}",
            wandb_aliases=["final_model"],
        )

        pbar.close()
        return self.sae

    @torch.no_grad()
    def _estimate_norm_scaling_factor_if_needed(self) -> None:
        if self.cfg.normalize_activations == "expected_average_only_in":
            self.activation_store.estimated_norm_scaling_factor = (
                self.activation_store.estimate_norm_scaling_factor()
            )
        else:
            self.activation_store.estimated_norm_scaling_factor = 1.0

    def _train_step(
        self,
        sae: TrainingSAE,
        sae_in: torch.Tensor,
        epoch_number: int,
        total_equivalent_epochs: int,
        labels: torch.Tensor=None
    ):

        sae.train()
        # Make sure the W_dec is still zero-norm
        if self.cfg.normalize_sae_decoder:
            sae.set_decoder_norm_to_unit_norm()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        # for documentation on autocasting see:
        # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with self.autocast_if_enabled:

            train_step_output = self.sae.training_forward_pass(
                sae_in=sae_in,
                dead_neuron_mask=self.dead_neurons,
                rank_dead_neurons=self.n_forward_passes_since_fired,
                current_l1_coefficient=self.current_l1_coefficient,
                epoch_number=epoch_number,
                total_epochs=total_equivalent_epochs,
                labels=labels
            )

            with torch.no_grad():
                did_fire = (train_step_output.feature_acts > 0).float().sum(-2) > 0
                self.n_forward_passes_since_fired += 1
                self.n_forward_passes_since_fired[did_fire] = 0
                self.act_freq_scores += (
                    (train_step_output.feature_acts.abs() > 0).float().sum(0)
                )
                self.n_frac_active_tokens += self.cfg.train_batch_size_tokens


        # Scaler will rescale gradients if autocast is enabled
        self.scaler.scale(
            train_step_output.loss
        ).backward()  # loss.backward() if not autocasting
        self.scaler.unscale_(self.optimizer)  # needed to clip correctly
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        self.scaler.step(self.optimizer)  
        self.scaler.update()

        if self.cfg.normalize_sae_decoder:
            sae.remove_gradient_parallel_to_decoder_directions()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.l1_scheduler.step()

        return train_step_output

    @torch.no_grad()
    def _log_train_step(self, step_output: TrainStepOutput):
        if (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0:
            wandb.log(
                self._build_train_step_log_dict(
                    output=step_output,
                    n_training_tokens=self.n_training_tokens,
                ),
                step=self.n_training_steps,
            )

    @torch.no_grad()
    def _build_train_step_log_dict(
        self,
        output: TrainStepOutput,
        n_training_tokens: int,
    ) -> dict[str, Any]:
        sae_in = output.sae_in
        sae_out = output.sae_out
        feature_acts = output.feature_acts
        mse_loss = output.mse_loss
        l1_loss = output.l1_loss
        ghost_grad_loss = output.ghost_grad_loss
        feature_sparsity_loss = output.feature_sparsity_loss
        vcr_loss = output.vcr_loss
        decoder_columns_similarity_loss = output.decoder_columns_similarity_loss
        classifier_loss = output.classifier_loss
        loss = output.loss.item()
        

        # metrics for currents acts
        l0 = (feature_acts > 0).float().sum(-1).mean()
        current_learning_rate = self.optimizer.param_groups[0]["lr"]

        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
        total_variance = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        explained_variance = 1 - per_token_l2_loss / total_variance

        return {
            # losses are normalized by their coefficents in the objective function
            "losses/mse_loss": mse_loss / self.cfg.lmbda_mse if self.cfg.lmbda_mse>0 else mse_loss,
            "losses/l1_loss": l1_loss
            / self.current_l1_coefficient if self.current_l1_coefficient!=0 else l1_loss,
            "losses/auxiliary_reconstruction_loss": output.auxiliary_reconstruction_loss,
            "losses/ghost_grad_loss": ghost_grad_loss,
            "losses/feature_sparsity_loss": feature_sparsity_loss / self.cfg.lmbda_feature_sparsity if self.cfg.lmbda_feature_sparsity>0 else feature_sparsity_loss, 
            "losses/vcr_loss": vcr_loss / self.cfg.lmbda_vcr if self.cfg.lmbda_vcr>0 else vcr_loss, 
            "losses/decoder_columns_similarity_loss": decoder_columns_similarity_loss / self.cfg.lmbda_decoder_columns_similarity if self.cfg.lmbda_decoder_columns_similarity>0 else decoder_columns_similarity_loss, 
            "losses/classifier_loss": classifier_loss / self.cfg.lmbda_classifier if self.cfg.lmbda_classifier>0 else classifier_loss,
            "losses/overall_loss": loss,
            # variance explained
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/explained_variance_std": explained_variance.std().item(),
            "metrics/l0": l0.item(),
            # sparsity
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/dead_features": self.dead_neurons.sum().item(),
            "details/current_learning_rate": current_learning_rate,
            "details/current_l1_coefficient": self.current_l1_coefficient,
            "details/n_training_tokens": n_training_tokens,
        }

   
    @torch.no_grad()
    def _build_sparsity_log_dict(self) -> dict[str, Any]:

        log_feature_sparsity = _log_feature_sparsity(self.feature_sparsity)
        wandb_histogram = wandb.Histogram(log_feature_sparsity.numpy())
        return {
            "metrics/mean_log10_feature_sparsity": log_feature_sparsity.mean().item(),
            "plots/feature_density_line_chart": wandb_histogram,
            "sparsity/below_1e-5": (self.feature_sparsity < 1e-5).sum().item(),
            "sparsity/below_1e-6": (self.feature_sparsity < 1e-6).sum().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:

        self.act_freq_scores = torch.zeros(
            self.cfg.d_sae,  
            device=self.cfg.device,
        )
        self.n_frac_active_tokens = 0

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_tokens > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint(
                trainer=self,
                checkpoint_name=self.n_training_tokens,
            )
            self.checkpoint_thresholds.pop(0)

    @torch.no_grad()
    def _update_pbar(self, step_output: TrainStepOutput, pbar: tqdm, update_interval: int = 100): 

        if self.n_training_steps % update_interval == 0:

            unscaled_mse_loss = (step_output.mse_loss * (1/self.cfg.lmbda_mse)) if (self.cfg.lmbda_mse > 0) else step_output.mse_loss
            unscaled_classifier_loss = (step_output.classifier_loss *  (1/self.cfg.lmbda_classifier)) if (self.cfg.lmbda_classifier > 0) else step_output.classifier_loss

            pbar.set_description(
                f"Training steps {self.n_training_steps}| MSE Loss {unscaled_mse_loss:.3f} | Classifier loss {unscaled_classifier_loss:.3f}"
            )
            pbar.update(update_interval * self.cfg.train_batch_size_tokens)


class SentenceSAETrainingRunner:
    """
    Class adapted from SAELens to run the training of a Sentence Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig
    model: HookedRootModule
    sae: TrainingSAE
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
        model : HookedRootModule | None = None,
        dataset: HfDataset | None = None,
    ):
       
        set_seed(cfg.seed)
        self.cfg = cfg
        self.model = model

        self.activations_store = ActivationsStore.from_config(
            self.cfg,
            self.model,
            dataset
        )
        
        if self.cfg.from_pretrained_path is not None:
            self.sae = TrainingSAE.load_from_pretrained(
                self.cfg.from_pretrained_path, self.cfg.device
            )
        else:

            self.sae = TrainingSAE(
                TrainingSAEConfig.from_dict(
                    self.cfg.get_training_sae_cfg_dict(),
                )
            )
            self._init_sae_group_b_decs()

       
    def run(self):
        """
        Run the training of the SAE.
        """
      

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = SAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae

    def _compile_if_needed(self):

        # Compile model and SAE
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)

        if self.model is not None:

            if self.cfg.compile_llm:
                self.model = torch.compile(
                    self.model,
                    mode=self.cfg.llm_compilation_mode,
                )  # type: ignore

        if self.cfg.compile_sae:
            if self.cfg.device == "mps":
                backend = "aot_eager"
            else:
                backend = "inductor"

            self.sae.training_forward_pass = torch.compile(  # type: ignore
                self.sae.training_forward_pass,
                mode=self.cfg.sae_compilation_mode,
                backend=backend,
            )  # type: ignore

    def run_trainer_with_interruption_handling(self, trainer: SAETrainer):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            print("interrupted, saving progress")
            checkpoint_name = trainer.n_training_tokens
            self.save_checkpoint(trainer, checkpoint_name=checkpoint_name)
            print("done saving")
            raise

        return sae


    def _init_sae_group_b_decs(
        self,
    ) -> None:
        """
        extract all activations at a certain layer and use for sae b_dec initialization
        """

        if self.cfg.b_dec_init_method == "geometric_median":
            layer_acts = self.activations_store.storage_buffer.detach()[:, 0, :]
            # get geometric median of the activations if we're using those.
            median = compute_geometric_median(
                layer_acts,
                maxiter=100,
            ).median
            self.sae.initialize_b_dec_with_precalculated(median)  
        elif self.cfg.b_dec_init_method == "mean":
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, 0, :]
            self.sae.initialize_b_dec_with_mean(layer_acts) 

    def save_checkpoint(
        self,
        trainer: SAETrainer,
        checkpoint_name: int | str,
        wandb_aliases: list[str] | None = None,
    ) -> str:

        checkpoint_path = f"{trainer.cfg.checkpoint_path}/{checkpoint_name}"

        os.makedirs(checkpoint_path, exist_ok=True)

        path = f"{checkpoint_path}"
        os.makedirs(path, exist_ok=True)

        if self.sae.cfg.normalize_sae_decoder:
            self.sae.set_decoder_norm_to_unit_norm()
        self.sae.save_model(path)

        # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
        # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
        config = trainer.cfg.to_dict()
        with open(f"{path}/cfg.json", "w") as f:
            json.dump(config, f)

        log_feature_sparsities = {"sparsity": trainer.log_feature_sparsity}

        log_feature_sparsity_path = f"{path}/{SPARSITY_PATH}"
        save_file(log_feature_sparsities, log_feature_sparsity_path)

        if trainer.cfg.log_to_wandb and os.path.exists(log_feature_sparsity_path):
            # Avoid wandb saving errors such as:
            #   ValueError: Artifact name may only contain alphanumeric characters, dashes, underscores, and dots. Invalid name: sae_google/gemma-2b_etc
            #sae_name = self.sae.get_name().replace("/", "__")
            sae_name = 'my_sae'
            
            model_artifact = wandb.Artifact(
                sae_name,
                type="model",
                metadata=dict(trainer.cfg.__dict__),
            )

            model_artifact.add_file(f"{path}/{SAE_WEIGHTS_PATH}")
            model_artifact.add_file(f"{path}/{SAE_CFG_PATH}")

            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            sparsity_artifact = wandb.Artifact(
                f"{sae_name}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(trainer.cfg.__dict__),
            )
            sparsity_artifact.add_file(log_feature_sparsity_path)
            wandb.log_artifact(sparsity_artifact)

        return checkpoint_path


def weighted_average(points: torch.Tensor, weights: torch.Tensor):
    weights = weights / weights.sum()
    return (points * weights.view(-1, 1)).sum(dim=0)


@torch.no_grad()
def geometric_median_objective(
    median: torch.Tensor, points: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:

    norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore

    return (norms * weights).sum()


def compute_geometric_median(
    points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    maxiter: int = 100,
    ftol: float = 1e-20,
    do_log: bool = False,
):
    """
    :param points: ``torch.Tensor`` of shape ``(n, d)``
    :param weights: Optional ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero.
        Equivalently, this is a smoothing parameter. Default 1e-6.
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :param do_log: If true will return a log of function values encountered through the course of the algorithm
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``torch.Tensor`` object of shape :math:``(d,)``
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list (None if do_log is false).
    """
    with torch.no_grad():

        if weights is None:
            weights = torch.ones((points.shape[0],), device=points.device)
        # initialize median estimate at mean
        new_weights = weights
        median = weighted_average(points, weights)
        objective_value = geometric_median_objective(median, points, weights)
        if do_log:
            logs = [objective_value]
        else:
            logs = None

        # Weiszfeld iterations
        early_termination = False
        pbar = tqdm.tqdm(range(maxiter))
        for _ in pbar:
            prev_obj_value = objective_value

            norms = torch.linalg.norm(points - median.view(1, -1), dim=1)  # type: ignore
            new_weights = weights / torch.clamp(norms, min=eps)
            median = weighted_average(points, new_weights)
            objective_value = geometric_median_objective(median, points, weights)

            if logs is not None:
                logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break

            pbar.set_description(f"Objective value: {objective_value:.4f}")

    median = weighted_average(points, new_weights)  # allow autodiff to track it
    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination=(
            "function value converged within tolerance"
            if early_termination
            else "maximum iterations reached"
        ),
        logs=logs,
    )
