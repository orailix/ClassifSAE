from loguru import logger
import os 
import itertools
from copy import deepcopy
from ..utils import LLMLoadConfig
from typing import Any, Dict
from sae_implementation import SentenceSAETrainingRunner, LanguageModelSAERunnerConfig

# Call the SAE training procedure implemented in `sae_implementation`
def sae_trainer(config):
    
    """
    Args:
        config: Path/name for LLMLoadConfig.autoconfig, or a filesystem path.

    Returns:
        A list of trained SAE objects (as returned by SentenceSAETrainingRunner.run()).
        
    """

    cfg = LLMLoadConfig.autoconfig(config)

    # Require pre-cached activations.
    use_cached = cfg.task_args.get("use_cached_activations", False)
    cached_path = cfg.task_args.get("cached_activations_path", None)
    if not use_cached or cached_path is None:
        raise ValueError(
            "Please first cache the activations of the investigated LLM classifier's "
            "layer before training the Sparse AutoEncoder (set both "
            "'use_cached_activations'=True and 'cached_activations_path')."
        )
    if not os.path.isdir(cached_path):
        raise ValueError(
            f"cached_activations_path does not exist or is not a directory: {cached_path}"
        )

    if cfg.task_args.get("log_to_wandb", False):
        if "WANDB_MODE" not in os.environ:
            logger.info("W&B logging enabled; WANDB_MODE not set. Using default (online).")
    
    # if 'log_to_wandb' in cfg.task_args and cfg.task_args['log_to_wandb']:
    #     os.environ["WANDB_MODE"] = "offline"
    
    
    # Work on a copy of task_args to avoid mutating the config in place.
    base_args: Dict[str, Any] = deepcopy(cfg.task_args)    
    
    # Build sweep only over keys that are explicitly lists (user intent).
    sweep_keys = [k for k, v in base_args.items() if isinstance(v, list)]
    sweep_values = [base_args[k] for k in sweep_keys]
    if not sweep_keys:
        # No sweep; use a single empty selection so we still run once
        selections = [{}]
    else:
        selections = [dict(zip(sweep_keys, vals)) for vals in itertools.product(*sweep_values)]


    for sel in selections:
        # For each experiment, merge base args with this selection.
        experiment: Dict[str, Any] = {**base_args, **sel}
        
        # Prepare activation_fn kwargs only when using topk.
        if experiment.get("activation_fn", "") == "topk":
            if "topk" not in experiment:
                raise ValueError("activation_fn='topk' requires a 'topk' value or list in task_args.")
            topk_val = experiment["topk"]
            if isinstance(topk_val, list):
                # If user gives a list, we sweep over it via the selections above.
                # Here, use the selected topk from sel, not the whole list.
                topk_selected = sel.get("topk", topk_val[0])
            else:
                topk_selected = topk_val
            experiment["activation_fn_kwargs"] = {"k": int(topk_selected)}
        # Avoid leaking 'topk' to the runner when not needed.
        experiment.pop("topk", None)
        
        # Compute d_sae for naming if not explicitly set rely on expansion_factor.
        d_in = int(experiment.get("d_in", 0))
        expansion_factor = experiment.get("expansion_factor", 0)
        d_sae = experiment.get("d_sae")
        if d_sae is None:
            if not d_in or not expansion_factor:
                raise ValueError(
                    "Must provide either 'd_sae' or both 'd_in' and 'expansion_factor' in task_args."
                )
            d_sae = int(d_in * expansion_factor)
        
        # Validate supervised classifier settings.
        lmbda_classifier = float(experiment.get("lmbda_classifier", 0.0))
        save_label = bool(experiment.get("save_label", False))
        if lmbda_classifier > 0.0 and not save_label:
            raise ValueError(
                "Supervised classifier training requested (lmbda_classifier > 0) "
                "but 'save_label' is False. Re-cache activations with labels or set save_label=True."
            )
        
        # Construct a readable run_name; leave wandb_id optional (managed by runner/config).
        base_name = experiment.get("run_name") or experiment.get("wandb_project", "sae")
        far = experiment.get("feature_activation_rate", 0.0)
        if lmbda_classifier > 0.0 and save_label:
            display_info = "Training SAE with supervised classifier"
            base_name = base_name.replace("sae", "ClassifSAE")
            run_name = f"{base_name}_d_sae_{d_sae}_activation_rate_{far}"
        else:
            display_info = "Training SAE"
            run_name = f"{base_name}_d_sae_{d_sae}_activation_rate_{far}"
        experiment["run_name"] = run_name
        experiment['wandb_id'] = run_name

        # Instantiate the runner config.
        lm_sae_runner_config = LanguageModelSAERunnerConfig(**experiment)
        logger.info(f"Starting SAE training: run_name={run_name}")

        print(f"\n######################################## BEGIN : Concepts-based explainability method - {display_info}  ########################################")

        sae = SentenceSAETrainingRunner(lm_sae_runner_config).run()

        print(f"\n######################################## END : Concepts-based explainability method - {display_info}  ########################################")

        
