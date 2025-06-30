from loguru import logger
import sys
import os 
sys.path.append("../../")
#sys.path.append("SAELens")
import itertools

from ..utils import LLMLoadConfig
from ..model_training import get_hook_model, process_dataset


from sae_implementation import CustomSAETrainingRunner, LanguageModelSAERunnerConfig, MLPClassifierConfig
from transformers import AutoTokenizer
import numpy as np



def sae_trainer(config):

    cfg = LLMLoadConfig.autoconfig(config)
    
    
    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    '''
    In the original SAELens code, the dataset is used if there are not enough activations stored in 'cached_activations' given the required number of activations we specified
    we want to train on.
    
    We change this on : activations_store.py -> get_buffer : self.next_cache_idx -> reset self.next_cache_idx when we do not find files 'self.cached_activations_path}/{self.next_cache_idx}.safetensors'
    anymore after incrementation. So that we restart training on the same cached data. So the number of realized epochs can be computed as (number_tokens_to_train_on / number_tokens_cached)
    Here tokens have to be understood as activations of the hidden state corresponding to one token for the specified layer in hook_layer.
    '''
    # if not cfg.task_args['use_cached_activations']:
    #     dataset_tokenized = process_dataset(cfg,split="unsupervised",tokenizer=tokenizer) 
    # else :
    #     dataset_tokenized = None

    #We use the tokenized dataset here with the test split so that when it evaluates the quality of the SAE during training, it can use tokenized sentences of the test split (to compute ce loss and KL divergence w.r.t the baseline)
    dataset_tokenized = process_dataset(cfg,split="test",tokenizer=tokenizer)
    
    #Get model hooked (HookedTransformer)
    hook_model = get_hook_model(cfg,tokenizer)

    
    if 'log_to_wandb' in cfg.task_args and cfg.task_args['log_to_wandb']:
        os.environ["WANDB_MODE"] = "offline"
    
    
    # Ensure all values are lists, in case we want to try multiple hyperparameters
    for key, value in cfg.task_args.items():
        if not isinstance(value, list):
            cfg.task_args[key] = [value]
    
    #Special case for the parameter topk used when we set up a TopK activation function because SAELens expect a dictionary with that parameter
    topk_list_dict = []
    for k in cfg.task_args['topk']:
        topk_list_dict.append({"k" : k})     
    cfg.task_args['activation_fn_kwargs'] = topk_list_dict
    del cfg.task_args['topk']

    # Create all combinations of hyperparameters
    keys, values = zip(*cfg.task_args.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    
    
    #For every experiment, we set the d_sae and percentile_max parameters in the wandb_id
    for i,experiment in enumerate(experiments):
        d_sae = experiment['d_sae']
        threshold_activation_frequency = experiment['percentile_max']
        with_label =  experiment['save_label']
        #experiment['wandb_id'] += "_Run_"+str(i)
        if with_label:
            experiment['wandb_id'] += f"_d_sae_{d_sae}_activation_freq_{threshold_activation_frequency}_with_label"
        else:
            experiment['wandb_id'] += f"_d_sae_{d_sae}_activation_freq_{threshold_activation_frequency}"
        experiment['run_name'] = experiment['wandb_id']

        #experiment = {key: value for key, value in experiment.items() if key != "is_eos"}
        
        lm_sae_runner_config = LanguageModelSAERunnerConfig(
            **experiment
        )
        
        #The SAE is automatically saved at PATH_CHECKPOINT_SAE/wandb_id/final_{cfg.task_args.training_tokens}
        sparse_autoencoder = CustomSAETrainingRunner(lm_sae_runner_config,override_model=hook_model,override_dataset=dataset_tokenized).run()
        #sparse_autoencoder = SAETrainingRunner(lm_sae_runner_config,override_model=hook_model,override_dataset=dataset_tokenized).run()
    
    
    
    
    
    
