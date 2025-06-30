from loguru import logger
import os 
import itertools
from ..utils import LLMLoadConfig
from ..llm_classifier_tuning import process_dataset, get_hook_model
from sae_implementation import SentenceSAETrainingRunner, LanguageModelSAERunnerConfig
from transformers import AutoTokenizer


# Call the SAE training procedure implemented in `sae_implementation`
def sae_trainer(config):

    cfg = LLMLoadConfig.autoconfig(config)

    '''    
    In the original SAELens code, the dataset is used if there are not enough activations stored in 'cached_activations' given the required number of activations we specified we want to train on.
    
    We change this in : activations_store.py -> get_buffer : self.next_cache_idx -> reset self.next_cache_idx when we do not find files 'self.cached_activations_path}/{self.next_cache_idx}.safetensors'
    anymore after incrementation. So that we restart training on the same cached data. The number of actual epochs can be computed as (number_tokens_to_train_on / number_tokens_cached)
    Here tokens have to be understood as activations of the hidden state of the predicted class-generating token for the specified layer in hook_layer.
    '''
    
    # We check whether the activations have been cached already, in that case, we don't need to load neither the model nor the dataset
    if not cfg.task_args['use_cached_activations'] or cfg.task_args['cached_activations_path'] == None:

        #Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
        )
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        dataset_tokenized = process_dataset(cfg,split=cfg.split,tokenizer=tokenizer)

        #Get model hooked (HookedTransformer)
        hook_model = get_hook_model(cfg,tokenizer)

    
    if 'log_to_wandb' in cfg.task_args and cfg.task_args['log_to_wandb']:
        os.environ["WANDB_MODE"] = "offline"
    
    
    # Ensure all values are lists, in case we want to try multiple hyperparameters
    for key, value in cfg.task_args.items():
        if not isinstance(value, list):
            cfg.task_args[key] = [value]
    
    # Special case for the parameter topk. When the TopK activation function is set up, SAELens expects a dictionary with that parameter.
    topk_list_dict = []
    for k in cfg.task_args['topk']:
        topk_list_dict.append({"k" : k})     
    cfg.task_args['activation_fn_kwargs'] = topk_list_dict
    del cfg.task_args['topk']

    # Create all combinations of hyperparameters
    keys, values = zip(*cfg.task_args.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    
    # For every experiment, we set the dimension of the hidden layer 'd_sae', the targeted maximum feature activation rate 'feature_activation_rate' and whether we use the labels for a supervised training of the SAE.
    for i,experiment in enumerate(experiments):
        d_sae = experiment['d_sae']
        feature_activation_rate = experiment['feature_activation_rate']
        with_label =  experiment['save_label']
        if with_label and (experiment['lmbda_classifier'] > 0) :
            experiment['wandb_id'] += f"_d_sae_{d_sae}_activation_rate_{feature_activation_rate}_supervised_classification"
        else:
            experiment['wandb_id'] += f"_d_sae_{d_sae}_activation_rate_{feature_activation_rate}"
        experiment['run_name'] = experiment['wandb_id']
        
        lm_sae_runner_config = LanguageModelSAERunnerConfig(
            **experiment
        )
        
        # The SAE is automatically saved at PATH_CHECKPOINT_SAE/wandb_id/final_{cfg.task_args.training_tokens}
        if not cfg.task_args['use_cached_activations'] or cfg.task_args['cached_activations_path'] == None:
            
            sparse_autoencoder = SentenceSAETrainingRunner(lm_sae_runner_config,hook_model,dataset_tokenized).run()
        
        else:
            sparse_autoencoder = SentenceSAETrainingRunner(lm_sae_runner_config).run()
       
    
    
    
    
    
