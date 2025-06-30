import time
import os
import math
from loguru import logger


from ..utils import LLMLoadConfig
from ..llm_classifier_tuning import process_dataset,get_hook_model
from sae_implementation import SentenceCacheActivationsRunner, CacheActivationsRunnerConfig

import torch
from transformers import AutoTokenizer



# Call the caching procedure of the LLM classifier activations implemented in `sae_implementation`
def activation_caching(config):
    
    cfg = LLMLoadConfig.autoconfig(config)
        
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Process the dataset that we use to cache activations, we cache the train split (the test dataset is used for evaluation)
    dataset_tokenized = process_dataset(cfg,split=cfg.split,tokenizer=tokenizer) 
    
    
    # There is no use to have more "training_tokens" than the number of sentences in the dataset since it would mean caching identical activations.
    if cfg.task_args['training_tokens'] > len(dataset_tokenized):
        cfg.task_args['training_tokens'] = len(dataset_tokenized)

        dir_name = os.path.dirname(cfg.task_args['new_cached_activations_path']) 
        cfg.task_args['new_cached_activations_path'] = os.path.join(dir_name,str(len(dataset_tokenized)))
            
    
    nb_activations_in_buffer = cfg.task_args['n_batches_in_buffer']*cfg.task_args['store_batch_size_prompts']
    logger.info(f"Caching {nb_activations_in_buffer * math.ceil(cfg.task_args['training_tokens']/nb_activations_in_buffer)} activations, {cfg.task_args['training_tokens']} of which are unique ")
    
    cache_config = CacheActivationsRunnerConfig(**cfg.task_args)
    
    # Get model hooked (HookedTransformer)
    hook_model = get_hook_model(cfg,tokenizer)
    
    # Load the labels predicted by the investigated LLM classifier
    dir_to_load_predictions = os.path.join(cfg.dir_to_save_metrics,cfg.split,"predicted_labels")
    file_to_load_predicted_labels = os.path.join(dir_to_load_predictions,f"{cfg.model_name}_{cfg.dataset_name}.pt")
    if not os.path.isfile(file_to_load_predicted_labels):
        raise FileNotFoundError(f"File {file_to_load_predicted_labels} does not exist. Make sure to run 'eval-model' before caching the activations of the model so that it also includes the associated predicted labels")
    predicted_labels = torch.load(file_to_load_predicted_labels,map_location=torch.device('cpu'),weights_only=True)

    if len(dataset_tokenized) != predicted_labels.shape[0]:
        raise ValueError(
            f"Length mismatch: dataset length is {len(dataset_tokenized)} but the predicted labels are of number {predicted_labels.shape[0]}"
        )

    # Add the predicted labels to the dataset
    predicted_labels_list = predicted_labels.tolist()
    dataset_tokenized = dataset_tokenized.add_column("predicted_labels", predicted_labels_list)
    
    start_time = time.time()

    runner = SentenceCacheActivationsRunner(cache_config,hook_model,dataset_tokenized)

    print("-" * 50)
    print(runner.__str__())
    print("-" * 50)
    runner.run()


    end_time = time.time()


    print(f"Total time taken for caching : {end_time - start_time:.2f} seconds")
    print(
        f"{cfg.task_args['training_tokens'] / ((end_time - start_time)*10**6):.2f} Million Tokens / Second"
    )
        