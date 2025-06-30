import time
import os
import math
from loguru import logger
import sys
sys.path.append("../../")

from ..utils import LLMLoadConfig
from ..model_training import process_dataset,get_hook_model,compute_loss_last_token
from sae_implementation import CustomCacheActivationsRunner, CacheActivationsRunnerConfig

import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm



def prompt_tuning_hook_model(hook_model,dataset_tokenized,is_eos,proportion_to_train=1):

    vocab = hook_model.tokenizer.get_vocab()
    labels = np.unique(np.array(dataset_tokenized["token_labels"]))
    keys_labels = set(labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}
    
    n_selected = int(proportion_to_train * len(dataset_tokenized))
    dataset_tokenized = dataset_tokenized.shuffle(seed=42).select(range(n_selected))

    device = hook_model.cfg.device
    data_collator = DataCollatorForLanguageModeling(tokenizer=hook_model.tokenizer,mlm=False)
    dataloader = DataLoader(dataset_tokenized, batch_size=4, collate_fn=data_collator)

    for i,batch in enumerate(tqdm(dataloader, desc="Training", unit="batch")):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(dtype=int).to(device)

        loss = hook_model.train_step(input_ids, attention_mask,compute_loss_last_token)
        hook_model.scheduler.step()
        if i%200 == 0:
            tqdm.write(f"Loss of prompt tuning : {loss}")
        

    # #Evaluation
    # hook_model.eval()
    # total_matches = 0
    # with torch.no_grad():
    #     for batch in tqdm(dataloader,desc="Evaluating", unit="batch"):
    #         input_ids = batch['input_ids'].to(device)
    #         attention_mask = batch['attention_mask'].to(dtype=int).to(device)
    #         nb_matches = hook_model.evaluate(input_ids, attention_mask, labels_tokens_id,is_eos)
    #         total_matches += nb_matches

    #     accuracy = total_matches / len(dataset_tokenized)
    #     logger.info(f"\nAverage accuracy of the model on the evaluated dataset of {len(dataset_tokenized)} samples : {accuracy}\n")
    
    torch.save(hook_model.state_dict(), 'prompt_tuning/prompt_tuning_llama_32_instruct.pth')
    


def activation_caching(config):
    
    cfg = LLMLoadConfig.autoconfig(config)
    
    
    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    #Process the dataset that we use to cache activation, we cache the train (the test dataset is used for cross-entropy evaluation and then their sole activations are not used)
    dataset_tokenized = process_dataset(cfg,split=cfg.split,tokenizer=tokenizer) 
    
    '''
    In the context below, there is no use to have more "training_tokens" to understand in the sense
    training activations than the number of prompts in the dataset Since, it would mean caching
    identical activations.
    '''
   
    if cfg.task_args['training_tokens'] > len(dataset_tokenized):
        cfg.task_args['training_tokens'] = len(dataset_tokenized)

        dir_name = os.path.dirname(cfg.task_args['new_cached_activations_path']) 
        cfg.task_args['new_cached_activations_path'] = os.path.join(dir_name,str(len(dataset_tokenized)))
            
    
    nb_activations_in_buffer = cfg.task_args['n_batches_in_buffer']*cfg.task_args['store_batch_size_prompts']*1
    logger.info(f"Caching {nb_activations_in_buffer * math.ceil(cfg.task_args['training_tokens']/nb_activations_in_buffer)} activations, {cfg.task_args['training_tokens']} of which are unique ")
    
    cache_config = CacheActivationsRunnerConfig(**cfg.task_args)
    
    #Get model hooked (HookedTransformer)
    hook_model = get_hook_model(cfg,tokenizer)

    #If we do prompt_tuning, we tune the mebeddings prompt here
    if cfg.task_args['prompt_tuning']:
        is_eos = cfg.task_args['is_eos']
        # logger.info("Prompt tuning of the HookedTransformer model")
        # prompt_tuning_hook_model(hook_model,dataset_tokenized,is_eos)
        hook_model.load_state_dict(torch.load('prompt_tuning/prompt_tuning_llama_32_instruct.pth'))

    

    #print("Padding side of the hook model tokenizer : ",hook_model.tokenizer.padding_side)

    #Load the predicted labels
    dir_to_load_predictions = os.path.join(cfg.dir_to_save_metrics,cfg.split,"predicted_labels")
    file_to_load_predicted_labels = os.path.join(dir_to_load_predictions,f"{cfg.model_name}_{cfg.dataset_name}.pt")
    if not os.path.isfile(file_to_load_predicted_labels):
        raise FileNotFoundError(f"File {file_to_load_predicted_labels} does not exist. Make sure to run 'eval-model' before caching the activations of the model so that it also includes the associated predicted labels")
    predicted_labels = torch.load(file_to_load_predicted_labels)

    if len(dataset_tokenized) != predicted_labels.shape[0]:
        raise ValueError(
            f"Length mismatch: dataset length is {len(dataset_tokenized)} but the predicted labels are of number {predicted_labels.shape[0]}"
        )

    # Add the predicted labels to the dataset
    predicted_labels_list = predicted_labels.tolist()
    dataset_tokenized = dataset_tokenized.add_column("predicted_labels", predicted_labels_list)

    print(f"dataset_tokenized : {dataset_tokenized['predicted_labels']}")

    
    start_time = time.time()

    runner = CustomCacheActivationsRunner(cache_config,hook_model,dataset_tokenized)

    print("-" * 50)
    print(runner.__str__())
    print("-" * 50)
    runner.run()


    end_time = time.time()


    print(f"Total time taken for caching : {end_time - start_time:.2f} seconds")
    print(
        f"{cfg.task_args['training_tokens'] / ((end_time - start_time)*10**6):.2f} Million Tokens / Second"
    )
        