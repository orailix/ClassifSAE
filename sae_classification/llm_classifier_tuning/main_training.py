from ..utils import LLMTrainerConfig
from .models import get_model
from .handle_datasets import process_dataset
from .trainer import get_trainer

from transformers import AutoTokenizer
from loguru import logger
import numpy as np
import wandb
import gc
import torch

import wandb
import shutil, os

def fine_tuning_model(config):
    
    cfg = LLMTrainerConfig.autoconfig(config)

    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side="left"

    
    #Process the train and test dataset
    train_dataset_tokenized = process_dataset(cfg,split="train",tokenizer=tokenizer)
    test_dataset_tokenized = process_dataset(cfg,split="test",tokenizer=tokenizer)

    
    #Quick statistics on the datasets
    list_len_train_dataset = [len(seq_tokens) for seq_tokens in train_dataset_tokenized['input_ids']]
    max_length_train = max(list_len_train_dataset)
    logger.info(f"The maximum number of tokens in the longest sentence of the train dataset is: {max_length_train}")
    mean_length_train = np.mean(list_len_train_dataset)
    logger.info(f"Number of tokens in average in the sentences of the train dataset is: {mean_length_train}")
    
    list_len_test_dataset = [len(seq_tokens) for seq_tokens in test_dataset_tokenized['input_ids']]
    max_length_test = max(list_len_test_dataset)
    logger.info(f"The maximum number of tokens in the longest sentence of the test dataset is: {max_length_test}")
    mean_length_test = np.mean(list_len_test_dataset)
    logger.info(f"Number of tokens in average in the sentences of the test dataset is: {mean_length_test}")
    
    
    #Load the local model
    model = get_model(cfg)
    
    #Load the trainer
    trainer = get_trainer(cfg,model,train_dataset_tokenized,test_dataset_tokenized,tokenizer)

    # Fine-tune the autoregressive pre-trained LLM on the classification task given a template so that its answer is encoded with only one integer.
    res_train = trainer.train()
    
    if 'report_to' in cfg.training_args and (cfg.training_args['report_to']=='wandb'):
        #Rename the wandb folder of the run
        run = wandb.run  
        run_id = run.id
        run_name = f"{cfg.model_name}_fine_tuning_{cfg.dataset_name}_{cfg.training_args['max_steps']}"

        wandb.finish()

        wandb_root = "wandb"
        offline_folders = os.listdir(wandb_root)

        for folder in offline_folders:
            if folder.endswith(run_id):
                original_path = os.path.join(wandb_root, folder)
                new_path = os.path.join(wandb_root, run_name)
                shutil.move(original_path, new_path)
                break        

    
    
    #Checkpoints of the model in PATH_CHECKPOINTS
    del model
    gc.collect()
    torch.cuda.empty_cache()
    