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

   
    #Temporary
    #load_indices = np.load("important_indices_original.npy")
    #k = int(0.01*120000)
    #indices_to_keep = load_indices[:k].tolist()
    #indices_to_keep = np.random.choice(len(train_dataset_tokenized),k,replace=False)
    # all_indices = np.arange(len(train_dataset_tokenized))
    # print(indices_to_remove)
    #keep_indices = np.setdiff1d(all_indices, indices_to_remove).tolist()
    #train_dataset_tokenized = train_dataset_tokenized.select(indices_to_keep)
    #print(f'New len ! : {len(train_dataset_tokenized)}')

    #print(train_dataset_tokenized)
    
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
    
    #Do the finetuning
    res_train = trainer.train()
    
    if 'report_to' in cfg.training_args and (cfg.training_args['report_to']=='wandb'):
        wandb.finish()
    
    #Checkpoints of the model in PATH_CHECKPOINTS
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    #Copy the latest checkpoint model into 