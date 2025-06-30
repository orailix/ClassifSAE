from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from safetensors import safe_open
from loguru import logger
import torch
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve


from transformers.utils.hub import TRANSFORMERS_CACHE
from transformers.utils.hub import default_cache_path

from ..utils import LLMLoadConfig
from .models import get_model
from .handle_datasets import process_dataset


# Compute metrics of model performance on the batch
def compute_metrics_eval(predicted_labels, inputs,labels_tokens_id,dict_metrics,is_eos=True):
  
    true_labels = inputs[:,(-1-int(is_eos))].view(-1)

    exact_matches = (predicted_labels==true_labels)
    count_exact_matches = exact_matches.sum() #tensor

    for key,value in labels_tokens_id.items():
      position_key = (true_labels==value)
      number_samples_key = (true_labels==value).sum().item()
      dict_metrics[f'number real samples_{key}'] += number_samples_key

      exact_matches_key = position_key & exact_matches
      count_exact_matches_key = exact_matches_key.sum().item()
      dict_metrics[f'true matches_{key}']+= count_exact_matches_key

      count_predicted_key = (predicted_labels==value).sum().item()
      dict_metrics[f'number predicted samples_{key}'] += count_predicted_key

    return count_exact_matches

def eval_model_classification_perf(model,dataset,data_collator,labels_tokens_id={'0' : 17, '1': 18},is_eos=True,batch_size=4,proportion_to_evaluate=1.):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  #Number of sentences to select among the evaluation dataset
  n_selected = int(proportion_to_evaluate * len(dataset))
  evaluated_dataset = dataset.select(range(n_selected))

  # Create DataLoader
  dataloader = DataLoader(evaluated_dataset, batch_size=batch_size, collate_fn=data_collator)

  # Evaluation loop
  model.eval()
  total_matches = 0

  #Dictionary of metrics for the considered labels
  dict_metrics = {'Global accuracy':0.}
  for key, value in labels_tokens_id.items():
    dict_metrics[f'recall_{key}'] = 0
    dict_metrics[f'precision_{key}'] = 0
    dict_metrics[f'f1-score_{key}'] = 0
    dict_metrics[f'true matches_{key}'] = 0
    dict_metrics[f'number real samples_{key}'] = 0
    dict_metrics[f'number predicted samples_{key}'] = 0


  dataset_predicted_labels = []
  with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        inputs = batch['input_ids'].to(device)
        #inputs = batch['input_ids'] #device_map="auto" handles automatically to which hardware to attribute the inputs


        outputs = model(input_ids=inputs)

        logits = outputs.logits
        _,_,vocab_size = logits.size()
        predicted_labels = logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size).argmax(dim=1)

        acc = compute_metrics_eval(predicted_labels, inputs,labels_tokens_id,dict_metrics,is_eos=is_eos)
        total_matches += acc.item()
        dataset_predicted_labels.append(predicted_labels)

    accuracy = total_matches / len(evaluated_dataset)

    for key in labels_tokens_id.keys():
      dict_metrics[f'recall_{key}'] = dict_metrics[f'true matches_{key}'] / dict_metrics[f'number real samples_{key}']
      dict_metrics[f'precision_{key}'] = 0  if dict_metrics[f'number predicted samples_{key}']==0  else dict_metrics[f'true matches_{key}'] / dict_metrics[f'number predicted samples_{key}'] 
      dict_metrics[f'f1-score_{key}'] = 0 if (dict_metrics[f'recall_{key}'] + dict_metrics[f'precision_{key}'])==0 else 2 * dict_metrics[f'recall_{key}'] * dict_metrics[f'precision_{key}'] / (dict_metrics[f'recall_{key}'] + dict_metrics[f'precision_{key}'])


    logger.info(f"\nAverage accuracy of the model on the evaluated dataset of {n_selected} samples : {accuracy}\n")
    dict_metrics['Global accuracy'] = accuracy

    dataset_predicted_labels_tensor = torch.cat(dataset_predicted_labels, dim=0)

    return dict_metrics, dataset_predicted_labels_tensor


def main_evaluation(config):

    
    cfg = LLMLoadConfig.autoconfig(config)
     
    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
     
    #Process the test dataset
    dataset_tokenized = process_dataset(cfg,split=cfg.split,tokenizer=tokenizer) 
    
    #Load the local model
    model = get_model(cfg)
    print(model)
    
    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = tokenizer.get_vocab()
    labels = np.unique(np.array(dataset_tokenized["token_labels"]))
    keys_labels = set(labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}


    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    #Get model performance metrics dictionary on the classification task for the test split of the selected dataset
    logger.info(f"Evaluation of the model {cfg.model_name} on the test split of the dataset {cfg.dataset_name}")
    dict_metrics, dataset_predicted_labels_tensor = eval_model_classification_perf(model,dataset_tokenized,data_collator,labels_tokens_id,**cfg.task_args)
    
    #Save the dictionary of metrics and the predicted labels
    dir_to_save_model_results = os.path.join(cfg.dir_to_save_metrics,cfg.split)
    dir_to_save_predictions = os.path.join(dir_to_save_model_results,"predicted_labels")

    #Create the directory if it does not exist
    if not os.path.exists(dir_to_save_predictions):
            os.makedirs(dir_to_save_predictions)
    
    file_to_save_metrics = os.path.join(dir_to_save_model_results,f"{cfg.model_name}_{cfg.dataset_name}.json") 
    file_to_save_predicted_labels = os.path.join(dir_to_save_predictions,f"{cfg.model_name}_{cfg.dataset_name}.pt")
    with open(file_to_save_metrics, 'w') as file:
        json.dump(dict_metrics, file, indent=4)

    print(dataset_predicted_labels_tensor)
    torch.save(dataset_predicted_labels_tensor, file_to_save_predicted_labels) 
    




