from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from loguru import logger
import torch
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from tqdm import tqdm


from ..utils import LLMLoadConfig
from .models import get_model
from .handle_datasets import process_dataset


# Compute metrics of model performance on the batch
def compute_metrics_eval(predicted_labels, true_labels, labels_tokens_id, dict_metrics, eos=True):
  
    exact_matches = (predicted_labels==true_labels)
    count_exact_matches = exact_matches.sum() #tensor

    for _,value in labels_tokens_id.items():
      position_key = (true_labels==int(value))
      number_samples_key = (true_labels==int(value)).sum().item()
      dict_metrics[f'number real samples_{value}'] += number_samples_key

      exact_matches_key = position_key & exact_matches
      count_exact_matches_key = exact_matches_key.sum().item()
      dict_metrics[f'true matches_{value}']+= count_exact_matches_key

      count_predicted_key = (predicted_labels==int(value)).sum().item()
      dict_metrics[f'number predicted samples_{value}'] += count_predicted_key

    return count_exact_matches

def eval_model_classification_perf(model,dataset,data_collator,labels_tokens_id={ 17: 0, 18: 1 },eos=True,batch_size=4,proportion_to_evaluate=1.):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # Number of sentences to select among the evaluation dataset
  n_selected = int(proportion_to_evaluate * len(dataset))
  evaluated_dataset = dataset.select(range(n_selected))

  # Create DataLoader
  dataloader = DataLoader(evaluated_dataset, batch_size=batch_size, collate_fn=data_collator,shuffle=False)

  # Evaluation loop
  model.eval()
  total_matches = 0

  # Dictionary of metrics for the considered labels
  dict_metrics = {'Global accuracy':0.}
  for _, value in labels_tokens_id.items():
    dict_metrics[f'recall_{value}'] = 0
    dict_metrics[f'precision_{value}'] = 0
    dict_metrics[f'f1-score_{value}'] = 0
    dict_metrics[f'true matches_{value}'] = 0
    dict_metrics[f'number real samples_{value}'] = 0
    dict_metrics[f'number predicted samples_{value}'] = 0


  # Filter predictions based on the logits corresponding to an accepted answer
  ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                          key=lambda kv: kv[1])]

  dataset_predicted_labels = []
  with torch.no_grad():
    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):

        inputs = batch['input_ids'].to(device)
        true_labels = batch["true_label"].to(device)
        outputs = model(input_ids=inputs)

        logits = outputs.logits
        _,_,vocab_size = logits.size()
        # Given the provided template, the ground-truth prediction token is located at position (-1-int(eos)), we therefore focus on the hidden state of the token preceding the latter. 
        logits_next_token = logits[:,(-2-int(eos))].contiguous().view(-1, vocab_size)
        logits_next_token = logits_next_token[:,ordered_old_idxs]  #shape : (bs,nb_classes)
        predicted_labels = logits_next_token.argmax(dim=1)

        acc = compute_metrics_eval(predicted_labels, true_labels,labels_tokens_id,dict_metrics,eos=eos)
        total_matches += acc.item()
        dataset_predicted_labels.append(predicted_labels.cpu())

    accuracy = total_matches / len(evaluated_dataset)

    for value in labels_tokens_id.values():
      dict_metrics[f'recall_{value}'] = dict_metrics[f'true matches_{value}'] / dict_metrics[f'number real samples_{value}']
      dict_metrics[f'precision_{value}'] = 0  if dict_metrics[f'number predicted samples_{value}']==0  else dict_metrics[f'true matches_{value}'] / dict_metrics[f'number predicted samples_{value}'] 
      dict_metrics[f'f1-score_{value}'] = 0 if (dict_metrics[f'recall_{value}'] + dict_metrics[f'precision_{value}'])==0 else 2 * dict_metrics[f'recall_{value}'] * dict_metrics[f'precision_{value}'] / (dict_metrics[f'recall_{value}'] + dict_metrics[f'precision_{value}'])


    logger.info(f"\nAverage accuracy of the model on the evaluated dataset of {n_selected} samples : {accuracy}\n")
    dict_metrics['Global accuracy'] = accuracy

    dataset_predicted_labels_tensor = torch.cat(dataset_predicted_labels, dim=0)

    return dict_metrics, dataset_predicted_labels_tensor


def main_evaluation(config):

    
    cfg = LLMLoadConfig.autoconfig(config)
     
    #Load tokenizer
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
    
    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = tokenizer.get_vocab()
    labels_dataset = dataset_tokenized["true_label"]
    unique_labels = np.unique(np.array(labels_dataset))
    keys_labels = set(unique_labels)
    labels_tokens_id = { vocab[str(key)] : key for key in keys_labels if str(key) in vocab}
    # Example for IMDB : { 17: 0, 18: 1 } (0:Negative /1: Positive)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    # Get model performance metrics dictionary on the classification task for the test split of the selected dataset
    logger.info(f"Evaluation of the model {cfg.model_name} on the {cfg.split} split of the dataset {cfg.dataset_name}")
    dict_metrics, dataset_predicted_labels_tensor = eval_model_classification_perf(model,dataset_tokenized,data_collator,labels_tokens_id,**cfg.task_args)
    
    # Save the dictionary of metrics and the predicted labels (for the joint training of the classifier and the SAE)
    dir_to_save_model_results = os.path.join(cfg.dir_to_save_metrics,cfg.split)
    dir_to_save_predictions = os.path.join(dir_to_save_model_results,"predicted_labels")

    # Create the directory if it does not exist
    if not os.path.exists(dir_to_save_predictions):
            os.makedirs(dir_to_save_predictions)
    
    file_to_save_metrics = os.path.join(dir_to_save_model_results,f"{cfg.model_name}_{cfg.dataset_name}.json") 
    file_to_save_predicted_labels = os.path.join(dir_to_save_predictions,f"{cfg.model_name}_{cfg.dataset_name}.pt")
    with open(file_to_save_metrics, 'w') as file:
        json.dump(dict_metrics, file, indent=4)

    torch.save(dataset_predicted_labels_tensor, file_to_save_predicted_labels) 
    




