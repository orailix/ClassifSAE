from ..utils import LLMTrainerConfig, LLMLoadConfig
from datasets import Dataset, load_from_disk, load_dataset
from transformers import AutoTokenizer
from typing import Union
from loguru import logger
import os


TEMPLATE = '{example}{statement}\n\n{options} \n'

def format_template(
    data: dict,
    cfg
) -> dict:
  """Format prompt for the sentences of the datasets.

  Args:
    data : sentences from the dataset
  """
  template = TEMPLATE

  new_format = {
                "example" : "".join(cfg.example),
                "statement" : data['text'],
                "choices" : {
                    "label" : list(cfg.match_label_category.keys()),
                    "category" : list(cfg.match_label_category.values())
                },
                "label" : data["label"]
                }

  options_details = "OPTIONS:"

  for category, label in zip(new_format["choices"]["category"],new_format["choices"]["label"]):
      options_details += f"\n{label}({category})"

  #Return the new dict that we map to the original dataset
  if cfg.add_example :
      return {
          "prompt" : template.format(example=new_format['example'],statement=new_format["statement"], options=options_details),
          "label" : new_format["label"]
      }
  else:
      return {
          "prompt" : template.format(example='',statement=new_format["statement"], options=options_details),
          "label" : new_format["label"]
      }


def format_label(
    data : dict
) -> dict:
  """
   Add the ground-truth label to end of the prompt for the sentence to classify.

   Args:
    data : sentences from the dataset
  """

  return {
      "prompt" : data["prompt"] + str(data["label"]),
      "label" : data["label"]
      }


def preprocess_tokenization(data,tokenizer):
    output =  tokenizer(data['prompt'], add_special_tokens=True)
    output['true_label'] = data['label'].copy()
    return output



def template_dataset(
    dataset: Dataset,
    cfg
) -> Dataset:

    formatting_fct = lambda args: format_template(args,cfg=cfg)
    return dataset.map(formatting_fct, remove_columns=dataset.column_names)



def add_label_dataset(
     dataset: Dataset
) -> Dataset:
    
    add_label_fct = lambda args: format_label(args)
    return  dataset.map(add_label_fct)


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer
) -> Dataset:
    
    preprocess_tokenization_fct = lambda args: preprocess_tokenization(args,tokenizer=tokenizer)
    tokenized_ds = dataset.map(preprocess_tokenization_fct, batched=True,load_from_cache_file=False)
    return tokenized_ds.select_columns(["input_ids", "attention_mask","true_label"])


def process_dataset(
    cfg: Union[LLMTrainerConfig,LLMLoadConfig],
    split: str,
    tokenizer: AutoTokenizer,
    add_template: bool = True
    ) -> Dataset:
    
    if split not in ['train','test','unsupervised']:
        raise ValueError('Split variable should be either "train" or "test" or "unsupervised"')
    
    
    # We tokenize the dataset
    # Try to load the dataset locally
    if cfg.dataset_in_local:
        work_dataset = load_from_disk(cfg.dataset_path)
    else:
        work_dataset = load_dataset(cfg.dataset_name)
        # Save the dataset locally
        work_dataset.save_to_disk(cfg.dataset_path)
    work_dataset_split = work_dataset[split]


    # If we want to add the classification template format
    if add_template:
    
        #Add the template prompt at the end of each statement
        dataset_template = template_dataset(work_dataset_split,cfg)
        
        #Add the true label at the end of the Template
        dataset_labelled = add_label_dataset(dataset_template)

        print("First sentence : ",dataset_labelled[0])
        
        #Tokenize the dataset
        logger.info("Tokenization of the dataset")
        dataset_tokenized = tokenize_dataset(dataset_labelled, tokenizer)
        
        
    else:
        
        work_dataset_split = work_dataset_split.rename_column('text', 'prompt')
        logger.info("Tokenization of the dataset")
        dataset_tokenized = tokenize_dataset(work_dataset_split, tokenizer)

    
    return dataset_tokenized
    