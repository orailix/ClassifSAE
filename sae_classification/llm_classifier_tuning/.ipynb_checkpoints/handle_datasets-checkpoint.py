from ..utils import LLMTrainerConfig, LLMLoadConfig
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
from typing import Union
from loguru import logger
import os

import datasets


TEMPLATE = '{example}{statement}\n\n{options} \n'

def format_template(
    data: dict,
    cfg
) -> dict:
  """Format prompt for IMDB dataset.

  Args:
    data : sentences from the IMDB dataset
  """
  template = TEMPLATE

  new_format = {
                "example" : "".join(cfg.example),
                "statement" : data['text'],
                "choices" : {
                    "label" : list(cfg.match_label_category.keys()),
                    "category" : list(cfg.match_label_category.values())
                },
                "trueLabel" : data["label"]
                }

  options_details = "OPTIONS:"

  for category, label in zip(new_format["choices"]["category"],new_format["choices"]["label"]):
      #print(f"\n{label}({category})")
      options_details += f"\n{label}({category})"

  #Return the new dict that we map to the original dataset
  if cfg.add_example :
      return {
          "prompt" : template.format(example=new_format['example'],statement=new_format["statement"], options=options_details),
          "labels" : new_format["trueLabel"]
      }
  else:
      return {
          "prompt" : template.format(example='',statement=new_format["statement"], options=options_details),
          "labels" : new_format["trueLabel"]
      }


def format_label(
    data : dict
) -> dict:
  """
   Add label to end of the prompt for IMDB dataset.

   Args:
    data : sentences from the IMDB dataset
  """

  return {
      "prompt" : data["prompt"] + str(data["labels"]),
      "labels" : data["labels"]
      }


def preprocess_tokenization(data,tokenizer):
    output =  tokenizer(data['prompt'], add_special_tokens=True)
    output['token_labels'] = data['labels'].copy()
    return output


# def get_dataset(
#     cfg: Union[LLMTrainerConfig,LLMLoadConfig],
#     split: str
#     ) -> Dataset:
    
#     if split not in ['train','test','unsupervised']:
#         raise ValueError('Split variable should be either "train" or "test" or "unsupervised"')
    
#     work_dataset = load_from_disk(cfg.dataset_path)
#     return work_dataset[split]


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
    tokenized_ds = dataset.map(preprocess_tokenization_fct, batched=True)
    return tokenized_ds.select_columns(["input_ids", "attention_mask","token_labels"]).shuffle(seed=0)


def process_dataset(
    cfg: Union[LLMTrainerConfig,LLMLoadConfig],
    split: str,
    tokenizer: AutoTokenizer,
    add_template: bool = True
    ) -> Dataset:
    
    if split not in ['train','test','unsupervised']:
        raise ValueError('Split variable should be either "train" or "test" or "unsupervised"')
    
    #If dataset_path contains paths to already tokenized version of this dataset with this tokenizer
    name_split = f"{split}_template" if add_template else split
    
    # if name_split in cfg.dataset_path_tokenized:
    #     dataset_tokenized = load_from_disk(cfg.dataset_path_tokenized[name_split])
    #     return dataset_tokenized
    
    #Otherwise we tokenize the dataset
    work_dataset = load_from_disk(cfg.dataset_path)
    work_dataset_split = work_dataset[split]

    #We format the dataset in case where this is the jigsaw dataset
    if (cfg.dataset_name == "jigsaw_toxic_comment") :  
        # The features to keep
        features_to_keep = ['text', 'label']
        work_dataset_split = work_dataset_split.map(lambda example: example, remove_columns=[col for col in work_dataset_split.column_names if col not in features_to_keep])
        print(work_dataset_split)

    #If we want to add the template format
    if add_template:
    
        #Add the template prompt at the end of each statement
        dataset_template = template_dataset(work_dataset_split,cfg)

        
        #Add the true label at the end of the Template
        dataset_labelled = add_label_dataset(dataset_template)

        print("Première phrase : ",dataset_labelled[0])
        print("Deuxième phrase : ",dataset_labelled[1])
        
         #Tokenize the dataset
        logger.info("Tokenization of the dataset")
        dataset_tokenized = tokenize_dataset(dataset_labelled, tokenizer)
        
        
    else:
        
        #Temporary code lines because probably specific to IMDB
        work_dataset_split = work_dataset_split.rename_column('text', 'prompt')
        work_dataset_split = work_dataset_split.rename_column('label', 'labels')
        logger.info("Tokenization of the dataset")
        dataset_tokenized = tokenize_dataset(work_dataset_split, tokenizer)

  
   
    #We saved the tokenized version of the dataset so that we can re-use it
    path_to_saved_ds = os.path.join(cfg.dir_to_save_tokenized_ds,f"{cfg.dataset_name}_{cfg.model_name}_{name_split}")
    dataset_tokenized.save_to_disk(path_to_saved_ds)
    
    return dataset_tokenized
    