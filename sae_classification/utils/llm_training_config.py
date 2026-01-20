from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser

from .paths import *
from .datasets_parameters import DICT_CATEGORIES, DICT_DATASET_ALIAS


class LLMTrainerConfig:
    """
    Configuration for fine-tuning an  LLM backbone on a text classification dataset.
    If the LLM has a decoder-only architecture, we use a template to specify the possible labels to choose from.
    
    Args:
        model_name (str)
        model_path (str)
        dataset_name (str)
        imbalanced_classes (bool) : True if want to take to into account that categories are imbalanced in the dataset for the classification fine-tuning 
        quantized (bool) : Load the the model weights in 8-bit

        eos (bool) : does the model adds end of sentence token
        
        training_args (to use directly in TrainingArguments and pass this to the transformer Trainer)
    
    """
    
    model_name: str = "pythia_14m"
    dataset_name: str = "imdb"
    quantized: bool = False 
    imbalanced_classes: bool = False

    training_args: dict = {}

    # Does the tokenizer add an end-of-sentence token after processing the sentence ?
    eos: bool = False

    model_path: str 
    model_path_pre_trained: str
    dataset_path: str

    match_label_category: dict = {}

    
    def __init__(self,
                 model_name: str,
                 model_path: str,
                 dataset_name: str,
                 eos: bool,
                 imbalanced_classes: bool,
                 quantized: bool,
                 training_args: dict):

        ############### MODEL Loading ########################
        
     
        self.model_name = model_name
        self.model_path = model_path
        self.model_path_pre_trained = model_path
        self.quantized = quantized
        
        #################### DATASET Loading ###################################

        if dataset_name in list(DICT_CATEGORIES.keys()):
            self.match_label_category = DICT_CATEGORIES[dataset_name]
        else:
             raise ValueError(f"The dataset name provided {dataset_name} is not yet managed. Available datasets are {list(DICT_CATEGORIES.keys())}")
        
        
        self.dataset_name = dataset_name
        self.imbalanced_classes = imbalanced_classes

        self.dataset_path = os.path.join(PATH_LOCAL_DATASET, dataset_name)
        self.dataset_in_local = os.path.exists(self.dataset_path) and os.path.isdir(self.dataset_path)
        self.official_name_repo = DICT_DATASET_ALIAS[self.dataset_name]
        
            
        ################### TOKENIZER Loading #############################
        
        self.tokenizer_path = self.model_path_pre_trained
        self.eos = eos

        
        #If no training_args provided, we go for the defaults
        self.training_args = dict(
                output_dir=os.path.join(PATH_CHECKPOINTS, f"{model_name}/{dataset_name}"),
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                lr_scheduler_type   = "cosine_with_min_lr",
                warmup_steps=500,
                max_grad_norm = 1.0,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=1,
                max_steps=7000,  
                weight_decay=0.01,
                save_strategy = "steps",
                save_steps=5000,
                report_to="wandb",
                run_name=f'fine_tune_{dataset_name}',
                logging_strategy="steps",
                logging_steps=50,
                seed=42,
                data_seed=42
            )
        
        for key, value in training_args.items():
            if not isinstance(key, str):
                raise ValueError(f"training_args.{key} key is not a string.")

            if isinstance(value, str):
                try:
                    value = float(value)
                    if int(value) == value:
                        value = int(value)
                except ValueError:
                    pass

                if value == "false":
                    value = False

                if value == "true":
                    value = True

            self.training_args[key] = value
        

    
    def __repr__(self) -> str:
        
        result=f"""LLMTrainerConfig :
        
        Model used : {self.model_name}
        Dataset used : {self.dataset_name}
        
        Training Arguments passed to the Trainer :"""
        
            
        for key in sorted(self.training_args):
            result += f"\n    - {key}: {self.training_args[key]}"

        return result
    
    
    @classmethod
    def autoconfig(cls,name):
        
        if isinstance(name,Path):
            logger.info(f"`file`: {name} is a valid path, building from it.")
            return cls.from_config_file(name)
        
        elif Path(name).is_file():
            logger.info(f"Autoconfig `name`: {name} is a valid path, building from it.")
            return cls.from_config_file(name)
        
        elif (PATH_CONFIG / name).exists():
            logger.info(
                f"Autoconfig `name`: {name} is a config file in the config folder, building from it."
            )
            return cls.from_config_file(PATH_CONFIG / f"{name}")
        
        elif (PATH_CONFIG / f"{name}.cfg").exists():
            logger.info(
                f"Autoconfig `name`: {name}.cfg is a config file in the config folder, building from it."
            )
            return cls.from_config_file(PATH_CONFIG / f"{name}.cfg")
        
        else: 
            raise TypeError("The config file for the LLM classifier fine-tuning is not found")
        
        
    @classmethod
    def from_config_file(cls,path):
        
        if isinstance(path,str):
            path=Path(path)
        
        parser = ConfigParser()
        parser.read(path)
        
        #Read the config file
        if "main" not in parser:
            raise ValueError("Your config should contain a 'main' section.")
        if "model" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'model' entry."
            )
        if "model_path" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'model_path' entry."
            )
        if "dataset" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'dataset' entry."
            )
        if "eos" not in parser["main"]:
            eos = True
        else:
            eos = (parser["main"]["model"].lower()=='true')

        if "imbalanced_classes" not in parser["main"]:
            imbalanced_classes = False
        else:
            imbalanced_classes = (parser["main"]["imbalanced_classes"].lower()=='true')

        if "quantized" not in parser["main"]:
            quantized=False
        else:
            quantized=(parser["main"]["quantized"].lower()=='true')

             
        model_name = parser["main"]["model"]
        model_path = parser["main"]["model_path"]
        dataset_name = parser["main"]["dataset"]
        
        if "training_args" in parser:
            training_args = parser["training_args"]
        else:
            training_args = {}
            logger.info(
                "No 'training_args' section found in your training config, using default values."
            )
        
        return cls(
            model_name=model_name,
            model_path=model_path,
            dataset_name=dataset_name,
            eos=eos,
            imbalanced_classes=imbalanced_classes,
            quantized=quantized,
            training_args=training_args
        )
