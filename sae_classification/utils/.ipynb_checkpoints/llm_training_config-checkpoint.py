from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser

from .paths import *
from .datasets_parameters import DICT_CATEGORIES


class LLMTrainerConfig:
    """
    Configuration for training a LLM model on a dataset.
    
    Args:
        model_name (str)
        dataset_name (str)
        
        training_args (to use directly in TrainingArguments and pass this to the transformer Trainer)
    
    """
    
    model_name: str = "pythia_14m"
    dataset_name: str = "imdb"
    training_args: dict = {}

    is_eos: bool = True

    tuning: str = "vanilla"
    quantized: bool = False

    model_path: str 
    model_path_pre_trained: str
    dataset_path_tokenized: dict
    dataset_path: str

    match_label_category: dict = {}

    dir_to_save_tokenized_ds : str


    
    def __init__(self,
                 model_name: str,
                 model_path: str,
                 dataset_name: str,
                 is_eos: bool,
                 tuning: str,
                 quantized: bool,
                 training_args: dict):

        ############### MODEL Loading ########################
        
        # self.model_path = os.path.join(PATH_LOCAL_MODEL, model_name)
        # if os.path.exists(self.model_path) and os.path.isdir(self.model_path):
        #     self.model_name = model_name
        # else:
        #     raise ValueError(f"The model name provided is not present in {PATH_LOCAL_MODEL}. Models saved locally are {os.listdir(PATH_LOCAL_MODEL)}")
        self.model_name = model_name
        self.model_path = model_path
        self.model_path_pre_trained = model_path
        

        #################### DATASET Loading ###################################

        if dataset_name in list(DICT_CATEGORIES.keys()):
            self.match_label_category = DICT_CATEGORIES[dataset_name]
        else:
             raise ValueError(f"The dataset name provided {dataset_name} is not yet managed. Available datasets are {list(DICT_CATEGORIES.keys())}")
        
        #Check if the dataset has already been tokenized with this tokenizer and saved locally

        #First, creates the folder if it does not exist
        if not os.path.exists(PATH_LOCAL_DATASET_TOKENIZED):
            os.makedirs(PATH_LOCAL_DATASET_TOKENIZED)
        
        all_tokenized_ds = os.listdir(PATH_LOCAL_DATASET_TOKENIZED)
        prefix = f"{dataset_name}_{model_name}"
        matching_directories = {}
        for entry in all_tokenized_ds:
            if os.path.isdir(os.path.join(PATH_LOCAL_DATASET_TOKENIZED, entry)) and entry.startswith(prefix):
                suffix = entry[len(prefix) + 1:]  # Extract the suffix(it will be 'train','test','unsupervised','train_template','test_template','unsupervised_template')
                matching_directories[suffix] = os.path.join(PATH_LOCAL_DATASET_TOKENIZED, entry)

        
        self.dataset_path_tokenized = matching_directories
        self.dataset_name = dataset_name
            
        #In case, there is no tokenized version of the split of the dataset that we want, we provide the path
        #to the original dataset for tokenization
        self.dataset_path = os.path.join(PATH_LOCAL_DATASET, dataset_name)
        self.dir_to_save_tokenized_ds = PATH_LOCAL_DATASET_TOKENIZED
        dataset_is_present = os.path.exists(self.dataset_path) and os.path.isdir(self.dataset_path)
        if (not dataset_is_present) and (self.dataset_path_tokenized=={}) :
            raise ValueError(f"The dataset name provided is neither present in {PATH_LOCAL_DATASET} nor in {PATH_LOCAL_DATASET_TOKENIZED}. Datasets saved locally are {os.listdir(PATH_LOCAL_DATASET)}")

        ################### TOKENIZER Loading #############################
        
        # self.tokenizer_path = os.path.join(PATH_LOCAL_TOKENIZER, tokenizer_name)
        # if os.path.exists(self.tokenizer_path) and os.path.isdir(self.tokenizer_path):
        #     self.tokenizer_name = tokenizer_name
        # else:
        #     raise ValueError(f"The tokenizer name provided is not present in {PATH_LOCAL_TOKENIZER}. Tokenizers saved locally are {os.listdir(PATH_LOCAL_TOKENIZER)}")

        self.tokenizer_path = model_path
        self.is_eos = is_eos
        self.tuning = tuning
        self.quantized = quantized

        #There is no need to add previous examples for training
        self.add_example = False
        self.example = []
        
        #If no training_args provided, we go for the defaults
        self.training_args = dict(
                output_dir=os.path.join(PATH_CHECKPOINTS, f"{model_name}/{dataset_name}"),
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=1,
                max_steps=7000,  # Train for only 7000 steps (batches)
                weight_decay=0.01,
                save_strategy = "steps",
                save_steps=5000,
                report_to="wandb",
                run_name=f'fine_tune_{dataset_name}',
                logging_steps=200,
            
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
            raise TypeError("The config file for the training is not found")
        
        
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
        if "is_eos" not in parser["main"]:
            is_eos = True
        else:
            is_eos = (parser["main"]["model"].lower()=='true')
        if "tuning" not in parser["main"]:
            tuning = "vanilla"
        else:
            tuning = parser["main"]["tuning"]
        if "quantized" not in parser["main"]:
            quantized = False
        else:
            quantized = (parser["main"]["quantized"].lower() == 'true')
     
            
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
            is_eos=is_eos,
            tuning=tuning,
            quantized=quantized,
            training_args=training_args,
        )
