from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser
import re
import torch
import ast

from .paths import *
from .datasets_parameters import DICT_CATEGORIES

class ClassifierConfig:
    """
    Configuration for loading and training classifier at each layer of the LLM
    
    Args:
        hook_layers_names : list of the hook layer names where to extract the activations that will be used as features for the classifier
        
        device : device on which train the classifier
        
        dim_activation : dimension of the extracted internal model activations
        
        directory_to_save (str) : directory where to save the parameters of the classifiers
        
        training_args : dict
    
    
    """
    
    hook_layers_names: list = ['blocks.1.hook_resid_pre',
                               'blocks.2.hook_resid_pre',
                               'blocks.3.hook_resid_pre',
                               'blocks.4.hook_resid_pre',
                               'blocks.5.hook_resid_pre',
                               'blocks.5.hook_resid_post']
    
    device: str = 'cuda'
    
    dim_activation: int = 1024
    
    directory_to_save = ''
    
    
    def __init__(self,
                 hook_layers_names: str,
                 dim_activation: int,
                 training_args: dict
                 ):
        
        self.hook_layers_names = hook_layers_names
        
        if torch.backends.mps.is_available(): #For Apple fans
            self.device = "mps"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.dim_activation = dim_activation
        
        self.directory_to_save = PATH_CLASSIFIER
        
        #If no training_args provided, we go for the defaults given the specified task
        self.training_args = dict(
                    batch_size=4,
                    learning_rate=0.001,
                    num_epochs=10,
                    hidden_size=512,
                    temperature=2
                )
        
        for key, value in training_args.items():
            if not isinstance(key, str):
                raise ValueError(f"task_args.{key} key is not a string.")

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
            raise TypeError("The config file for the Classifier loading and training is not found")
        
    
    @classmethod
    def from_config_file(cls,path):
        
        if isinstance(path,str):
            path=Path(path)
        
        parser = ConfigParser()
        parser.read(path)
        
        #Read the config file
        if "main" not in parser:
            raise ValueError("Your config should contain a 'main' section.")
        if "hook_layers_names" not in parser["main"]:
            hook_layers_names = ['blocks.1.hook_resid_pre',
                               'blocks.2.hook_resid_pre',
                               'blocks.3.hook_resid_pre',
                               'blocks.4.hook_resid_pre',
                               'blocks.5.hook_resid_pre',
                               'blocks.5.hook_resid_post']
        else:
            hook_layers_names = ast.literal_eval(parser["main"]["hook_layers_names"])
        
        if "dim_activation" not in parser["main"]:
            dim_activation = 1024
        else:
            dim_activation = int(parser["main"]["dim_activation"])
        
            
        
        if "training_args" in parser:
            training_args = parser["training_args"]
        else:
            training_args = {}
            logger.info(
                "No 'training_args' section found in your loading config, using default values."
            )
        
        return cls(
            hook_layers_names=hook_layers_names,
            dim_activation=dim_activation,
            training_args=training_args
        )
        
