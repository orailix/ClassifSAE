from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser
import re
import torch


from .paths import *

class DRMethodsConfig:
    """
    Configuration for training alternative concepts extraction methods to ours. 
    
    Args:
        method_name (str) : name of the method. So far, concept_shap or ica 

        hook_layer (int) : depth at which the activations extraction is resolved

        Similarly to the training of the SAE, we used the cached activations to train our benchmarks. To retrieve the activations, we need to know the split, model and dataset useed to generate the targeted activations
        split (str) : either "train" or "test". Logicaly, it should be "train"
        model_name (str) : Name of the language model used to generate the activations
        dataset_name (str) : Name of the dataset used to generate the activations
        
        dr_methods_args (dict)
    
    """
    method_name: str = "ica"
    
    dr_methods_args: dict = {}
    

    
    def __init__(self,
                 method_name: str,
                 hook_layer: int = 4,
                 split: str= "train",
                 model_name: str = "",
                 dataset_name: str ="",
                 dr_methods_args: dict = {}):
    

        self.method_name = method_name
        self.hook_name = f"blocks.{hook_layer}.hook_resid_pre"
        self.hook_layer = hook_layer

        supported_approaches = ["ica","concept_shap"]
        assert method_name in supported_approaches, f"Error: The method {method_name} is not supported in the benchmarks. Currently the only supported methods are {supported_approaches}"
        
        ######## Method Loading ###############

        if method_name == 'concept_shap':
            dir_layer_activations = os.path.join(PATH_CACHED_ACT, model_name, dataset_name, split, "with_labels", self.hook_name)
        else:
            dir_layer_activations = os.path.join(PATH_CACHED_ACT, model_name, dataset_name, split, "without_labels", self.hook_name)

        if not (os.path.exists(dir_layer_activations) and os.path.isdir(dir_layer_activations)):
                raise ValueError(f"The model name provided is either not present in {PATH_CHECKPOINTS} or its activations on the dataset provided were not cached at the hook point {self.hook_name}.")

        numbers_steps = []
        version_names = {}
        for version in os.listdir(dir_layer_activations):
            if os.path.isdir(os.path.join(dir_layer_activations, version)) :
    
                number_steps = int(version)
                numbers_steps.append(number_steps)
                version_names[number_steps] = version
        if numbers_steps==[]:
            raise ValueError(f"No activations available in {dir_layer_activations}.")

        numbers_steps.sort(reverse=True)
        max_number_steps = numbers_steps[0]
        selected_version = version_names[max_number_steps]

        #Path to fit the model
        self.activations_path = os.path.join(dir_layer_activations,selected_version)
        
        #Create the directory where to save the file if it does not exist
        if not os.path.exists(os.path.join(PATH_DR_METHODS, split)):
            os.makedirs(os.path.join(PATH_DR_METHODS, split))
        
        self.path_to_dr_methods = os.path.join(PATH_DR_METHODS, split,f'{method_name}_{model_name}_{dataset_name}_layer_{hook_layer}')

        # #Path to retrieve activations from the test set with the ground truth labels (ActivationDataset)
        # self.dir_acts_with_labels = PATH_ACTS_LABELS 

        #Path where to save reconstruction metrics 
        self.metrics_reconstruction = os.path.join(PATH_DR_METHODS_METRICS,f'{method_name}_{model_name}_{dataset_name}_layer_{hook_layer}')
        #Create the directory where to save the reconstruction results if it does not exist
        if not os.path.exists(self.metrics_reconstruction):
            os.makedirs(self.metrics_reconstruction)

        if method_name == 'pca':
            self.dr_methods_args = dict(
                n_components = 5,
                whiten = True,
                svd_solver = 'auto',
                random_state=42
            )
        elif method_name == 'ica':
           self.dr_methods_args = dict(
                n_components = 5,
                whiten = True,
                algorithm = 'parallel',
                fun='logcosh',
                random_state=42
            )
        elif method_name == 'concept_shap':
           self.dr_methods_args = dict(
                l1 = 0.001,
                l2 = 0.001,
                topk = 10,
                batch_size=32,
                epochs=3,
                loss_reg_epoch=2,
                n_concepts=10,
                hidden_dim=512,
                thres=0.1
            ) 
        else:
            raise ValueError(f"The only supported dr methods so far are ICA and ConceptSHAP")
        
        for key, value in dr_methods_args.items():
            if not isinstance(key, str):
                raise ValueError(f"dr_methods_args.{key} key is not a string.")

            if isinstance(value, str):

                #We allow the input of lists of parameters. In that case, we launch multpile runs to test all combinations of the specified hyperparameters
                if value.startswith('[') and value.endswith(']'):
                    value = ast.literal_eval(value)
                
                else:
                    try:
                        value = float(value)
                        if int(value) == value:
                            value = int(value)
                    except ValueError:
                        pass

                    if isinstance(value, str):
                        if value.lower() == "false":
                            value = False
                        elif value.lower() == "true":
                            value = True

            self.dr_methods_args[key] = value
    
    
    def __repr__(self) -> str:
        
        result=f"""DRMethodsConfig :
        
        Method name : {self.method_name}
        
        
       Additional arguments passed  :"""
        
            
        for key in sorted(self.dr_methods_args):
            result += f"\n    - {key}: {self.dr_methods_args[key]}"

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
            raise TypeError("The config file for the DR methods fitting and evaluation is not found")
        
        
    @classmethod
    def from_config_file(cls,path):
        
        if isinstance(path,str):
            path=Path(path)
        
        parser = ConfigParser()
        parser.read(path)
        
        #Read the config file
        if "main" not in parser:
            raise ValueError("Your config should contain a 'main' section.")
        if "method" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'method' entry."
            )
        if "model" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'model' entry."
            )
        if "dataset" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'dataset' entry."
            )
       
            
        method_name = parser["main"]["method"]
        model_name = parser["main"]["model"]
        dataset_name = parser["main"]["dataset"]


        if "split" not in parser["main"]:
            split = "train"
        else:
            split = parser["main"]["split"]

        if "hook_layer" not in parser["main"]:
            hook_layer = 4
        else:
            hook_layer = int(parser["main"]["hook_layer"])
        
        
        if "dr_methods_args" in parser:
            dr_methods_args = parser["dr_methods_args"]
        else:
            dr_methods_args = {}
            logger.info(
                "No 'dr_methods_args' section found in your config, using default values."
            )

        
        return cls(
            method_name=method_name,
            split=split,
            hook_layer=hook_layer,
            model_name=model_name,
            dataset_name=dataset_name,
            dr_methods_args=dr_methods_args
        )