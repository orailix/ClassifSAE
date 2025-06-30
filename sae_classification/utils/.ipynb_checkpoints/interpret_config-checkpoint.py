from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser
import re
import torch


from .paths import *

class InterpretConfig:
    """
    Configuration for interpreting concepts extracted by the different evaluated methods
    
    Args:
        method_name (str) : name of the method. So far, ICA, ConceptSHAP and SAE 
        
        hook_layer (int) : depth at which the activations extraction is resolved

        split (str) : either train or test, wether the method was trained on "trai" or "test"

        model_name (str) : the model the dimension reduction method was trained on

        dataset_name (str) : the dataset the dimension reduction method was trained on
        
        methods_args (dict)
    
    """
    method_name: str = "ica"

    
    methods_args: dict = {}


    
    def __init__(self,
                 method_name: str,
                 split: str = "train",
                 hook_layer: int = 4,
                 model_name:str = "",
                 dataset_name:str = "",
                 sae_name: str = "",
                 checkpoint_version: int = 5000,
                 latest_version: bool = True,
                 methods_args: dict = {}):
    

        self.method_name = method_name
        self.hook_name = f"blocks.{hook_layer}.hook_resid_pre"
        self.hook_layer = hook_layer
        
        ######## Method Loading ###############

        supported_approaches = ["ica","concept_shap","sae"]
        assert method_name in supported_approaches, f"Error: The method {method_name} is not supported in analysis. Currently the only supported methods are {supported_approaches}"

        if method_name in ["ica","concept_shap"]:

        
            #Create the directory where to save the file if it does not exist
            if not os.path.exists(os.path.join(PATH_DR_METHODS, split)):
                os.makedirs(os.path.join(PATH_DR_METHODS, split))
        
            self.path_to_dr_methods = os.path.join(PATH_DR_METHODS, split,f'{method_name}_{model_name}_{dataset_name}_layer_{hook_layer}')

            #Path where to save reconstruction metrics 
            self.metrics_reconstruction = os.path.join(PATH_DR_METHODS,"metrics",f'{method_name}_{model_name}_{dataset_name}_layer_{hook_layer}')
            #Create the directory where to save the reconstruction results if it does not exist
            if not os.path.exists(self.metrics_reconstruction):
                os.makedirs(self.metrics_reconstruction)

        elif method_name in ["sae"]:

            ######## SAE Loading ###############
        
            sae_load_dir = os.path.join(PATH_CHECKPOINT_SAE, f"{sae_name}")
            ## The different versions of the tuned model are saved in folders named 'final_X'
            
            if not (os.path.exists(sae_load_dir) and os.path.isdir(sae_load_dir)):
                raise ValueError(f"The sae name provided is either not present in {PATH_CHECKPOINT_SAE}")
            
            numbers_steps = []
            version_names = {}
            for version in os.listdir(sae_load_dir):
                if os.path.isdir(os.path.join(sae_load_dir, version)) and re.match(r'final_\d+', version):
        
                    number_steps = int(version.split('_')[-1])
                    numbers_steps.append(number_steps)
                    version_names[number_steps] = version
            if numbers_steps==[]:
                raise ValueError(f"No checkpoints available in {sae_load_dir}. Verify that the checkpoints folders are named according to the template 'final_X' with 'X' an integer.")
            
            if latest_version:
                numbers_steps.sort(reverse=True)
                max_number_steps = numbers_steps[0]
                selected_version = version_names[max_number_steps]
                self.checkpoint_version = max_number_steps
            else:
                #Find the model tuned the with closest number of steps to the one provided in checkpoint_version 
                closest_number_steps = min(numbers_steps, key=lambda x: abs(x - checkpoint_version))
                selected_version = version_names[closest_number_steps]
                self.checkpoint_version = checkpoint_version
            
            self.sae_name = f"{sae_name}_{self.checkpoint_version}"
            self.sae_path = os.path.join(sae_load_dir,selected_version)
            self.path_select_features = os.path.join(PATH_SELECT_SAE_FEATURES,self.sae_name)
        

        #Path to retrieve activations from the test set with the ground truth labels (ActivationDataset)
        self.dir_acts_with_labels = PATH_ACTS_LABELS 

        #Default evaluation arguments to pass
        if torch.backends.mps.is_available(): #Apple
                device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            


        if method_name == 'concept_shap':
            self.methods_args = dict(
                n_concepts=10,
                hidden_dim=512,
                thres=0.1
            ) 
        elif method_name == 'sae':
            self.methods_args = dict(
              device=device
            ) 
        

        for key, value in methods_args.items():
            if not isinstance(key, str):
                raise ValueError(f"methods_args.{key} key is not a string.")

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

            self.methods_args[key] = value

            
            
        
    
    
    def __repr__(self) -> str:
        
        result=f"""InterpretConfig :
        
        Method name : {self.method_name}    
        
        Interpretability Arguments passed to the analysis function :"""
        
            
        for key in sorted(self.methods_args):
            result += f"\n    - {key}: {self.methods_args[key]}"

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
      
            
        method_name = parser["main"]["method"]
       

        if "split" not in parser["main"]:
            split = "train"
        else:
            split = parser["main"]["split"]

        if "hook_layer" not in parser["main"]:
            hook_layer = 4
        else:
            hook_layer = int(parser["main"]["hook_layer"])

        if "sae" not in parser["main"]:
            sae_name = ""
        else:
            sae_name = parser["main"]["sae"]

        if "model" not in parser["main"]:
              model_name = ""
        else:    
            model_name = parser["main"]["model"]

        if "dataset" not in parser["main"]:
              dataset_name = ""
        else:    
            dataset_name = parser["main"]["dataset"]
            

        if "version" in parser:
            
            if "checkpoint_version" in parser["version"]:
                checkpoint_version = int(parser["version"]["checkpoint_version"])
            else:
                checkpoint_version = 500
            if "latest_version" in parser["version"]:
                latest_version = (parser["version"]["latest_version"].lower() == 'true')
            else:
                latest_version = True
        else:
            checkpoint_version = 500
            latest_version = True
        
        
        
        if "methods_args" in parser:
            methods_args = parser["methods_args"]
        else:
            methods_args = {}
            logger.info(
                "No 'methods_args' section found in your config, using default values."
            )

        
        
        return cls(
            method_name=method_name,
            split=split,
            hook_layer=hook_layer,
            model_name=model_name,
            dataset_name=dataset_name,
            sae_name=sae_name,
            checkpoint_version=checkpoint_version,
            latest_version=latest_version,
            methods_args=methods_args
        )