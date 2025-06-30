from pathlib import Path
import os
import ast
from loguru import logger
from configparser import ConfigParser
from .paths import *

class BaselineMethodConfig:
    """
    Configuration for training baseline concepts discovery methods : ConceptSHAP and ICA. 
    
    Args:
        method_name (str) : name of the method. So far, concept_shap or ica 

        hook_layer (int) : depth at which the activations extraction is resolved

        Similarly to the training of the SAE, we used the cached activations to train the baseline methods. 
        To retrieve the activations, we need to know which model and dataset were used to generate them.
        model_name (str) : Name of the language model used to generate the activations
        dataset_name (str) : Name of the dataset used to generate the activations
        
        baseline_method_args (dict)
    
    """
    method_name: str = "ica"
    
    baseline_method_args: dict = {}
    
    with_label : bool = True
    
    def __init__(self,
                 method_name: str,
                 hook_layer: int = 4,
                 model_name: str = "",
                 dataset_name: str ="",
                 baseline_method_args: dict = {}):
    

        self.method_name = method_name
        self.hook_name = f"blocks.{hook_layer}.hook_resid_pre"
        self.hook_layer = hook_layer

        supported_approaches = ["ica","concept_shap"]
        assert method_name in supported_approaches, f"Error: The method {method_name} is not supported in the benchmarks. Currently the only supported methods are {supported_approaches}"
        
        ######## Method Loading ###############

        dir_layer_activations = os.path.join(PATH_CACHED_ACT, model_name, dataset_name, "train", "with_labels", self.hook_name)

        # We don't necessarily need the predicted labels if we do not use a supervised approach like ConceptSHAP
        if (not os.path.isdir(dir_layer_activations) or not os.listdir(dir_layer_activations)) and method_name!='concept_shap':
            dir_layer_activations = os.path.join(PATH_CACHED_ACT, model_name, dataset_name, "train", "without_labels", self.hook_name)
            self.with_label = False


        if (not os.path.isdir(dir_layer_activations) or not os.listdir(dir_layer_activations)):
                raise ValueError(f"The model name provided is either not present in {PATH_CHECKPOINTS} or its activations on the dataset provided are not cached at the hook point {self.hook_name}.")

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

        # Path to fit the model
        self.activations_path = os.path.join(dir_layer_activations,selected_version)
        
        # Create the directory where to save the model if it does not exist
        if not os.path.exists(os.path.join(PATH_BASELINE_METHODS)):
            os.makedirs(os.path.join(PATH_BASELINE_METHODS))
        
        self.path_to_baseline_methods = os.path.join(PATH_BASELINE_METHODS,f'{method_name}_{model_name}_{dataset_name}_layer_{hook_layer}')

        # Path where to save reconstruction metrics 
        self.metrics_reconstruction = os.path.join(PATH_BASELINE_METHODS_METRICS,f'{method_name}_{model_name}_{dataset_name}_layer_{hook_layer}')
        # Create the directory where to save the reconstruction results if it does not exist
        if not os.path.exists(self.metrics_reconstruction):
            os.makedirs(self.metrics_reconstruction)

        if method_name == 'ica':
           self.baseline_method_args = dict(
                n_components = 5,
                whiten = True,
                algorithm = 'parallel',
                fun='logcosh',
                max_iter=1000,
                random_state=42
            )
        elif method_name == 'concept_shap':
           self.baseline_method_args = dict(
                l1 = 0.001,
                l2 = 0.001,
                topk = 10,
                batch_size=32,
                epochs=3,
                loss_reg_epoch=2,
                n_concepts=10,
                hidden_dim=512,
                thres=0.1,
                nb_classes=4,
                seed=42
            ) 
        else:
            raise ValueError(f"The only supported baseline methods so far are ICA and ConceptSHAP")
        
        for key, value in baseline_method_args.items():
            if not isinstance(key, str):
                raise ValueError(f"baseline_method_args.{key} key is not a string.")

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

            self.baseline_method_args[key] = value
    
    
    def __repr__(self) -> str:
        
        result=f"""BaselineMethodConfig :
        
        Method name : {self.method_name}
        
        
       Additional arguments passed  :"""
        
            
        for key in sorted(self.baseline_method_args):
            result += f"\n    - {key}: {self.baseline_method_args[key]}"

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
            raise TypeError("The config file for BaselineMethodConfig is not found")
        
        
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



        if "hook_layer" not in parser["main"]:
            hook_layer = 4
        else:
            hook_layer = int(parser["main"]["hook_layer"])
        
        
        if "baseline_method_args" in parser:
            baseline_method_args = parser["baseline_method_args"]
        else:
            baseline_method_args = {}
            logger.info(
                "No 'baseline_method_args' section found in your config, using default values."
            )

        
        return cls(
            method_name=method_name,
            hook_layer=hook_layer,
            model_name=model_name,
            dataset_name=dataset_name,
            baseline_method_args=baseline_method_args
        )