from pathlib import Path
import os
from .paths import *
from loguru import logger
from configparser import ConfigParser

class SelectionNeuronsConfig():
    """
    Configuration for the process of selecting neurons (original or sae-based) the most adapted to the required task based on their cached activations.
    
    Args:
        activations_run_name (str) : name of the directory in PATH_SAE_ACT in which activations of the original and sae neurons are stored. 
        
        dataset_name (str) : Dataset on which we did the forward pass to store the original and sae neurons activations 
        
        top_k (int) : Number of top neurons

        imitation (bool) : Wether we want to copy the predictions of the LLM or if we want to obtain the best possible accuracy on the ground truth
        
        (Optional - Not used by some functions)
        
        model (str) : The type of model used to bridge the stored activations to the LLM probabilities distributions. For now, we only support 'Linear Classifier' or 'Decision Tree'
        
        distillation_args (dict) contains (for now)
            num_epochs (int) : Number of epochs for the distillation training
            batch_size (int) : Batch size used during the distillation training

    """
    
    distillation_args : dict = {}
    
    def __init__(self,
                 activations_run_name: str,
                 dataset_name: str,
                 top_k : int = 2,
                 imitation = False,
                 model : str = "Linear Classifier",
                 distillation_args: dict = {}):
        
        self.activations_run_name = activations_run_name
        self.dataset_name = dataset_name
        self.imitation = imitation
        
        
        #Load activations path
        self.activations_path = os.path.join(PATH_SAE_ACT, activations_run_name,f"{dataset_name}.pth")
         
        if not (os.path.exists(self.activations_path)):
            raise ValueError(f"The directory name provided is either not present in {PATH_SAE_ACT} or does not contain a file linked to the specified dataset")
        
        self.top_k = top_k
        self.model  = model
        
        #Path where to save the results
        self.save_path = os.path.join(PATH_NEURONS_SELECTION_METRICS,activations_run_name,dataset_name)
        #Create the path if it does not exist yet
        os.makedirs(self.save_path, exist_ok=True)
        
        #Default evaluation arguments to pass
        self.distillation_args = dict(
            num_epochs = 100,
            batch_size = 32
        )
        
        for key, value in distillation_args.items():
            if not isinstance(key, str):
                raise ValueError(f"distillation_args.{key} key is not a string.")
            
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

            self.distillation_args[key] = value
        
        
        
        
    def __repr__(self) -> str:
        
            result=f"""SelectionNeuronsConfig :
            
            Activations of the Run {self.self.activations_run_name} on the dataset {self.dataset_name}.
            
            Number of k selected neurons : {self.top_k}
            
            Small model : {self.model}
            
            Distillation Training Arguments :"""
            
                
            for key in sorted(self.training_args):
                result += f"\n    - {key}: {self.evaluation_args[key]}"

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
            raise TypeError("The config file for the neurons selection is not found")
        
        
    @classmethod
    def from_config_file(cls,path):
        
        if isinstance(path,str):
            path=Path(path)
        
        parser = ConfigParser()
        parser.read(path)
        
        #Read the config file
        if "main" not in parser:
            raise ValueError("Your config should contain a 'main' section.")
        if "activations_run" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'activations_run' entry."
            )
        if "dataset" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'dataset' entry."
            )
        if "top_k" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'top_k' entry."
            )
        if "imitation" not in parser["main"]:
            imitation = False
        else:
            imitation = (parser["main"]["imitation"].lower() == "true")
            
        activations_run_name = parser["main"]["activations_run"]
        dataset_name = parser["main"]["dataset"]
        top_k = int(parser["main"]["top_k"])
        try:
            top_k = int(top_k)
        except ValueError:
            print("top_k must be an integer")

        
        if "type_model" in parser and "model" in parser["type_model"]:
            model = parser["type_model"]["model"]
        else :
            model = ""
        
        if "distillation_args" in parser:
            distillation_args = parser["distillation_args"]
        else:
            distillation_args = {}
            logger.info(
                "No 'distillation_args' section found in your distillation config, using default values."
            )
        
        return cls(
            activations_run_name=activations_run_name,
            dataset_name=dataset_name,
            top_k=top_k,
            imitation=imitation,
            model=model,
            distillation_args=distillation_args
        )