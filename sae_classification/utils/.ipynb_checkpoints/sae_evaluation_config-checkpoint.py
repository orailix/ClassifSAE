from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser
import re
import torch


from .paths import *

class SAELoadConfig:
    """
    Configuration for loading and evaluating a SAE on a dataset.
    
    Args:
        sae_name (str) : name of the directory in PATH_CHECKPOINT_SAE in which the TrainingSAE is stored
        
        checkpoint_version (int) : The version is measured by the number of activations the sae has been trained on

        causality (bool) : Wether or not we compute the metrics of causality (delta in logits and accuracy when the ablation is operated on one of the feature)

        topk_mean_activation (int): Due to the possible huge number of features in the SAE, it would be too computationally expensive to measure the causality of each feature. This is why we do a pre-selection of    'topk_mean_activation' features based on their mean actiavtions on the test dataset. We only select the k most activated features in average.
        
        evaluation_args (dict)
    
    """
    sae_name: str = ""
    
    checkpoint_version: int = 5000 #The version is measured by the number of activations the sae has been trained on
    
    evaluation_args: dict = {}
    dir_to_save_metrics: str
    dir_to_save_activations: str

    causality: bool
    topk_mean_activation: int
    
    def __init__(self,
                 sae_name: str,
                 checkpoint_version: int = 5000,
                 latest_version: bool = True,
                 causality: bool = False,
                 topk_mean_activation: int = 10,
                 evaluation_args: dict = {}):
    
    
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
        

        #Causality parameters
        self.causality = causality
        self.topk_mean_activation = topk_mean_activation

        #Directories of save
        self.dir_to_save_metrics = PATH_SAE_METRICS
        self.dir_to_save_activations = PATH_SAE_ACT
        self.dir_to_save_top_logits = PATH_SAE_TOP_LOGITS

        self.dir_acts_with_labels = PATH_ACTS_LABELS

        #self.path_select_features = os.path.join(PATH_SELECT_SAE_FEATURES,self.sae_name)
        
        #Default evaluation arguments to pass
        if torch.backends.mps.is_available(): #Apple
                device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.evaluation_args = dict(
            device=device,
            return_feature_activations=True,
            batch_size=2,
            proportion_to_evaluate=1,
            is_eos=True,
            prompt_tuning=False,
            loss_type='cross_entropy'
        )
        
        for key, value in evaluation_args.items():
            if not isinstance(key, str):
                raise ValueError(f"evaluation_args.{key} key is not a string.")

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

            self.evaluation_args[key] = value
        
        
    
    
    def __repr__(self) -> str:
        
        result=f"""SAELoadConfig :
        
        SAE used : {self.sae_name}
        
        
        Number of activations on which the sae was trained : {self.checkpoint_version}
        
        Evaluation Arguments passed to the evaluation function :"""
        
            
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
            raise TypeError("The config file for the sae evaluation is not found")
        
        
    @classmethod
    def from_config_file(cls,path):
        
        if isinstance(path,str):
            path=Path(path)
        
        parser = ConfigParser()
        parser.read(path)
        
        #Read the config file
        if "main" not in parser:
            raise ValueError("Your config should contain a 'main' section.")
        if "sae" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'sae' entry."
            )
       
            
        sae_name = parser["main"]["sae"]
        
        
        if "version" in parser:
            if "checkpoint_version" in parser["version"]:
                checkpoint_version = int(parser["version"]["checkpoint_version"])
            else:
                checkpoint_version = 500
            if "latest_version" in parser["version"]:
                latest_version = (parser["version"]["latest_version"].lower() == 'true')
            else:
                latest_version = True

        if "causality" in parser:
            if "causality" in parser["causality"]:
                causality = (parser["causality"]["causality"].lower() == 'true')
            else:
                causality = False
            if "topk_mean_activation" in parser["causality"]:
                try:
                    topk_mean_activation = int(parser["causality"]["topk_mean_activation"])
                except:
                    topk_mean_activation = 10
            else:
                topk_mean_activation = 10
        
        if "evaluation_args" in parser:
            evaluation_args = parser["evaluation_args"]
        else:
            evaluation_args = {}
            logger.info(
                "No 'evaluation_args' section found in your evaluation config, using default values."
            )
        
        return cls(
            sae_name=sae_name,
            checkpoint_version=checkpoint_version,
            latest_version=latest_version,
            evaluation_args=evaluation_args,
            causality=causality,
            topk_mean_activation=topk_mean_activation
        )