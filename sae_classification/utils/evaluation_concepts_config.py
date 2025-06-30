from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser
import torch
import ast

from .paths import *

class EvaluationConceptsConfig:
    """
    Configuration for evaluating concepts extracted by the different post-hoc methods 
    This config can be either used for:
        - the post-processing and selection of z_class along with a segmentation of the features
        - the computation of the recovery accuracy and selective causality of the concepts
        - the interpretability assessment of the extracted concepts according to coherence measures and ConceptSim/SentenceSim metrics 
    
    Args:
        method_name (str) : name of the method. So far, ICA, ConceptSHAP and SAE 
        
        hook_layer (int) : depth at which the activations extraction is resolved

        Specific arguments for ICA and ConceptSHAP :

            model_name (str) : the model the concepts extraction method was trained on

            dataset_name (str) : the dataset the concepts extraction method was trained on
        
        Specific arguments for SAE : 

            sae_name (str) : Name of the trained SAE

            checkpoint_version (int)  : Version of the SAE i.e. number of tokens seen during the training (counting duplicate)
            
            latest_version (bool) : If specified, it overrides 'checkpoint_version' and automatically assigned the SAE version with the highest number of training tokens

        ## TO DO : Unify above arguments

        causality (bool) : Whether we run an ablation study on the selected concepts after the simple accuracy reconstruction evaluation
        
        methods_args (dict)
    
    """
    method_name: str = "ica"

    
    methods_args: dict = {}


    def __init__(self,
                 method_name: str,
                 hook_layer: int = 4,
                 model_name:str = "",
                 dataset_name:str = "",
                 sae_name: str = "",  #Specifc parameter for SAE
                 checkpoint_version: int = 5000,  #Specifc parameter for SAE
                 latest_version: bool = True,  #Specifc parameter for SAE
                 causality: bool = False,
                 methods_args: dict = {}):
    

        self.method_name = method_name
        self.hook_name = f"blocks.{hook_layer}.hook_resid_pre"
        self.hook_layer = hook_layer
        self.test_causality = causality

        ######## Method Loading ###############

        supported_approaches = ["ica","concept_shap","sae"]
        assert method_name in supported_approaches, f"Error: The method {method_name} is not supported in evaluation. Currently the only supported methods are {supported_approaches}"

        # Path where to save reconstruction metrics 
        self.metrics_reconstruction = os.path.join(PATH_CONCEPTS_METRICS)
        # Path where to save custom datasets of cached activations, labels, attention_mask.
        # It is used to save time for the causality study which requires many inference runs for ablation tests.
        self.dir_dataset_activations = os.path.join(PATH_DATASET_ACTIVATIONS)
        
        # Store original and concepts activations from the inference on test dataset for potential further analysis
        self.activations_interpretability_methods_post_analysis = os.path.join(PATH_POST_ANALYSIS_ACTIVATIONS)

        if method_name in ["ica","concept_shap"]:

                
            self.path_to_baseline_methods = os.path.join(PATH_BASELINE_METHODS, f'{method_name}_{model_name}_{dataset_name}_layer_{hook_layer}')
            assert os.path.exists(os.path.join(PATH_BASELINE_METHODS)), f"Error: The folder {os.path.join(PATH_BASELINE_METHODS)} does not exist"


        elif method_name in ["sae"]:

            ######## SAE Loading ###############
        
            sae_name = f"sae_{model_name}_{dataset_name}_{self.hook_name}"
            self.sae_name = sae_name
            self.sae_checkpoint = PATH_CHECKPOINT_SAE
            self.latest_version = latest_version
            self.checkpoint_version = checkpoint_version

            
        # Path where to retrieve and save the selected features indices and their segmentation computed in the post-processing
        self.post_processed_features_path = PATH_CONCEPTS_METRICS


        #Default evaluation arguments to pass
        if torch.backends.mps.is_available(): #Apple
                device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"


        # Additional arguments for specific use-cases
        if method_name == 'concept_shap':
            self.methods_args = dict(
                n_concepts=10,
                hidden_dim=512,
                thres=0.1,
                device = device,
            )
        elif method_name == 'sae':
            '''
            The hidden layer of the SAE may be larger than the number of concepts we imposed to extract. 
            We implement additional post-selection procedure to extract a subset of the learned SAE features to match this number, it corresponds to the sub-vector z_class
            The variable 'features_selection' imposes which post-selection procedure to use, it can be either :
                    - 'truncation' : We select the first 'n_concepts' features in the hidden layer (based on index) 
                       This is the method we use when the joint classifier takes as input the first 'n_concepts' features of the hidden layer.
                    - 'logistic_regression' : We use a logistic regression to fill in z_class
            All the other features are constantly ablated for all evaluations. 
            '''
            self.methods_args = dict(
                features_selection = 'truncation',  
                n_concepts = 80,
                d_sae = 1536,
                feature_activation_rate=0.05,
                supervised_classification=True,
                device=device
            )
        
        elif method_name == 'ica':
            self.methods_args = dict(
                device='cpu'
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
        
        result=f"""EvaluationConceptsConfig :
        
        Method name : {self.method_name}    
        
        Arguments passed to the evaluation function :"""
        
            
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
            raise TypeError("The config file for EvaluationConceptsConfig is not found")


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

        
        causality = False
        if "causality" in parser:
            if "causality" in parser["causality"]:
                causality = (parser["causality"]["causality"].lower() == 'true')
            
        
        if "methods_args" in parser:
            methods_args = parser["methods_args"]
        else:
            methods_args = {}
            logger.info(
                "No 'methods_args' section found in your config, using default values."
            )

        
        
        return cls(
            method_name=method_name,
            hook_layer=hook_layer,
            model_name=model_name,
            dataset_name=dataset_name,
            sae_name=sae_name,
            checkpoint_version=checkpoint_version,
            latest_version=latest_version,
            causality=causality,
            methods_args=methods_args
        )