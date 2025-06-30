from pathlib import Path
import os
from .paths import *
from loguru import logger
from configparser import ConfigParser
import json
        

class FeatureVisualizationConfig():
    """
    Configuration for the visualization of selected SAE features. It creates an interactive view (thanks to sae-vis) either feature-centric or prompt-centric. 
    For the feature-centric view, we provide only a list of features to analyze. It displays the input texts activating the most these features and the top tokens that they promote in the vocabulary space with a similar method to Logit Lens.
    For the prompt-centric view, on top of the list of features, we provide a prompt (or a list of prompts - working on that) and for each token it displays the features (among those selected) the most relevant to it (according to a metric - currently either the quantile activation or the loss effect) 
        
    Args:
        view (str) : Either 'feature' (feature-centric view), 'prompt' (prompt-centric view) or 'all' (both - in that case, it creates successively the feature-centric vis and then prompt-centric vis according their respective parameters) 
        
        features_indices (list) : List of integers indices which correspond to the SAE features indices we want to analyze (parameter used for the feature/prompt-centric vis) - to know which ones are the most interesting, first run 'selection_small_model_neurons' or 
                                  'best_unique_neurons'.
        
        prompts (list) : List of prompts to analyze (parameter used for the prompt-centric vis).


    """
    
    
    def __init__(self,
                 view : str,
                 features_indices: list,
                 prompts: list):
        
        
        if view not in ['feature','prompt','all']:
            raise ValueError(f"The 'view' option must be either 'feature', 'prompt' or 'all'")
        
        self.view = view
        self.features_indices = []
        self.prompts = []
        
        if not features_indices:
                raise ValueError(f"The list of SAE features provided is empty")
        self.features_indices = features_indices
    
        
        if self.view == 'prompt' or 'all':
            
            if not prompts:
                raise ValueError(f"The list of test prompts provided is empty")
            self.prompts = prompts
        
        self.save_path = PATH_SAVE_VIS
        
        
        
        
    def __repr__(self) -> str:
        
            result=f"""FeatureVisualizationConfig :
            
            Point of View : {self.view}
            
            List of inspected features : {self.features_indices}
            
            List of tested prompts : {self.prompts}"""
            

            return result     
         
         
    @classmethod
    def autoconfig(cls,name):
        
        if isinstance(name,Path):
            logger.info(f"`file`: {name} is a valid path, building from it.")
            return cls.from_file(name)
        
        elif Path(name).is_file():
            logger.info(f"Autoconfig `name`: {name} is a valid path, building from it.")
            return cls.from_file(name)
        
        elif (PATH_CONFIG / name).exists():
            logger.info(
                f"Autoconfig `name`: {name} is a config file in the config folder, building from it."
            )
            return cls.from_file(PATH_CONFIG / f"{name}")
        
        elif (PATH_CONFIG / f"{name}.cfg").exists():
            logger.info(
                f"Autoconfig `name`: {name}.cfg is a config file in the config folder, building from it."
            )
            return cls.from_file(PATH_CONFIG / f"{name}.cfg")

        elif (PATH_CONFIG / f"{name}.json").exists():
            logger.info(
                f"Autoconfig `name`: {name}.json is a config file in the config folder, building from it."
            )
            return cls.from_file(PATH_CONFIG / f"{name}.json")
        
        else: 
            raise TypeError("The config file for the features visualization is not found")
        

    @classmethod
    def from_file(cls,path):
         """Builds a config object from a file.

         Args:
            - path: The path to the file, either .cfg or .json"""

         if type(path) == str:
                path = Path(path)
    
         if path.suffix == ".cfg":
                return cls.from_cfg(path)
         elif path.suffix == ".json":
                return cls.from_json(path)
         else:
                raise ValueError(f"Expected a .json or .cfg file: {path}")

    
    @classmethod
    def from_cfg(cls,path):
        
        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        
        parser = ConfigParser()
        parser.read(path)

        return cls.from_parser(parser)


    @classmethod
    def from_json(cls,path):
        """Builds a config object from a json file.

        Args:
            - path: The path to the json file"""

        path=Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        
        with open(path, "r") as f:
            json_content = json.load(f)

        return cls.from_parser(json_content)
        
    
    @classmethod
    def from_parser(cls,parser):

        """Builds a config object from a parsed config file.

        Args:
            - path: The configparser.ConfigParser object representing the parsed config.
        """

        
        #Read the config file
        if "main" not in parser:
            raise ValueError("Your config should contain a 'main' section.")
        if "view" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'view' entry."
            )
        if "features" not in parser["main"]:
            raise ValueError(
                "Section 'main' of your config should contain a 'features' entry."
            )
        if "prompts" not in parser["main"]:
            prompts = []
        else:
            prompts = parser["main"]["prompts"]
            
        view = parser["main"]["view"]
        features_indices = parser["main"]["features"]
        
        
        return cls(
            view=view,
            features_indices=features_indices,
            prompts=prompts
        )