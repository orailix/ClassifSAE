from pathlib import Path
import os
from loguru import logger
from configparser import ConfigParser
import torch
import ast
import re

from .paths import *
from .datasets_parameters import DICT_CATEGORIES, DICT_DATASET_ALIAS

class LLMLoadConfig:
    """
    Configuration for loading the LLM model tasked with the classification task.
    It can be used either for the evaluation of its classification performance or for running an inference and caching its activations.
    
    Args:
        model_name (str)
        model_path (str) : Usually the path used to retrieve the pre-trained model on HuugingFace. This the path used by default if the finetuned option is deactivated, it is also used by HookedTransformer to ensure that the architecture is handled by transformer_lens
        dataset_name (str) : either dataset for caching or evaluation
        quantized (bool) : Load the the model weights in 8-bit
        prompt_tuning (bool) : Will the forward passes on the LLM require doing prompt fine-tuning ?

        finetuned (bool) : Whether the model is to be searched in the directory of finetuned models, otherwise it is considered as a pre-trained model
        
        (Only considered if 'finetuned'=True)
        dataset_tuned_on (str) : name of the dataset on which the model was finetuned
        checkpoint_version (int) : Which tuning version is to be evaluated (corresponds to the number of tuning steps).
        latest_version (bool) : If True, it overrides 'checkpoint_version' to select the latest checkpoint of the tuned model
        
        task (str) Possible tasks, must be in ['evaluation','caching','sae_training']

        
        task_args (dict)
        (
            Depending if we use this config to load for evaluation , activation caching or training an SAE :
            - Evaluation : evaluation parameters that are used by eval_model_classification_perf() in evaluation.py
            - Activation Caching : Parameters passed to CacheActivationsRunnerConfig
            - SAE Training : Parameters passed to LanguageModelSAERunnerConfig
        )
    
    """
    
    model_name: str = "pythia_14m"
    model_architecture: str = "pythia-14m"
    dataset_name: str = "imdb"

    # We have not implemented extraction of concepts for models in few shot learning yet
    add_example: bool = False
    
    checkpoint_version: int = 5000
    
    task_args: dict = {}

    split: str = "train"    
    dir_to_save_metrics: str
  
    model_path: str 
    model_path_pre_trained: str
    dataset_path: str

    
         
    def __init__(self,
                model_name: str,
                model_path: str,
                dataset_name: str,
                split: str,
                quantized: bool,
                finetuned: bool,
                prompt_tuning: bool,
                vtok: int,
                prompt_tuning_total_steps: int,
                dataset_tuned_on: str,
                checkpoint_version: int = 5000,
                latest_version: bool = True,
                task: str = "evaluation",
                task_args: dict = {}):

        available_tasks = ["evaluation","caching","sae_training"]
        if task not in available_tasks:
            raise ValueError(f"{task} does not belong to the available tasks for the loaded LLM. Available tasks must be in {available_tasks}")

        ############### MODEL Loading ########################

        self.model_path_pre_trained = model_path
        self.model_architecture = model_path
        self.quantized = quantized

        if finetuned:
            load_dir = os.path.join(PATH_CHECKPOINTS, f"{model_name}/{dataset_tuned_on}")
            ## The different versions of the tuned model are saved in folders named 'checkpoint-X'
            
            if not (os.path.exists(load_dir) and os.path.isdir(load_dir)):
                raise ValueError(f"The model name provided is either not present in {PATH_CHECKPOINTS} or not tuned with the provided dataset name.")
            
            numbers_steps = []
            version_names = {}
            for version in os.listdir(load_dir):
                if os.path.isdir(os.path.join(load_dir, version)) and re.match(r'checkpoint-\d+', version):
        
                    number_steps = int(version.split('-')[-1])
                    numbers_steps.append(number_steps)
                    version_names[number_steps] = version
            if numbers_steps==[]:
                raise ValueError(f"No checkpoints available in {load_dir}. Verify that the checkpoints folders are named according to the template 'checkpoint-X' with 'X' an integer.")
            
            if latest_version:
                numbers_steps.sort(reverse=True)
                max_number_steps = numbers_steps[0]
                selected_version = version_names[max_number_steps]
                self.checkpoint_version = max_number_steps
            else:
                # Find the model tuned the with closest number of steps to the one provided in checkpoint_version 
                closest_number_steps = min(numbers_steps, key=lambda x: abs(x - checkpoint_version))
                selected_version = version_names[closest_number_steps]
                self.checkpoint_version = checkpoint_version
            
            self.model_name = f"{model_name}_ft_{dataset_tuned_on}_{self.checkpoint_version}"
            self.model_path = os.path.join(load_dir,selected_version)
        
        else:
            # We use directly a pre-trained model to evaluate its performance on the evaluated dataset
            self.model_name = model_name
            self.model_path = model_path
            
        # Will the forward passes on the LLM require doing prompt fine-tuning ?
        self.prompt_tuning  = prompt_tuning
        self.prompt_embeddings_dir = PATH_PROMPT_EMBEDDINGS

        if vtok==-1:
            vtok=10

        if prompt_tuning_total_steps==-1:
            prompt_tuning_total_steps=2000

        self.prompt_tuning_params = {'vtok': vtok, 'total_steps':prompt_tuning_total_steps}

        if self.prompt_tuning:
            self.prompt_embeddings_path = os.path.join(self.prompt_embeddings_dir,model_name,dataset_name,f"embeddings_prompt_{self.prompt_tuning_params['vtok']}_{self.prompt_tuning_params['total_steps']}.pt")
        else:
            self.prompt_embeddings_path=None

        #################### DATASET Loading ###################################

        if dataset_name in list(DICT_CATEGORIES.keys()):
            self.match_label_category = DICT_CATEGORIES[dataset_name]
        else:
             raise ValueError(f"The dataset name provided {dataset_name} is not yet managed. Available datasets are {list(DICT_CATEGORIES.keys())}")
        
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(PATH_LOCAL_DATASET, dataset_name)
        
        self.dataset_in_local = os.path.exists(self.dataset_path) and os.path.isdir(self.dataset_path)
        self.official_name_repo = DICT_DATASET_ALIAS[self.dataset_name]
    
        
        ################### TOKENIZER Loading #############################

    
        self.tokenizer_path = self.model_path_pre_trained
        self.split = split


        self.dir_to_save_metrics = os.path.join(PATH_MODEL_METRICS,self.model_name)

        # If no task_args provided, we go for the defaults given the specified task
        if task=="evaluation":
            self.task_args = dict(
                    eos=False,
                    batch_size=1,
                    proportion_to_evaluate=1.
                )
        if task=="sae_training":
            
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            total_training_steps = 10 
            batch_size = 40
            training_tokens = total_training_steps * batch_size
            self.task_args = dict(
                architecture = "standard",
                hook_layer = 5,
                d_in=128,
                prepend_bos=False, #There is no bos in the already tokenized dataset used.
                use_cached_activations=True,
                prompt_tuning=False,
                len_epoch=120000,
                save_label=False,
                # SAE Parameters
                activation_fn = 'topk', # relu, tanh-relu, topk
                topk = [1], #for topk
                mse_loss_normalization=None, 
                expansion_factor=4,  # the width of the SAE. Larger will result in better stats but slower training.
                b_dec_init_method="zeros",  # The geometric median can be used to initialize the decoder weights.
                apply_b_dec_to_input=False,  # We won't apply the decoder weights to the input.
                normalize_sae_decoder=True,
                decoder_orthogonal_init = False,
                scale_sparsity_penalty_by_decoder_norm=False,
                decoder_heuristic_init=False,
                init_encoder_as_decoder_transpose=False,
                normalize_activations='constant_norm_rescale',
                # Training Parameters
                lr=5e-5,  
                adam_beta1=0.9,  # adam params 
                adam_beta2=0.999,
                lr_scheduler_name="constant",  # constant learning rate with warmup. 
                lr_warm_up_steps=0,  # this can help avoid too many dead features initially.
                lr_decay_steps=total_training_steps // 5,  # this will help us avoid overfitting.
                l1_coefficient=5, 
                lmbda_mse=1, # weight on the reconstruction loss of the hidden state in the training objective
                lmbda_activation_rate=0.0, # weight on the activation rate loss (sparsity of the feature activation over the batch) in the training objective
                lmbda_vcr=0.0, # weight on the vcr loss in the training objective
                lmbda_decoder_columns_similarity=0.0, # weight on the decoder columns similarity loss in the training objective
                lmbda_classifier=0.0, # weight on the classifier loss in the training objective 
                num_classifier_features=80, # number of features within the SAE hidden layer used as inputs for the trained imitation classifier, it defines the dimension of z_class (only useful if lmbda_classifier > 0 )
                nb_classes=4, # For each sentence of the dataset, the number of classes from which the classifier must select (only useful if lmbda_classifier > 0 )
                feature_activation_rate=0.05, # Targeted maximum fraction of sentences across the batch on which a feature is active (i.e non zero). The associated activation rate sparsity loss becomes non-zero above this threshold. The frequency is normalized between 0.0 and 1.0
                l1_warm_up_steps=total_training_steps // 20 ,  # this can help avoid too many dead features initially.
                lp_norm=1.0,  # the L1 penalty (and not a Lp for p < 1)
                train_batch_size_tokens=batch_size, #Number of activations per batch
                # Activation Store Parameters
                n_batches_in_buffer=4,  # controls how many activations  shuffle.
                training_tokens=training_tokens,  
                store_batch_size_prompts=2, #The batch size for storing activations. This controls how many prompts are in the batch of the language model when generating activations.
                # Resampling protocol
                use_ghost_grads=False, 
                feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
                dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
                dead_feature_threshold=1e-4,  # would effect resampling or ghost grads if we were using it.
                #VERBOSE
                verbose=True,
                # WANDB
                log_to_wandb=True,  # always use wandb unless you are just testing code.
                wandb_log_frequency=30,
                # Misc
                device=device,
                seed=42,
                n_checkpoints=2,
                checkpoint_path=PATH_CHECKPOINT_SAE,
                dtype="float32"
            )
           
            
        if task=="caching":
            self.task_args = dict(
               hook_layer = 5,
               d_in=128,
               prepend_bos=False, #There is no bos in the already tokenized dataset used.
               training_tokens = 400,
               eos=False,
               prompt_tuning=False,
               prompt_embeddings_path=self.prompt_embeddings_path,
               save_label=False,
               #buffer details
               n_batches_in_buffer=4,
               store_batch_size_prompts=2,
               normalize_activations='none',
               #
               shuffle_every_n_buffers=8,
               n_shuffles_with_last_section=1,
               n_shuffles_in_entire_dir=1,
               n_shuffles_final=1,
               # Misc
               device='cpu', #We use the 'cpu' to store the activations. #Not needed to be specified in the file by the user
               seed=42,
               dtype="float32"
            )
        
        for key, value in task_args.items():
            if not isinstance(key, str):
                raise ValueError(f"task_args.{key} key is not a string.")

            if isinstance(value, str):

                #We allow the input of lists of parameters. In that case, we launch multiple runs to test all combinations of the specified hyperparameters
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

            self.task_args[key] = value

        ## Based on the task type, we set up a last few parameters
        if task=="caching":
            directory_label = "with_labels" if self.task_args['save_label'] else "without_labels"
            self.task_args['hook_name'] =   f"blocks.{self.task_args['hook_layer']}.hook_resid_pre" # We always branch at the residual stream entering the specified layer l. It means that it also corresponds to the residual stream exiting the (l-1)-th layer.
            self.task_args['new_cached_activations_path'] = os.path.join(PATH_CACHED_ACT,self.model_name,self.dataset_name,self.split,directory_label,self.task_args['hook_name'],str(self.task_args['training_tokens']))
        elif task=="sae_training":
            directory_label = "with_labels" if self.task_args['save_label'] else "without_labels"
            self.task_args['hook_name'] =  f"blocks.{self.task_args['hook_layer']}.hook_resid_pre" # We always branch at the residual stream entering the specified layer l. It means that it also corresponds to the residual stream exiting the (l-1)-th layer.
            self.task_args["wandb_project"] = f"sae_{self.model_name}_{self.dataset_name}_{self.task_args['hook_name']}"
            self.task_args["wandb_id"] = f"sae_{self.model_name}_{self.dataset_name}_{self.task_args['hook_name']}"
            
            #In case if there are multiple possibilities in the cache activation directories, we go for the one that has the most stored activations
            load_cache_dir = os.path.join(PATH_CACHED_ACT,self.model_name,self.dataset_name,self.split,directory_label,self.task_args['hook_name'])
            if os.path.exists(load_cache_dir) and os.path.isdir(load_cache_dir):
                all_entries = os.listdir(load_cache_dir)
                integer_directories = [name for name in all_entries if name.isdigit() and os.path.isdir(os.path.join(load_cache_dir,name))]
                integer_directories = [int(name) for name in integer_directories]
                if integer_directories:
                    max_steps_store = max(integer_directories)
                    max_steps_store = str(max_steps_store)
                else:
                    raise ValueError(f"The directory provided that is supposed to contain the activations is empty : {load_cache_dir}")
            else:
                raise ValueError(f"The directory provided that is supposed to contain the activations does not exist : {load_cache_dir}")
            
            self.task_args['cached_activations_path']=os.path.join(load_cache_dir,max_steps_store)
            logger.info(f"We train on the activations from {self.task_args['cached_activations_path']}")
            
        
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
            raise TypeError("The config file for the LLM classifier loading is not found")
        
        
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
            dataset_name = "ag_news"
        else:
            dataset_name = parser["main"]["dataset"]
        if "split" not in parser["main"]:
            split="train"
        else:
            split=parser["main"]["split"]
        if "quantized" not in parser["main"]:
            quantized=False
        else:
            quantized=(parser["main"]["quantized"].lower()=='true')

        
        model_name = parser["main"]["model"]
        model_path = parser["main"]["model_path"]
        
        
        if "task" not in parser["main"]:
            task = "evaluation"
        else:
            task = parser["main"]["task"]
        
        if "version" in parser:
            if "finetuned" in parser["version"]:
                finetuned = (parser["version"]["finetuned"].lower() == 'true')
            else:
                finetuned=True
            if "checkpoint_version" in parser["version"]:
                checkpoint_version = int(parser["version"]["checkpoint_version"])
            else:
                checkpoint_version = 500
            if "latest_version" in parser["version"]:
                latest_version = (parser["version"]["latest_version"].lower() == 'true')
            else:
                latest_version = True
            if "dataset_tuned_on" in parser["version"]:
                dataset_tuned_on = parser["version"]["dataset_tuned_on"]
            else:
                dataset_tuned_on = dataset_name

        vtok=-1
        prompt_tuning_total_steps = -1

        # In case we do prompt-tuning
        if "prompt_tuning" in parser:
            prompt_tuning = True
            if "vtok" in parser["prompt_tuning"]:
                vtok = int(parser["prompt_tuning"]["vtok"])
            
            if "total_steps" in parser["prompt_tuning"]:
                prompt_tuning_total_steps = int(parser["prompt_tuning"]["total_steps"])
                

        else:
            prompt_tuning=False
        

        if "task_args" in parser:
            task_args = parser["task_args"]
        else:
            task_args = {}
            logger.info(
                "No 'task_args' section found in your loading config, using default values."
            )
        
        return cls(
            model_name=model_name,
            model_path=model_path,
            dataset_name=dataset_name,
            split=split,
            quantized=quantized,
            prompt_tuning=prompt_tuning,
            vtok=vtok,
            prompt_tuning_total_steps=prompt_tuning_total_steps,
            finetuned=finetuned,
            dataset_tuned_on=dataset_tuned_on,
            checkpoint_version=checkpoint_version,
            latest_version=latest_version,
            task=task,
            task_args=task_args
        )
