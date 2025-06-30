from ..utils import LLMTrainerConfig, LLMLoadConfig
from transformer_lens import HookedTransformer

from loguru import logger
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM,AutoTokenizer
from typing import Union
from peft import PromptTuningConfig, get_peft_model, TaskType
from torch.optim import AdamW

import sys

class PromptTunerForHookedTransformer(HookedTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for param in self.parameters():
            param.requires_grad = False

        # Initialize learnable prompt embeddings
        self.num_prompt_tokens = 20
        self.prompt_embeddings = torch.nn.Parameter(
            torch.randn(self.num_prompt_tokens, self.cfg.d_model, device=self.cfg.device)
        )

        
        # Add hook to inject prompt embeddings
        self.add_hook("hook_embed", self._add_prompt_embeddings)

        self.optimizer = torch.optim.Adam([self.prompt_embeddings], lr=1e-2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7500, gamma=0.1)


    def _add_prompt_embeddings(self, input_embeddings, hook):
        """
        Hook function to prepend prompt embeddings to input embeddings and update the attention mask.
        """

        input_embeddings = input_embeddings.clone().detach().requires_grad_(True)


        # Get batch size
        batch_size = input_embeddings.size(0)


        prompt_broadcasted = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        # Create a new tensor with prompt embeddings prepended

        input_with_prompt = torch.cat([prompt_broadcasted, input_embeddings[:, self.num_prompt_tokens:, :]], dim=1)


        # Return both modified embeddings and updated attention mask
        return input_with_prompt

    def train_step(self, input_ids, attention_mask, loss_fn):

        batch_size = input_ids.size(0)
        prompt_token_ids = torch.full(
            (batch_size, self.num_prompt_tokens),
            self.tokenizer.eos_token_id,  # Using EOS token as a placeholder
            dtype=torch.long,
            device=input_ids.device,
        )
        input_ids = torch.cat([prompt_token_ids, input_ids], dim=1)
        prompt_attention_mask = torch.full(
            (batch_size, self.num_prompt_tokens),
            1,  # Using EOS token as a placeholder
            dtype=torch.long,
            device=input_ids.device,
        )
        attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

        logits = self(input_ids, attention_mask=attention_mask,return_type="logits")
        #logits = outputs[:, self.num_prompt_tokens:, :]  # Ignore prompt tokens

        loss = loss_fn(input_ids, logits, vocab_size=self.W_E.shape[0])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def evaluate(self,
                 input_ids : torch.Tensor,
                 attention_mask : torch.Tensor,
                 labels_tokens_id : dict,
                 is_eos : bool):

        
        with torch.no_grad():


            batch_size = input_ids.size(0)
            prompt_token_ids = torch.full(
                (batch_size, self.num_prompt_tokens),
                self.tokenizer.eos_token_id,  # Using EOS token as a placeholder
                dtype=torch.long,
                device=input_ids.device,
            )
            input_ids = torch.cat([prompt_token_ids, input_ids], dim=1)
            prompt_attention_mask = torch.full(
                (batch_size, self.num_prompt_tokens),
                1,  # Using EOS token as a placeholder
                dtype=torch.long,
                device=input_ids.device,
            )
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)

            logits = self(input_ids, attention_mask=attention_mask,return_type="logits")

            _,_,vocab_size = logits.size()

            logits_prediction = logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size).argmax(dim=1)
            labels_to_pred = input_ids[:,(-1-int(is_eos))].view(-1)

            exact_matches = (logits_prediction==labels_to_pred)
            count_exact_matches = exact_matches.sum() #tensor


        return count_exact_matches.item()




def get_model(
    cfg: Union[LLMTrainerConfig,LLMLoadConfig]
    ) -> PreTrainedModel:

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if cfg.model_path_pre_trained==cfg.model_path: #It you want only load the pre-trained model, either just to evaluate it like that or train the pre-train
    
        if cfg.quantized:

                logger.info("Quantization of the model")
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path_pre_trained,
                                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True)   # Enables quantization
                                                             )
        else:
                #model = AutoModelForCausalLM.from_pretrained(cfg.model_path_pre_trained,torch_dtype=torch.bfloat16)
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path_pre_trained)

        if isinstance(cfg,LLMTrainerConfig) and cfg.tuning == "lora" :
            # LoRA init
            lora_config = LoraConfig(
                r=8,  # LoRA rank
                lora_alpha=32,  # LoRA alpha
                target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA to
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"  # Task type
            )

            model = get_peft_model(model, lora_config)
    
    else:

        if cfg.tuning == "lora":

             if cfg.quantized:
                logger.info("Quantization of the model")
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path_pre_trained,
                                                             quantization_config=BitsAndBytesConfig(load_in_4bit=True)   # Enables quantization
                                                             )
             else:
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path_pre_trained)
             logger.info("Retrieval of the LoRa weights")
             model = PeftModel.from_pretrained(model, cfg.model_path)
             

        else:
            #model = AutoModelForCausalLM.from_pretrained(cfg.model_path,torch_dtype=torch.bfloat16)
            model = AutoModelForCausalLM.from_pretrained(cfg.model_path)

    # # Define Prompt Tuning configuration
    # prompt_tuning_config = PromptTuningConfig(
    #     task_type=TaskType.CAUSAL_LM,   # Task type (causal language modeling)
    #     num_virtual_tokens=10,         # Number of virtual tokens (soft prompts)
    #     token_dim=model.config.hidden_size
    # )

    # # Wrap the model with PEFT for Prompt Tuning
    # model = get_peft_model(model, prompt_tuning_config)

    #model = PeftModel.from_pretrained(model, "/lustre/fswork/projects/rech/dun/upa23os/v3/finetuned_models/llama3-1_8B_instruct/ag_news/checkpoint-20000")

    
    model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = torch.nn.DataParallel(model,output_device=None)
    
        
    return model

def get_hook_model(
    cfg :  Union[LLMLoadConfig],
    tokenizer: AutoTokenizer
    ) -> HookedTransformer:

    model = get_model(cfg)
    # print(f"dir(model) : {model.modules}")
    # print(f"Type model.model : {type(model.model)}")

    
    
    if not cfg.task_args['prompt_tuning'] :
        #If available, it will load on cuda
        hook_model = HookedTransformer.from_pretrained_no_processing(
                cfg.model_architecture,
                hf_model=model,
                tokenizer=tokenizer,
                #dtype=torch.bfloat16,
                #n_devices=torch.cuda.device_count()
            )
    
        hook_model.tokenizer = tokenizer
    else:
        hook_model = PromptTunerForHookedTransformer.from_pretrained_no_processing(
                             cfg.model_architecture,
                             hf_model=model,
                             tokenizer=tokenizer,
                             #dtype=torch.bfloat16 
                        )
    

    return hook_model