from ..utils import LLMTrainerConfig

from transformers import AutoTokenizer, Trainer,TrainingArguments,PreTrainedModel, DataCollatorForLanguageModeling # for dynamic padding
from datasets import Dataset
import torch
import os

# Custom loss function to train the autoregressive LLM backbones on the sentence classification task. Given the provided template, the ground-truth prediction token is located at position (-1-int(eos))
def compute_loss_last_token(
    inputs_labels: torch.Tensor,
    outputs_logits: torch.Tensor,
    vocab_size: int,
    eos : bool,
    reduction: str = "mean",
):
  """Computes the loss that focuses on the token corresponding to the classifier decision"""

  # outputs_logits of size [batch_size, max length sequence in that batch,vocabulary size]

  logits_prediction = outputs_logits[:,(-2-int(eos))].contiguous().view(-1, vocab_size)
  labels_to_pred = inputs_labels[:,(-1-int(eos))].view(-1)

  loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction)

  return loss_ce(logits_prediction,labels_to_pred)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Custom loss focused only on the entropy regarding the token predicting the class
        outputs = model(**inputs)
        logits = outputs.logits
        loss = compute_loss_last_token(inputs.labels,logits,vocab_size=model.config.vocab_size,eos=global_eos)
        return (loss, outputs) if return_outputs else loss



def get_trainer(
    cfg: LLMTrainerConfig,
    model: PreTrainedModel,
    train_dataset_tokenized: Dataset,
    test_dataset_tokenized: Dataset,
    tokenizer: AutoTokenizer
) -> Trainer :
    
    if 'report_to' in cfg.training_args and (cfg.training_args['report_to']=='wandb'):
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_PROJECT"]=f"{cfg.model_name}_fine_tuning_{cfg.dataset_name}"
        os.environ["WANDB_RUN_NAME"] = f"{cfg.model_name}_fine_tuning_{cfg.dataset_name}_{cfg.training_args['max_steps']}"
        

    global global_eos 
    # Do we have an end-of-sentence token in the tokenization. If it is the case, we have to shift the token to look at by 1.
    global_eos = cfg.eos 
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    used_training_args = TrainingArguments(**cfg.training_args)

    print(f'len train dataset : {len(train_dataset_tokenized)}')
    trainer = CustomTrainer(
        model=model,
        args=used_training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=test_dataset_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    return trainer