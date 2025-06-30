from ..utils import LLMTrainerConfig

from transformers import AutoTokenizer, Trainer,TrainingArguments,PreTrainedModel, DataCollatorForLanguageModeling #for dynamic padding
from datasets import Dataset, concatenate_datasets
import torch
import os
import wandb
import random
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter


def compute_loss_last_token(
    inputs_labels: torch.Tensor,
    outputs_logits: torch.Tensor,
    vocab_size: int,
    is_eos : bool,
    reduction: str = "mean",
):
  """Computes the loss that focuses on the token corresponding to the answer"""

  #outputs_logits of size [8,M,50304]
  #8 : batch size
  #M : max length sequence in that batch
  # 50304 : vocabulary size


  logits_prediction = outputs_logits[:,(-2-int(is_eos))].contiguous().view(-1, vocab_size)
  labels_to_pred = inputs_labels[:,(-1-int(is_eos))].view(-1)

  test = torch.ones(vocab_size).to(inputs_labels.device)

  test[17] = 1
  test[18] = 100

  loss_ce = torch.nn.CrossEntropyLoss(reduction=reduction,weight=test)

  return loss_ce(logits_prediction,labels_to_pred)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Customize the loss

        outputs = model(**inputs)
        logits = outputs.logits
        loss = compute_loss_last_token(inputs.labels,logits,vocab_size=model.config.vocab_size,is_eos=global_is_eos)
        return (loss, outputs) if return_outputs else loss




def oversample_dataset(dataset, target_col='token_labels'):
    # Count occurrences of each label
    label_counts = Counter(dataset[target_col])
    
    # Find the maximum count
    max_count = max(label_counts.values())
    total_count = sum(label_counts.values())
    
    # Filter minority class
    balanced_data = dataset.filter(lambda x: x[target_col] == 1)
    
    # Convert to list and oversample
    balanced_data_list = [row for row in balanced_data]
    oversampled_minority = random.choices(balanced_data_list, k=2*max_count-total_count)
    
    # Convert oversampled data back to a Dataset
    oversampled_minority_dataset = Dataset.from_dict({k: [row[k] for row in oversampled_minority] for k in balanced_data_list[0]})
    
    # Concatenate with original dataset and shuffle
    balanced_dataset = concatenate_datasets([dataset, oversampled_minority_dataset])
    return balanced_dataset.shuffle()




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
    
    global global_is_eos 
    global_is_eos = cfg.is_eos

    # if cfg.dataset_name == "jigsaw_toxic_comment":
    #    #Unbalanced dataset : apply oversampling to the dataset
    #    train_dataset_tokenized = oversample_dataset(train_dataset_tokenized)
    #    print(train_dataset_tokenized)
    #    print(Counter(train_dataset_tokenized['token_labels']))

    
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