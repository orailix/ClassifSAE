from dataclasses import dataclass, field, fields
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MLPClassifierConfig:

    input_dim: int = 90
    hidden_dim: int = 256
    nb_hidden_layer: int = 2
    output_dim: int=4
    dropout: float=0.3

    # lr: float = 5e-5
    # lr_scheduler_name: str = "cosineannealing"
    # adam_beta1: float = 0
    # adam_beta2: float = 0.999

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "MLPClassifierConfig":
        # remove any keys that are not in the dataclass
        valid_field_names = {field.name for field in fields(cls)}
        valid_config_dict = {
            key: val for key, val in config_dict.items() if key in valid_field_names
        }
        return MLPClassifierConfig(**valid_config_dict)

    # def __post_init__(self):
    #     self.lr_end = self.lr / 10

    


class MLPClassifier(nn.Module):

    cfg: MLPClassifierConfig
    
    def __init__(self,  cfg: MLPClassifierConfig):
        """
        A simple Multi-Layer Perceptron (MLP) classifier.
        
        Args:
            input_dim (int): Number of input features (from AE's hidden layer).
            hidden_dims (list): List of hidden layer sizes.
            output_dim (int): Number of output classes.
            dropout (float): Dropout rate for regularization.
        """
        super(MLPClassifier, self).__init__()

        self.cfg = cfg
        
        layers = []
        prev_dim = cfg.input_dim
        h_dim = cfg.hidden_dim
        
        
        for i in range(cfg.nb_hidden_layer):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(cfg.dropout))  # Helps prevent overfitting
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, cfg.output_dim))
        
        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
      
        return self.model(x)

    def classification_loss(self, sae_activations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        predictions = self.forward(sae_activations)
        return self.loss_fn(predictions, labels)

