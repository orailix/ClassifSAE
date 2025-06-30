from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from loguru import logger
import torch
from torch.utils.data import DataLoader, Dataset
import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeRegressor, plot_tree

from ..utils import LLMLoadConfig, ClassifierConfig

from .handle_datasets import process_dataset
from .models import get_hook_model
import torch.nn as nn
import torch.optim as optim


class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels=None):
        # Soft targets with temperature scaling
        soft_targets = torch.softmax(teacher_logits / self.temperature, dim=1)
        student_soft = torch.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL Divergence between student and teacher
        loss_kl = self.kl_div(student_soft, soft_targets) * (self.temperature ** 2)
        
        return loss_kl


class VectorLabelDataset(Dataset):
    def __init__(self, vectors, labels):
        """
        Args:
            vectors (torch.Tensor): Tensor of vectors with shape (N, d).
            labels (torch.Tensor): Tensor of labels with shape (N,).
        """
        self.vectors = vectors
        self.labels = labels

    def __len__(self):
        # Return the number of vectors
        return len(self.vectors)

    def __getitem__(self, idx):
        # Return the vector and its corresponding label
        vector = self.vectors[idx]
        label = self.labels[idx]
        return vector, label

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, output_dim,  hidden_dim ):
        super(MLPClassifier, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),                          
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),   
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),   
            nn.Linear(hidden_dim, output_dim)   
        )

    def forward(self, x):
        # Flatten input from [N, 1, d] to [N, d] if necessary
        x = x.squeeze(1)  # Removes the dimension of size 1 (shape becomes [N, d])
        return self.mlp(x)

def get_predicted_token_activations(hook_model,
                                    train_dataset_tokenized,
                                    data_collator,
                                    tokenizer,
                                    hook_layers_names,
                                    labels_tokens_id,
                                    is_validation,
                                    device,dim_resid_stream=1014,batch_size=16):

    print(f"Length of the dataset : {len(train_dataset_tokenized)}")
    tokens_dataloader = DataLoader(train_dataset_tokenized, batch_size=batch_size, collate_fn=data_collator)
    
    #Get the number of labels
    # vocab = tokenizer.get_vocab()
    # unique_labels = np.unique(np.array(train_dataset_tokenized["token_labels"]))
    # nb_labels = len(unique_labels)
    
    dict_activations = {}
    
    hook_model.eval()
    with torch.no_grad():
    
        for hook_name_layer in hook_layers_names:
            dict_activations[hook_name_layer] = torch.zeros((len(train_dataset_tokenized),dim_resid_stream))

        if is_validation:
            labels_tensor = torch.zeros(len(train_dataset_tokenized))
        else:
            labels_tensor = torch.zeros((len(train_dataset_tokenized),len(labels_tokens_id)))
        
        for i,batch in tqdm(enumerate(tokens_dataloader), desc="Get activations", unit="batch",total=len(tokens_dataloader)):
                inputs = batch['input_ids'].to(device)
                #prompt_labels = batch['token_labels']
                
                #Manually fixing the issue of sentence longer than the context windows size since it is not automatically handled by transformer-lens and it causes conflict with the positional embedding that it dones on a vector of size the context attention windows
                if inputs.shape[1] > hook_model.cfg.n_ctx:
                    inputs = inputs[:,-hook_model.cfg.n_ctx:] #It should not be an issue since we want to predict for almost the last token, so it looses almost nothing of the context it could have seen otherwise
               
                
                output, cache = hook_model.run_with_cache(inputs,
                                                    names_filter=hook_layers_names,
                                                    prepend_bos=False)
        
                model_prediction_logits = output[:,-3].contiguous().view(-1, output.shape[-1])

                if is_validation:
                    predict_labels = model_prediction_logits.argmax(dim=1)
                    labels_to_pred_clone = predict_labels.clone()
                      #Rename tokens to their label
                    for key,value in labels_tokens_id.items():
                        labels_to_pred_clone[labels_to_pred_clone==value] = int(key)
                    labels_tensor[(i*batch_size):((i+1)*batch_size)] = labels_to_pred_clone.cpu()
                else:
                    #distillation training
                    #Tensor of probabilities on the classes
                    labels_to_pred_clone = torch.zeros((model_prediction_logits.shape[0],len(labels_tokens_id)))
                    for k , (key,value) in enumerate(labels_tokens_id.items()):
                        labels_to_pred_clone[:,k] = model_prediction_logits[:,value]
                    labels_tensor[(i*batch_size):((i+1)*batch_size),:] = labels_to_pred_clone.cpu()
                    
                #print(f'prompt_labels.shape : {prompt_labels.shape}')
                
      
                #print(labels_to_pred_clone)
                for hook_layer_name in hook_layers_names:
                    cache_tensor = cache[hook_layer_name][:,-3,:]
                    dict_activations[hook_name_layer][(i*batch_size):((i+1)*batch_size),:] = cache_tensor.cpu()
            
    if is_validation:
        labels_tensor = labels_tensor.long()
        
    return dict_activations, labels_tensor
    
    
def classifier_training(dict_activations, 
                        labels_tensor,
                        device,
                        dim_resid_stream,
                        batch_size=4,learning_rate=0.001,num_epochs=10,hidden_size=512,temperature=2):
    
    dict_classifiers = {}
    nb_labels = labels_tensor.shape[-1]
    #print(f'labels_tensor : {labels_tensor}')
    
    for hook_name_layer, activations_tensor in dict_activations.items():

       
        #Initialize the Decision Tree Regressor
        decision_tree = DecisionTreeRegressor(random_state=42,max_depth=3)
    
        #It uses MSE loss
        decision_tree.fit(activations_tensor.numpy(), labels_tensor)

        # training_dataset = VectorLabelDataset(activations_tensor, labels_tensor)
        # training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
       
        # linear_classifier = MLPClassifier(dim_resid_stream,nb_labels,hidden_size).to(device)
        # criterion = DistillationLoss(temperature=temperature)
        # optimizer = optim.Adam(linear_classifier.parameters(), lr=learning_rate)
        # linear_classifier.train()   
        
        # logger.info(f"Training of a classifier on the model on the activations extracted from hook {hook_name_layer}")
        # for epoch in range(num_epochs):
            
        #     for i, (activations, batch_labels_logits) in tqdm(enumerate(training_dataloader), total=len(training_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        
        #         activations = activations.to(device)
        #         batch_labels_logits = batch_labels_logits.to(device)
        
        #         #Forward pass on the linear classifier
        #         classifier_output_logits = linear_classifier(activations)
            
        #         # Compute distillation loss
        #         loss = criterion(classifier_output_logits, batch_labels_logits)
 
        #         # Backward pass and optimization
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        #dict_classifiers[hook_name_layer] = linear_classifier
        dict_classifiers[hook_name_layer] = decision_tree 
        
    
    return dict_classifiers


def classifier_evaluation(dict_classifiers,dict_activations,labels_tensor,device,batch_size=4):
    
    dict_metrics = {}

    labels_tensor = labels_tensor.long()
    
    for hook_name_layer, classifier in dict_classifiers.items():
        
        activations_tensor = dict_activations[hook_name_layer]
        
        # testing_dataset = VectorLabelDataset(activations_tensor, labels_tensor)
        # testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True)
        
        # classifier.to(device)
        # classifier.eval()
        
        # total_loss = 0
        # correct_predictions = 0
        # total_samples = 0
        # criterion = nn.CrossEntropyLoss()
        
        # with torch.no_grad():
        # # Loop over the validation or test data
        #     for activations, batch_labels in tqdm(testing_dataloader, desc="Evaluating"):
                
        #         activations = activations.to(device)
        #         batch_labels = batch_labels.to(device)
                
        #         # Forward pass on the linear classifier
        #         classifier_output = classifier(activations)
                
        #         # Compute loss
        #         loss = criterion(classifier_output, batch_labels)
        #         total_loss += loss.item()
                
        #         # Get predicted labels by taking the class with the highest score
        #         _, predicted_labels = torch.max(classifier_output, 1)
                
        #         # Update the correct predictions count
        #         correct_predictions += (predicted_labels == batch_labels).sum().item()
        #         total_samples += batch_labels.size(0)

        #     # Calculate average loss and accuracy
        #     average_loss = total_loss / len(testing_dataloader)
        #     accuracy = correct_predictions / total_samples * 100

        y_pred = classifier.predict(activations_tensor.numpy())
    
        y_pred = np.argmax(y_pred, 1)
        accuracy = np.sum(y_pred == labels_tensor.numpy()) / len(labels_tensor)
        dict_metrics[hook_name_layer] = {"Validation Loss" : 0, "Accuracy" : accuracy}
        #dict_metrics[hook_name_layer] = {"Validation Loss" : average_loss, "Accuracy" : accuracy}
        
    return dict_metrics




def train_classifier(config_model , config_classifier):
    
    cfg_model = LLMLoadConfig.autoconfig(config_model)
    
    cfg_classifier = ClassifierConfig.autoconfig(config_classifier)
     
    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
     
    #Process the test and train dataset
    train_dataset_tokenized = process_dataset(cfg_model,split="train",tokenizer=tokenizer) 
    test_dataset_tokenized = process_dataset(cfg_model,split="test",tokenizer=tokenizer) 
    
    #Get model hooked (HookedTransformer)
    hook_model = get_hook_model(cfg_model,tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    hook_layers_names = cfg_classifier.hook_layers_names
    print(f'hook_layers_names : {hook_layers_names}')
    print(f'hook_layers_names type : {type(hook_layers_names)}')
    
    device = cfg_classifier.device
    
    dim_resid_stream = cfg_classifier.dim_activation

    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(np.array(train_dataset_tokenized["token_labels"]))
    keys_labels = set(unique_labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}

    logger.info(f"Collecting {cfg_model.model_name} layers activations on the train split of the dataset {cfg_model.dataset_name}...")
    dict_activations_train, labels_tensor_train = get_predicted_token_activations(hook_model,train_dataset_tokenized,data_collator,tokenizer,hook_layers_names,labels_tokens_id,is_validation=False,device=device,dim_resid_stream=dim_resid_stream)    
    
    logger.info(f"Training of classifier(s) on the model {cfg_model.model_name} on the train split of the dataset {cfg_model.dataset_name}")
    dict_classifiers = classifier_training(dict_activations_train, labels_tensor_train,device,dim_resid_stream,**cfg_classifier.training_args )
    
    # #Save the state dictionaries (weights and biases) of the classifiers
    # save_dict = {hook_name: model.state_dict() for hook_name, model in dict_classifiers.items()}
    # # Save the dictionary of state dictionaries to a file
    # file_to_save_models = os.path.join(cfg_classifier.directory_to_save,cfg_model.dataset_name,f'classifiers_{cfg_model.model_name}.pth')
    # torch.save(save_dict, file_to_save_models)
    # logger.info(f"The trained classifiers on the layers {hook_layers_names} are saved at {file_to_save_models}")

    logger.info(f"Collecting {cfg_model.model_name} layers activations on the test split of the dataset {cfg_model.dataset_name}...")
    #Evaluate performance metrics on the classification task for the test split of the selected dataset for the classifiers
    dict_activations_test, labels_tensor_test = get_predicted_token_activations(hook_model,test_dataset_tokenized,data_collator,tokenizer,hook_layers_names,labels_tokens_id,is_validation=True,device=device,dim_resid_stream=dim_resid_stream)
    
    dict_metrics = classifier_evaluation(dict_classifiers,dict_activations_test,labels_tensor_test,device)
    file_to_save_metrics = os.path.join(cfg_classifier.directory_to_save,cfg_model.dataset_name,f'classifiers_{cfg_model.model_name}.json')
    with open(file_to_save_metrics, 'w') as file:
        json.dump(dict_metrics, file, indent=4) 
    
    print(dict_metrics)

