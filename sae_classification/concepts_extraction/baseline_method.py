from typing import Optional
import os
import torch
import numpy as np
from safetensors.torch import load_file
from loguru import logger
from ..utils import BaselineMethodConfig, LLMLoadConfig
from ..llm_classifier_tuning import get_hook_model
from sklearn.decomposition import FastICA
from transformers import AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib

# Concepts discovery method proposed in ConceptSHAP
class ConceptNet(nn.Module):

    def __init__(self, n_concepts, train_embeddings,hidden_dim,thres):
        super(ConceptNet, self).__init__()
        embedding_dim = train_embeddings.shape[1]
        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)
        self.n_concepts = n_concepts
        self.train_embeddings = train_embeddings.transpose(0, 1) # (dim, all_data_size)

        self.rec_vector_1 = nn.Parameter(self.init_concept(n_concepts, hidden_dim), requires_grad = True)
        self.rec_vector_2 = nn.Parameter(self.init_concept(hidden_dim, embedding_dim), requires_grad = True)

        self.thres = thres
        

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept

    
    # This method efficiently computes the first regularizer loss term of the method associated with the paper of ConceptSHAP
    # It computes for each concept the set of top-K nearest neighbors (Steps 1 to 4) and then computes the average dot product over the K neighbors
    def efficient_knn_dot_avg(self,sampled_train_embeddings, k):
        """
        concept: (activation_dim, n_concepts)
        sampled_train_embeddings: (activation_dim, num_total_activations)
        k: number of nearest neighbors
        """
    
        # 1. Precompute dot products and norms for distance
        #    c^T x = shape (n_concepts, num_total_activations)
        #    c^T c, x^T x are used to compute squared distances

        sampled_train_embeddings = sampled_train_embeddings.to(dtype=self.concept.dtype)
        
        cx_dot = self.concept.T @ sampled_train_embeddings  # (n_concepts, num_total_activations)
    
        # Squared L2 norms of each concept vector
        # concept_norm[i] = sum(concept[:, i] ** 2)
        concept_norm = (self.concept * self.concept).sum(dim=0)  # (n_concepts,)
    
        # Squared L2 norms of each training embedding
        # embedding_norm[j] = sum(embedding[:, j] ** 2)
        embedding_norm = (sampled_train_embeddings * sampled_train_embeddings).sum(dim=0)  # (num_total_activations,)
    
        # 2. Compute the distance-squared matrix
        # dist_sq[i, j] = ||c_i||^2 + ||x_j||^2 - 2 * (c_i^T x_j)
        # Top-k w.r.t. squared distances is the same as top-k w.r.t. distances
        dist_sq = concept_norm.unsqueeze(1) + embedding_norm.unsqueeze(0) - 2.0 * cx_dot
        #print(f"dist_sq : {dist_sq}")
        # dist_sq: (n_concepts, num_total_activations)
    
        # 3. For each concept (each row), find the k smallest distances
        #    largest=False -> we want the smallest distances
        _, knn_indices = dist_sq.topk(k, dim=1, largest=False)  # (n_concepts, k)
    
        # 4. Gather the dot products for these k neighbors
        #    shape = (n_concepts, k)
        knn_dot = torch.gather(cx_dot, dim=1, index=knn_indices)
    
        # 5. Compute the average dot product over k neighbors, and then average over n_concepts
        #    This replicates what your loop was doing: sum(c * c_knn)/k per concept, then average across concepts.
        #    shape of knn_dot.mean(dim=1) = (n_concepts,)
        L_sparse_1_new = knn_dot.mean(dim=1).mean(dim=0)  # scalar
    
        return L_sparse_1_new

    # Reconstruct the activations from the concepts activations (by calling a MLP)
    def reconstruct_from_concepts_activations(self,concept_score_thres_prob):

        rec_layer_1 = F.relu(torch.mm(concept_score_thres_prob, self.rec_vector_1))
        rec_layer_2 = torch.mm(rec_layer_1, self.rec_vector_2) 

        return rec_layer_2
        
    # Reconstruct the activations from the original hidden state activations
    def reconstruct(self,train_embedding):

        concept_score_thres_prob = self.concepts_activations(train_embedding)

        reconstructed_activations = self.reconstruct_from_concepts_activations(concept_score_thres_prob)

        return reconstructed_activations, concept_score_thres_prob
    
    # Compute the concepts activations from the original hidden state activations
    def concepts_activations(self,train_embedding):

        concept_vector = self.concept
        train_embedding = train_embedding.type_as(concept_vector)
        concept_vector_n = F.normalize(concept_vector, dim = 0, p=2)
        train_embedding_n = F.normalize(train_embedding, dim = 1, p=2)
        concept_score_n = torch.mm(train_embedding_n, concept_vector_n)
        concept_score = torch.mm(train_embedding, concept_vector)
        concept_score_mask = torch.gt(concept_score_n, self.thres)
        concept_score_thres = concept_score_mask * concept_score
        concept_score_sum = torch.sum(concept_score_thres, axis = 1, keepdim=True)+1e-3
        concept_score_thres_prob = concept_score_thres / concept_score_sum

        return concept_score_thres_prob
        
    
    def forward(self, train_embedding, hook_model, hook_layer, topk):
        """
        train_embedding: shape (bs, embedding_dim)
        """

        for p in hook_model.parameters():
            p.requires_grad = False

        rec_layer_2,_ = self.reconstruct(train_embedding)
        
        rec_sentence_embedding = rec_layer_2.unsqueeze(1)
        train_embedding = train_embedding.unsqueeze(1)

    
        logits_reconstruction = hook_model.run_with_hooks(
                rec_sentence_embedding,
                start_at_layer=hook_layer,
                return_type="logits",
        )
        logits_reconstruction = logits_reconstruction.squeeze(1)

        logits_original = hook_model.run_with_hooks(
                train_embedding,
                start_at_layer=hook_layer,
                return_type="logits",
        )
        logits_original = logits_original.squeeze(1)

        # For the first regularizer term
        k = topk 
        self.train_embeddings = self.train_embeddings.to(self.concept.device)
        
        # Shuffle embeddings
        sampled_indices = torch.randperm(self.train_embeddings.shape[1])
        sampled_train_embeddings = self.train_embeddings[:,sampled_indices]
      
        # Compute first regularizer term
        L_sparse_1_new = self.efficient_knn_dot_avg(sampled_train_embeddings, k) 

        # Compute second regularizer term
        gram = self.concept.T @ self.concept # dot product between concepts vectors
        n = gram.shape[0]
        mask = ~torch.eye(n, device=gram.device).bool()
        off = gram.masked_select(mask)       # same off-diagonals
        L_sparse_2_new = (off ** 2).mean()

        return logits_original, logits_reconstruction, L_sparse_1_new, L_sparse_2_new

    def loss(self, train_embedding, train_y_true, regularize, labels_tokens_id, l_1, l_2, topk,nb_classes, hook_model, hook_layer):

        orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new = self.forward(train_embedding, hook_model, hook_layer, topk)
    
        train_y_true = train_y_true.to(hook_model.cfg.device)
        
        # Filter predictions based on the logits corresponding to an accepted answer
        ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                               key=lambda kv: kv[1])]
    
        orig_pred = orig_pred[:,ordered_old_idxs]
        y_pred = y_pred[:,ordered_old_idxs]


        ce_loss = nn.CrossEntropyLoss()
        loss_new = ce_loss(y_pred,train_y_true)
        pred_loss = torch.mean(loss_new)

        # completeness score
        def n(y_pred):
            orig_correct = torch.sum(train_y_true == torch.argmax(orig_pred, axis=1))
            new_correct = torch.sum(train_y_true == torch.argmax(y_pred, axis=1))
            return torch.div(new_correct - (1/nb_classes), orig_correct - (1/nb_classes))

        completeness = n(y_pred)

        if regularize:
            final_loss = pred_loss + (l_1 * L_sparse_1_new * -1) + (l_2 * L_sparse_2_new)
        else:
            final_loss = pred_loss

        return completeness, final_loss, pred_loss, L_sparse_1_new, L_sparse_2_new



def set_seed(seed: int):
   
    #  OS hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    #  NumPy RNG
    np.random.seed(seed)
    #  PyTorch CPU RNG
    torch.manual_seed(seed)
    #  PyTorch GPU RNG (for single-GPU & multi-GPU)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #  cuDNN backend: deterministic, no auto-benchmarking
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_plot(list_values,title,path):

    plt.figure(figsize=(8, 5))
    plt.plot(list_values, marker='o', linestyle='-', color='b', label='Values')
    
    # Adding labels and title
    plt.title(title, fontsize=14)
    plt.xlabel('Batch Iterations', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    
    # Adding a grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adding a legend
    plt.legend()
    
    file_path = os.path.join(path,f'{title}.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')



def concept_shap_train(cfg_baseline_method: BaselineMethodConfig, cfg_model : LLMLoadConfig):
 
   
    tensors = []
    # Load the classifier LLM cached activations
    logger.info(f"Loading the activations from {cfg_baseline_method.activations_path}")
    for file_name in os.listdir(cfg_baseline_method.activations_path):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(cfg_baseline_method.activations_path, file_name)
            # Load the safetensors file (assuming a single tensor per file)
            tensor_data = load_file(file_path)  # Returns a dictionary
            for key, tensor in tensor_data.items():
                tensors.append(tensor)

    
    layer_activations = torch.cat(tensors, dim=0)  # Concatenate along the first axis (n1+n2+...+nN) (N,1,d+1)

   
    layer_activations_without_labels = layer_activations[:,0,:-1]
    labels = layer_activations[:,0,-1].long()

    l_1 = cfg_baseline_method.baseline_method_args['l1']
    l_2 = cfg_baseline_method.baseline_method_args['l2']
    topk = cfg_baseline_method.baseline_method_args['topk']
    batch_size = cfg_baseline_method.baseline_method_args['batch_size']
    epochs = cfg_baseline_method.baseline_method_args['epochs']
    hidden_dim = cfg_baseline_method.baseline_method_args['hidden_dim']
    thres = cfg_baseline_method.baseline_method_args['thres']
    nb_classes = cfg_baseline_method.baseline_method_args['nb_classes']

    loss_reg_epoch = cfg_baseline_method.baseline_method_args['loss_reg_epoch']

    n_concepts = cfg_baseline_method.baseline_method_args['n_concepts']
    hook_layer = cfg_baseline_method.hook_layer

    # Load tokenizer to access the vocab ids and the model to obtain the new and original logits for the completeness loss
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    hook_model = get_hook_model(cfg_model,tokenizer)
    logger.info(f"Get HookedTransfomer model loaded")

    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(np.array(labels))
    keys_labels = set(unique_labels)
    labels_tokens_id = { vocab[str(key)] : key for key in keys_labels if str(key) in vocab}

    set_seed(cfg_baseline_method.baseline_method_args['seed'])

    concept_net = ConceptNet(n_concepts, layer_activations_without_labels, hidden_dim,thres).to(hook_model.cfg.device)
    optimizer = torch.optim.Adam(concept_net.parameters(), lr=1e-3)

    train_size = layer_activations_without_labels.shape[0]

    n_iter = 0

    sum_losses = []
    pred_losses = []
    l1_losses = []
    l2_losses = []
    completenesses = []
    
    for i in tqdm(range(epochs)):
        if i < loss_reg_epoch:
          regularize = False
        else:
          regularize = True

        batch_start = 0
        batch_end = batch_size

        # do a shuffle of train_embeddings, train_y_true
        new_permute = torch.randperm(train_size)
        layer_activations = layer_activations[new_permute]
        permuted_train_embeddings = layer_activations[:, 0 ,:-1]
        permuted_train_y_true = layer_activations[:,0,-1].long()

        while batch_end < train_size:
          # generate training batch
          train_embeddings_narrow = permuted_train_embeddings.narrow(0, batch_start, batch_end - batch_start).to(hook_model.cfg.device)
          train_y_true_narrow = permuted_train_y_true.narrow(0, batch_start, batch_end - batch_start).to(hook_model.cfg.device)          

          completeness, final_loss, pred_loss, l1, l2 = concept_net.loss(train_embeddings_narrow,
                                                                   train_y_true_narrow,
                                                                   regularize=regularize,
                                                                   labels_tokens_id=labels_tokens_id,
                                                                   l_1=l_1, l_2=l_2, topk=topk,
                                                                   nb_classes=nb_classes,
                                                                   hook_model=hook_model,
                                                                   hook_layer=hook_layer)

            
        
          # update gradients
          optimizer.zero_grad()
          final_loss.backward()
          optimizer.step()

        
          sum_losses.append(final_loss.data.item())
          pred_losses.append(pred_loss.data.item())
          l1_losses.append(l1.data.item())
          l2_losses.append(l2.data.item())
          completenesses.append(completeness.data.item())

    
          # update batch indices
          batch_start += batch_size
          batch_end += batch_size
          n_iter += 1


    

    save_plot(sum_losses,'Total loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(pred_losses,'Prediction loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(l1_losses,'L1 loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(l2_losses,'L2 loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(completenesses,'Completenesses',cfg_baseline_method.metrics_reconstruction)
    

    # Create directory and save the ConceptNet model 
    if not os.path.isdir(f'{cfg_baseline_method.path_to_baseline_methods}_{thres}'):
        os.mkdir(f'{cfg_baseline_method.path_to_baseline_methods}_{thres}')
    torch.save(concept_net.state_dict(), os.path.join(f'{cfg_baseline_method.path_to_baseline_methods}_{thres}',f'conceptshap_weights.pth') )
    logger.info(f"Fitted {cfg_baseline_method.method_name} weights are saved in {cfg_baseline_method.path_to_baseline_methods}_{thres}")
        


def ica_fitting(cfg_baseline_method: BaselineMethodConfig):
 
    tensors = []
    # Load the classifier LLM cached activations
    logger.info(f"Loading the activations from {cfg_baseline_method.activations_path}")
    for file_name in os.listdir(cfg_baseline_method.activations_path):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(cfg_baseline_method.activations_path, file_name)
            # Load the safetensors file (assuming a single tensor per file)
            tensor_data = load_file(file_path)  # Returns a dictionary
            for key, tensor in tensor_data.items():
                tensors.append(tensor)

   
    layer_activations = torch.cat(tensors, dim=0)  # Concatenate along the first axis (n1+n2+...+nN)
    
    # No need to get the predicted labels for ICA, the method is unsupervised
    if cfg_baseline_method.with_label:
        layer_activations = layer_activations[:,0,:-1]
    else:
        layer_activations = layer_activations.squeeze(1).numpy()    

    
    ica_method = FastICA(**cfg_baseline_method.baseline_method_args)

    # ICA fitting ~ Find candidate concept vectors
    independent_components = ica_method.fit_transform(layer_activations)
    logger.info(f"{independent_components.shape[1]} independent components have been fitted by ICA.")

    # Get the mixing matrix
    mixing_matrix = ica_method.mixing_

    # Create directory and save the ICA model 
    if not os.path.isdir(cfg_baseline_method.path_to_baseline_methods):
        os.mkdir(cfg_baseline_method.path_to_baseline_methods)
    np.save(os.path.join(cfg_baseline_method.path_to_baseline_methods,'mixing_matrix.npy'), mixing_matrix) 
    joblib.dump(ica_method,  os.path.join(cfg_baseline_method.path_to_baseline_methods,'ica.pkl'))
    logger.info(f"Fitted {cfg_baseline_method.method_name} is saved in {cfg_baseline_method.path_to_baseline_methods}")



def baseline_concept_method_train(config_baseline_method: str, config_llm_classifier : Optional[str] = None):

    # Retrieve the config of the baseline method
    cfg_baseline_method = BaselineMethodConfig.autoconfig(config_baseline_method)

    if cfg_baseline_method.method_name == 'concept_shap':

        # ConceptSHAP needs to load the LLM classifier since the training requires to compare the original predictions (already cached) with the predictions made by the classifier when the original hidden state goes through the concepts layer bottleneck 
        if config_llm_classifier is None:
            raise ValueError(f"You use the ConceptSHAP baseline which requires to load the original classifier. You should pass a model config with the --config-model option.")
        
        # Retrieve the config of the model
        cfg_model = LLMLoadConfig.autoconfig(config_llm_classifier)

        concept_shap_train(cfg_baseline_method, cfg_model)
    
    elif cfg_baseline_method.method_name == 'ica':

        # Currently, the only other baseline is ICA which does not require to load a LLM classifier config. We assume the activations of the studied layer are already cached.
        ica_fitting(cfg_baseline_method)













