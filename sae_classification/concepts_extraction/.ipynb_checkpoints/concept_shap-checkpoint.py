from typing import Optional
import os
import torch
import numpy as np
from safetensors.torch import load_file
from loguru import logger
from ..utils import DRMethodsConfig, LLMLoadConfig
from ..model_training import process_dataset,get_hook_model
from transformers import AutoTokenizer
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
import torch.nn as nn
import matplotlib.pyplot as plt
import time





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


    # def sparse_loss_vectorized(self, sampled_train_embeddings, k):
    #     """
    #     Vectorized version that computes:
    #     1) top-k NN for each concept
    #     2) average dot-product with each concept
    #     3) final mean over all concepts
    #     """
    
    #     # Suppose:
    #     #   self.concept has shape [activation_dim, n_concepts]
    #     #   sampled_train_embeddings has shape [activation_dim, n_activations]
    
    #     # 1) Compute all pairwise distances using torch.cdist (L2 distance)
    #     #    shape of distances => [n_concepts, n_activations]

    #     print(f"self.concept.T shape : {self.concept.T.shape}")
    #     print(f"sampled_train_embeddings.T shape : {sampled_train_embeddings.T.shape}")
    #     print(f"self.concept.T : {self.concept.T}")
        
    #     distances = torch.cdist(
    #         self.concept.T.to(dtype=sampled_train_embeddings.dtype),
    #         sampled_train_embeddings.T,
    #         p=2
    #     )

    #     print(f"distances : {distances}")
    
    #     # 2) For each concept (row), pick the top-k nearest neighbors
    #     #    knn.indices has shape => [n_concepts, k]
    #     knn = distances.topk(k, dim=-1, largest=False)
    #     indices = knn.indices
    
    #     # 3) Gather those k nearest embedding vectors for each concept
    #     #    We want a result of shape => [n_concepts, activation_dim, k]
    #     #    We'll expand the train embeddings to [n_concepts, activation_dim, n_activations]
    #     #    then use torch.gather along the last dimension.
    #     expanded_embeddings = sampled_train_embeddings.unsqueeze(0)  # [1, dim, n_activations]
    #     expanded_embeddings = expanded_embeddings.expand(self.n_concepts, -1, -1)  # [n_concepts, dim, n_activations]
        
    #     # Prepare an index tensor of shape => [n_concepts, 1, k] to gather across dim=2
    #     gather_indices = indices.unsqueeze(1).expand(-1, sampled_train_embeddings.shape[0], -1) 
    #     c_knn = torch.gather(expanded_embeddings, 2, gather_indices)  # [n_concepts, dim, k]
    
    #     # 4) Compute dot products between each concept and its k neighbors
    #     #    First, reshape concept to [n_concepts, dim, 1]
    #     concept_expanded = self.concept.T.unsqueeze(-1)  # [n_concepts, dim, 1]
        
    #     #    Elementwise multiply and sum over dim=1 => [n_concepts, k], then average over k
    #     dot_prod = (c_knn * concept_expanded).sum(dim=[1,2]) / k  # => [n_concepts]
    
    #     # 5) Finally, average over concepts
    #     L_sparse_1_new = dot_prod.mean()
    
    #     return L_sparse_1_new
    
    
    def efficient_knn_dot_avg(self,sampled_train_embeddings, k):
        """
        concept:                  (activation_dim, n_concepts)
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


    def reconstruct_from_concepts_activations(concept_score_thres_prob):

        rec_layer_1 = F.relu(torch.mm(concept_score_thres_prob, self.rec_vector_1))
        rec_layer_2 = torch.mm(rec_layer_1, self.rec_vector_2) 

        return rec_layer_2
        
    
    def reconstruct(self,train_embedding):

        concept_score_thres_prob = self.concepts_activations(train_embedding)

        reconstructed_activations = self.reconstruct_from_concepts_activations(concept_score_thres_prob)

        return reconstructed_activations, concept_score_thres_prob
        
    def concepts_activations(self,train_embedding):

        concept_vector = self.concept
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
            hook_model.requires_grad = False

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


        # Calculate the regularization terms as in new version of paper
        k = topk # this is a tunable parameter

        self.train_embeddings = self.train_embeddings.to(self.concept.device)
        
        #To gain efficiency
        sampled_indices = torch.randperm(self.train_embeddings.shape[1])[:1000]
        sampled_train_embeddings = self.train_embeddings[:,sampled_indices]
        #print(f"sampled_train_embeddings shape : {sampled_train_embeddings.shape}")
        

        # # start = time.time()
        # ### calculate first regularization term, to be maximized
        # # 1. find the top k nearest neighbour
        # all_concept_knns = []
        # for concept_idx in range(self.n_concepts):
        #     c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)    
        #     # euc dist
        #     distance = torch.norm(sampled_train_embeddings - c, dim=0) # (num_total_activations)
        #     knn = distance.topk(k, largest=False)
        #     indices = knn.indices # (k)
        #     knn_activations = sampled_train_embeddings[:, indices] # (activation_dim, k)
        #     all_concept_knns.append(knn_activations)

        # print(f"knn_activations device : {knn_activations.device}")
        
        # # 2. calculate the avg dot product for each concept with each of its knn
        # L_sparse_1_new = 0.0
        # for concept_idx in range(self.n_concepts):
        #     c = self.concept[:, concept_idx].unsqueeze(dim=1) # (activation_dim, 1)
        #     c_knn = all_concept_knns[concept_idx] # knn for c
        #     dot_prod = torch.sum(c * c_knn) / k # avg dot product on knn
        #     L_sparse_1_new += dot_prod
        # L_sparse_1_new = L_sparse_1_new / self.n_concepts
        
        
        #start = time.time()
        #L_sparse_1_new = self.sparse_loss_vectorized(sampled_train_embeddings, k)   
        # stop = time.time()
        # duration = stop-start
        # print(f"new duration : {duration}")
        # print(f"L_sparse_1_new : {L_sparse_1_new}")
        #print(f"L_sparse_1_new : {L_sparse_1_new}")
        #start = time.time()
        L_sparse_1_new = self.efficient_knn_dot_avg(sampled_train_embeddings, k) 
        #stop = time.time()
        #duration = stop-start
        #print(f"new duration : {duration}")
        # print(f"L_sparse_1_new : {L_sparse_1_new}")
        # print(f"L_sparse_1_alt : {L_sparse_1_alt}")

        
        ### calculate Second regularization term, to be minimized
        all_concept_dot = self.concept.T @ self.concept
        mask = torch.eye(self.n_concepts).cuda() * -1 + 1 # mask the i==j positions
        L_sparse_2_new = torch.mean(all_concept_dot * mask)

        #norm_metrics = torch.mean(all_concept_dot * torch.eye(self.n_concepts).cuda())

        return logits_original, logits_reconstruction, L_sparse_1_new, L_sparse_2_new

    def loss(self, train_embedding, train_y_true, regularize, l_1, l_2, topk, lookup_table,nb_class, hook_model, hook_layer):
        """
        This function will be called externally to feed data and get the loss
        """
        # Note: it is important to MAKE SURE L2 GOES DOWN! that will let concepts separate from each other

        # total_start = time.time()
        
       
        orig_pred, y_pred, L_sparse_1_new, L_sparse_2_new = self.forward(train_embedding, hook_model, hook_layer, topk)
        end = time.time()
        
       
        train_y_true = lookup_table[train_y_true.cpu()]
        train_y_true = train_y_true.to(hook_model.cfg.device)
        # print(f"train_y_true shape : {train_y_true.shape}")
        # print(f"y_pred shape : {y_pred.shape}")
        
        ce_loss = nn.CrossEntropyLoss()
        loss_new = ce_loss(y_pred,train_y_true)
        pred_loss = torch.mean(loss_new)

        # completeness score
        def n(y_pred):
            orig_correct = torch.sum(train_y_true == torch.argmax(orig_pred, axis=1))
            new_correct = torch.sum(train_y_true == torch.argmax(y_pred, axis=1))
            return torch.div(new_correct - (1/nb_class), orig_correct - (1/nb_class))

        completeness = n(y_pred)

        if regularize:
            final_loss = pred_loss + (l_1 * L_sparse_1_new * -1) + (l_2 * L_sparse_2_new)
        else:
            final_loss = pred_loss

        # total_stop = time.time()
        # total_duration = total_stop - total_start
        # print(f"total_duration : {total_duration}")

        return completeness, final_loss, pred_loss, L_sparse_1_new, L_sparse_2_new



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



def concept_shap_train(config_dr_method: str, config_model : Optional[str] = None):
 
    #Retrieve the config of the DR method
    cfg_dr_method = DRMthodsConfig.autoconfig(config_dr_method)

    #Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)

    
    tensors = []
    for file_name in os.listdir(cfg_dr_method.activations_path):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(cfg_dr_method.activations_path, file_name)
            # Load the safetensors file (assuming a single tensor per file)
            tensor_data = load_file(file_path)  # Returns a dictionary
            for key, tensor in tensor_data.items():
                tensors.append(tensor)

   
    layer_activations = torch.cat(tensors, dim=0)  # Concatenate along the first axis (n1+n2+...+nN) (N,1,d+1)
   
    labels = layer_activations[:,0,-1]
    print(f"path : {cfg_dr_method.activations_path}")
    print(f"labels : {labels}")
    layer_activations = layer_activations[:,0,:-1]

    l_1 = cfg_dr_method.dr_methods_args['l1']
    l_2 = cfg_dr_method.dr_methods_args['l2']
    topk = cfg_dr_method.dr_methods_args['topk']
    batch_size = cfg_dr_method.dr_methods_args['batch_size']
    epochs = cfg_dr_method.dr_methods_args['epochs']
    hidden_dim = cfg_dr_method.dr_methods_args['hidden_dim']
    thres = cfg_dr_method.dr_methods_args['thres']

    loss_reg_epoch = cfg_dr_method.dr_methods_args['loss_reg_epoch']
    losses = []

    n_concepts = cfg_dr_method.dr_methods_args['n_concepts']
    hook_layer = cfg_dr_method.hook_layer

    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    cfg_model.task_args['prompt_tuning'] = False
    hook_model = get_hook_model(cfg_model,tokenizer)

    concept_net = ConceptNet(n_concepts, layer_activations, hidden_dim,thres).to(hook_model.cfg.device)
    optimizer = torch.optim.Adam(concept_net.parameters(), lr=1e-3)
    train_embeddings = layer_activations.to(hook_model.cfg.device)
    train_y_true = labels.to(torch.int).to(hook_model.cfg.device)

    train_size = train_embeddings.shape[0]

    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(train_y_true.cpu().numpy())
    keys_labels = set(unique_labels)
    str_keys_labels = {str(i) for i in keys_labels}
    #print(f"str_keys_labels : {str_keys_labels}")
    #print(f"vocab : {vocab}")
    
    match_token_label_class = { int(key) : value for key,value in vocab.items() if key in str_keys_labels }
    print(f"match_token_label_class : {match_token_label_class}")

    max_key = max(match_token_label_class.keys())
    lookup_table = torch.full((max_key + 1,), -1, dtype=torch.long)
    for key, val in match_token_label_class.items():
        lookup_table[key] = val

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
        train_y_true_float = train_y_true.float().unsqueeze(dim=1)
        data_pair = torch.cat([train_embeddings, train_y_true_float], dim=1)
        new_permute = torch.randperm(data_pair.shape[0])
        data_pair = data_pair[new_permute]
        permuted_train_embeddings = data_pair[:, :-1]
        permuted_train_y_true = data_pair[:, -1].long()

        while batch_end < train_size:
          # generate training batch
          train_embeddings_narrow = permuted_train_embeddings.narrow(0, batch_start, batch_end - batch_start)
          train_y_true_narrow = permuted_train_y_true.narrow(0, batch_start, batch_end - batch_start)
          
          completeness, final_loss, pred_loss, l1, l2 = concept_net.loss(train_embeddings_narrow,
                                                                   train_y_true_narrow,
                                                                   regularize=regularize,
                                                                   l_1=l_1, l_2=l_2, topk=topk,
                                                                   lookup_table=lookup_table,
                                                                   nb_class=len(unique_labels),
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


    

    save_plot(sum_losses,'Total loss',cfg_dr_method.metrics_reconstruction)
    save_plot(pred_losses,'Prediction loss',cfg_dr_method.metrics_reconstruction)
    save_plot(l1_losses,'L1 loss',cfg_dr_method.metrics_reconstruction)
    save_plot(l2_losses,'L2 loss',cfg_dr_method.metrics_reconstruction)
    save_plot(completenesses,'Completenesses',cfg_dr_method.metrics_reconstruction)
    

    torch.save(concept_net.state_dict(), f'{cfg_dr_method.path_to_dr_methods}_{thres}.pth')
    logger.info(f"Fitted {cfg_dr_method.method_name} weights are saved in {cfg_dr_method.path_to_dr_methods}_{thres}.pth")
        


 
    












