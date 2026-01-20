from typing import Optional
import os
import torch
import numpy as np
from safetensors.torch import load_file
from loguru import logger
from ..utils import BaselineMethodConfig, LLMLoadConfig
from ..llm_classifier_tuning import get_hook_model, get_model, set_seed, _init_tokenizer
from sklearn.decomposition import FastICA
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import joblib


# Concepts discovery method proposed in HI_Concept
class HIConcept(nn.Module):
    
    def __init__(self, n_concepts, embedding_dim, hidden_dim):
        super(HIConcept,self).__init__()
                
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True) # Shape = (embedding_dim, n_concept)
        self.n_concepts = n_concepts 
        
        self.rec_vector_1 = nn.Parameter(self.init_concept(n_concepts, hidden_dim), requires_grad = True)
        self.rec_vector_2 = nn.Parameter(self.init_concept(hidden_dim,embedding_dim), requires_grad = True)

        self.thres = 1/(self.n_concepts)
        self.criterion = nn.CrossEntropyLoss()
        self.ae_criterion = nn.MSELoss()

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept

    def ae_loss(self, xpred, xin):
        return self.ae_criterion(xpred, xin)
    
    def flip_loss(self, y_pred, y_pred_perturbed):
        
        y_pred = F.softmax(y_pred, dim = 1)
        y_pred_perturbed = F.softmax(y_pred_perturbed, dim = 1)
        # Encourage large distributional change across all classes
        return -torch.mean(torch.sum(torch.abs(y_pred - y_pred_perturbed),dim=1))
        
        
    ## R1 in ConceptSHAP 
    def concept_sim(self, concept_score_n):
        # maximize the top k 
        batch_size = concept_score_n.shape[0]
        res = torch.reshape(concept_score_n,(-1,self.n_concepts))
        res = torch.transpose(res,0,1)
        k = max(1, batch_size // 4)
        res = torch.topk(res,k=k,sorted=True).values
        res = torch.mean(res)
        return - res
    
    
    ## L2 in ConceptSHAP. The concepts are L2 normalizes, so it means we compute the mean cosine similarity
    def concept_far(self, concept_vector_n):
        # topic_vector_n: #hidden_dim, n_concepts
        # after norm: n_concepts, n_concepts
        return torch.mean(torch.mm(torch.transpose(concept_vector_n, 0, 1), concept_vector_n) - torch.eye(self.n_concepts).to(concept_vector_n.device))
    
        
    def concepts_activations(self,f_input,perturb=-1):

        concept_vector = self.concept # Shape (embedding_dim, n_concepts)
        concept_vector_n = F.normalize(concept_vector, dim = 0, p=2)
        
        f_input = f_input.type_as(concept_vector)
        f_input_n = F.normalize(f_input, dim = 1, p=2)
        
        concept_score = torch.mm(f_input, concept_vector_n)
        concept_score_n = torch.mm(f_input_n, concept_vector_n)
        concept_score_n = F.softmax(concept_score_n, dim = -1)

        thres_tensor = torch.tensor(self.thres, device=concept_score_n.device, dtype=concept_score_n.dtype)
        concept_score_mask =  (concept_score_n > thres_tensor)
        
        concept_score_thres = concept_score_mask * concept_score

        if perturb >=0:
            concept_score_thres[:, perturb] = 0
        
        concept_score_sum = torch.sum(concept_score_thres, axis = 1, keepdim=True)+1e-3
        concept_score_thres_prob = concept_score_thres / concept_score_sum
        
        return concept_score_thres_prob, f_input_n, concept_score_n, concept_score,concept_vector_n

    
    def reconstruct_from_concepts_activations(self,concept_score_thres_prob):

        rec_layer_1 = F.relu(torch.mm(concept_score_thres_prob, self.rec_vector_1))
        rec_layer_2 = torch.mm(rec_layer_1, self.rec_vector_2) 

        return rec_layer_2

    # f_train = all training embeddings dataset
    # f_input : batch - subset of f_train
    # topk fixed to Batch size // 4
    def forward(self, f_input, causal, model, hook_layer, labels_tokens_id, perturb = -1, one_correlated_dimension=True, random_masking_prob=0.2):
        
        concept_score_thres_prob, f_input_n, concept_score_n, concept_score, concept_vector_n = self.concepts_activations(f_input, perturb)
        
        reconstructed_activations = self.reconstruct_from_concepts_activations(concept_score_thres_prob)
    

        ae_loss = self.ae_loss(f_input_n, reconstructed_activations)
        
        rec_sentence_embedding = reconstructed_activations.unsqueeze(1)

        # Filter predictions based on the logits corresponding to an accepted answer
        ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                               key=lambda kv: kv[1])]
            
        logits_reconstruction = model.run_with_hooks(
                rec_sentence_embedding,
                start_at_layer=hook_layer,
                return_type="logits",
        )

        logits_reconstruction = logits_reconstruction.squeeze(1)
               
        y_pred = logits_reconstruction[:,ordered_old_idxs]
        
        
        # Regularizer 1 in ConceptSHAP
        concept_sim = self.concept_sim(concept_score_n) 


        # Regularize 2 in ConceptSHAP
        concept_far = self.concept_far(concept_vector_n) 
        

        if causal != True:
            flip_loss = torch.zeros((), device=ae_loss.device,requires_grad=True)
            return y_pred, flip_loss, concept_sim, concept_far, concept_score_thres_prob, ae_loss
        # Causal
        else:
            
            if one_correlated_dimension == True:
                # original_score_n_last_dim = concept_score_n[:, -1]
                concept_score_n = concept_score_n[:, :-1]
            
            # By default random masking for the causal as it is defined as default in the original HIConcept github repo
            concept_score_mask_far = (torch.rand_like(concept_score_n) > random_masking_prob)

            if one_correlated_dimension == True:
                
                concept_score_mask_new = torch.ones_like(concept_score)
                concept_score_mask_new[:, :-1] = concept_score_mask_far
                concept_score_mask_far = concept_score_mask_new
                
            concept_score_far = concept_score * concept_score_mask_far # topic_prob with zeroed out dimension
            concept_prob_sum_far = torch.sum(concept_score_far, axis=-1, keepdims=True)+1e-3 #bs, 1 #sum of the topic probabilities per instance
            concept_prob_nn_far = concept_score_far/concept_prob_sum_far # normalize probabilities
            
            rec_layer_1_far = F.relu(torch.mm(concept_prob_nn_far, self.rec_vector_1))
            rec_layer_2_far = torch.mm(rec_layer_1_far, self.rec_vector_2)
            
            rec_layer_2_far = rec_layer_2_far.unsqueeze(1)
            
            logits_perturbed = model.run_with_hooks(
                rec_layer_2_far,
                start_at_layer=hook_layer,
                return_type="logits",
            )
            logits_perturbed = logits_perturbed.squeeze(1)
            
            y_pred_perturbed = logits_perturbed[:,ordered_old_idxs]
            
            flip_loss = self.flip_loss(y_pred, y_pred_perturbed)

            
            return y_pred, flip_loss, concept_sim, concept_far, concept_score_thres_prob, ae_loss


    def loss(self, f_input, label_input, labels_tokens_id, causal, nb_classes, model, hook_layer):
        
        y_pred, flip_loss, concept_sim, concept_far, _, ae_loss  = self.forward(f_input, causal, model, hook_layer, labels_tokens_id)
        
        criterion = nn.CrossEntropyLoss()
        pred_loss = criterion(y_pred,label_input)
        pred_loss = torch.mean(pred_loss)
        
        return pred_loss, ae_loss, flip_loss, concept_sim, concept_far



class HIConceptEncoder(HIConcept):

    def classifier(self,hidden,f_input,model,hook_layer):

        hidden_norm = hidden.norm(p=2,dim=-1).clamp_min(1e-8)
        f_input_norm = f_input.norm(p=2,dim=-1)
        factor = f_input_norm / hidden_norm.squeeze(1)
        hidden = hidden * factor[:,None,None]


        base = getattr(model, model.base_model_prefix)
        encoder = base.encoder                         # BERT->bert.encoder, debrta->deberta.encoder
        
        # Build attention mask locally 
        attn_mask = torch.ones(hidden.size()[:2], dtype=torch.long, device=hidden.device)
        extended = model.get_extended_attention_mask(
            attention_mask=attn_mask,
            input_shape=attn_mask.shape,
            device=hidden.device,
        )
        
        for idx in range(hook_layer, model.config.num_hidden_layers):
            hidden = encoder.layer[idx](hidden, attention_mask=extended, output_attentions=False)[0]
            
        pooler = getattr(model, "pooler", None) or getattr(base, "pooler", None)
        pooled = pooler(hidden) if pooler is not None else hidden[:, 0]

        y_pred = model.classifier(model.dropout(pooled))

        return y_pred


    def forward(self, f_input, causal, model, hook_layer, labels_tokens_id, perturb = -1, one_correlated_dimension=True, random_masking_prob=0.2):
        
        concept_score_thres_prob, f_input_n, concept_score_n, concept_score, concept_vector_n = self.concepts_activations(f_input, perturb)
        
        reconstructed_activations = self.reconstruct_from_concepts_activations(concept_score_thres_prob)
    
        ae_loss = self.ae_loss(f_input_n, reconstructed_activations)
        
        rec_sentence_embedding = reconstructed_activations.unsqueeze(1)
        
        # B, T, H = reconstruct_act.shape
        hidden = rec_sentence_embedding
    
        y_pred = self.classifier(hidden,f_input,model,hook_layer)
                  
        # Regularizer 1 in ConceptSHAP
        concept_sim = self.concept_sim(concept_score_n) 


        # Regularize 2 in ConceptSHAP
        concept_far = self.concept_far(concept_vector_n) 
        

        if causal != True:
            flip_loss = torch.zeros((), device=ae_loss.device,requires_grad=True)
            return y_pred, flip_loss, concept_sim, concept_far, concept_score_thres_prob, ae_loss
        # Causal
        else:
            
            if one_correlated_dimension == True:
                original_score_n_last_dim = concept_score_n[:, -1]
                concept_score_n = concept_score_n[:, :-1]
            
            # By default random masking for the causal as it is defined as default in the original HIConcept github repo
            concept_score_mask_far = (torch.rand_like(concept_score_n) > random_masking_prob)

            if one_correlated_dimension == True:
                
                concept_score_mask_new = torch.ones_like(concept_score)
                concept_score_mask_new[:, :-1] = concept_score_mask_far
                concept_score_mask_far = concept_score_mask_new
                
            concept_score_far = concept_score * concept_score_mask_far # topic_prob with zeroed out dimension
            concept_prob_sum_far = torch.sum(concept_score_far, axis=-1, keepdims=True)+1e-3 #bs, 1 #sum of the topic probabilities per instance
            concept_prob_nn_far = concept_score_far/concept_prob_sum_far # normalize probabilities
            
            rec_layer_1_far = F.relu(torch.mm(concept_prob_nn_far, self.rec_vector_1))
            rec_layer_2_far = torch.mm(rec_layer_1_far, self.rec_vector_2)
            
            rec_layer_2_far = rec_layer_2_far.unsqueeze(1)

            # Prediction 
            hidden = rec_layer_2_far
            y_pred_perturbed = self.classifier(hidden,f_input,model,hook_layer)
    
            flip_loss = self.flip_loss(y_pred, y_pred_perturbed)

            
            return y_pred, flip_loss, concept_sim, concept_far, concept_score_thres_prob, ae_loss




def hi_concept_train(cfg_baseline_method: BaselineMethodConfig, cfg_model : LLMLoadConfig):
 
    print("\n######################################## BEGIN : Concepts-based explainability method - Training HI-Concept  ########################################")

    layer_activations = get_layer_activations(cfg_baseline_method)
    layer_activations_without_labels = layer_activations[:,0,:-1]
    labels = layer_activations[:,0,-1].long()

    r_1 = cfg_baseline_method.baseline_method_args['r1']
    r_2 = cfg_baseline_method.baseline_method_args['r2']
    ae_loss_reg = cfg_baseline_method.baseline_method_args['ae_loss_reg']
    pred_loss_reg = cfg_baseline_method.baseline_method_args['pred_loss_reg']
    flip_loss_reg = cfg_baseline_method.baseline_method_args['flip_loss_reg']
    
    batch_size = cfg_baseline_method.baseline_method_args['batch_size']
    epochs = cfg_baseline_method.baseline_method_args['epochs']
    hidden_dim = cfg_baseline_method.baseline_method_args['hidden_dim']
    nb_classes = cfg_baseline_method.baseline_method_args['nb_classes']

    loss_reg_epoch = cfg_baseline_method.baseline_method_args['loss_reg_epoch']

    n_concepts = cfg_baseline_method.baseline_method_args['n_concepts']
    hook_layer = cfg_baseline_method.hook_layer

    tokenizer, decoder = _init_tokenizer(cfg_model)

    if decoder:
        model = get_hook_model(cfg_model,tokenizer)
        vocab = tokenizer.get_vocab()
        unique_labels = np.unique(np.array(labels))
        keys_labels = set(unique_labels)
        labels_tokens_id = { vocab[str(key)] : key for key in keys_labels if str(key) in vocab}
        device = model.cfg.device
        
        # Ensure all classes are present in vocab and ordering matches 0..nb_classes-1
        if len(labels_tokens_id) != nb_classes:
            missing = set(range(nb_classes)) - set(labels_tokens_id.values())
            raise ValueError(f"Label tokens missing in vocab for classes: {sorted(missing)}")
        
    else:
        model = get_model(cfg_model,decoder,nb_classes) 
        labels_tokens_id = None
        device = model.device

    

    if decoder:
        hi_concept = HIConcept(n_concepts, layer_activations_without_labels.shape[-1], hidden_dim).to(device)
    else:
        hi_concept = HIConceptEncoder(n_concepts, layer_activations_without_labels.shape[-1], hidden_dim).to(device)


    optimizer = torch.optim.Adam(hi_concept.parameters(), lr=3e-4)

    train_size = layer_activations_without_labels.shape[0]

    n_iter = 0

    sum_losses = []
    pred_losses = []
    ae_losses = []
    flip_losses = []
    r1_losses = []
    r2_losses = []
    
    ## As in the original HIConcept paper, we activate the causal loss only at the middle of the training 
    shap_epochs = epochs // 2
    freeze = False
    
    model.eval()
    hi_concept.train()
    
    for epoch_number in tqdm(range(epochs)):
    
        batch_start = 0
        batch_end = batch_size

        # do a shuffle of train_embeddings, train_y_true
        new_permute = torch.randperm(train_size)
        layer_activations = layer_activations[new_permute]
        permuted_train_embeddings = layer_activations[:, 0 ,:-1]
        permuted_train_y_true = layer_activations[:,0,-1].long()

        while batch_end < train_size:
          # generate training batch
          train_embeddings_narrow = permuted_train_embeddings.narrow(0, batch_start, batch_end - batch_start).to(device)
          train_y_true_narrow = permuted_train_y_true.narrow(0, batch_start, batch_end - batch_start).to(device)  
          
          if epoch_number < shap_epochs:
              causal = False
          else:
              causal = True
              if freeze:
                    #in two-stage training, freeze cc other weights except for the topic vector
                    for param in hi_concept.parameters():
                        param.requires_grad = False
                    hi_concept.concept.requires_grad = True
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, hi_concept.parameters()),lr=3e-4)
                    freeze = False

          
          pred_loss, ae_loss, flip_loss, r1_loss, r2_loss = hi_concept.loss(train_embeddings_narrow,
                                                                    train_y_true_narrow,
                                                                    labels_tokens_id=labels_tokens_id,
                                                                    causal=causal,
                                                                    nb_classes=nb_classes,
                                                                    model=model,
                                                                    hook_layer=hook_layer)      
        
          final_loss = ae_loss_reg * ae_loss + pred_loss_reg * pred_loss + flip_loss_reg * flip_loss + r_1 * r1_loss + r_2 * r2_loss
        

          # update gradients
          optimizer.zero_grad()
          final_loss.backward()
          optimizer.step()
        
          sum_losses.append(final_loss.data.item())
          pred_losses.append(pred_loss.data.item())
          ae_losses.append(ae_loss.data.item())          
          flip_losses.append(flip_loss.data.item())          
          r1_losses.append(r1_loss.data.item())
          r2_losses.append(r2_loss.data.item())
        
          # update batch indices
          batch_start += batch_size
          batch_end += batch_size
          n_iter += 1


    save_plot(sum_losses,'Total loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(pred_losses,'Prediction loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(ae_losses,'AE loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(flip_losses,'Flip loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(r1_losses,'R1 loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(r2_losses,'R2 loss',cfg_baseline_method.metrics_reconstruction)
    

    # Create directory and save the HIConcept model 
    os.makedirs(cfg_baseline_method.path_to_baseline_methods, exist_ok=True)
    torch.save(hi_concept.state_dict(), os.path.join(f'{cfg_baseline_method.path_to_baseline_methods}',f'hiconcept_weights.pth') )
    logger.info(f"Fitted {cfg_baseline_method.method_name} weights are saved in {cfg_baseline_method.path_to_baseline_methods}")
        
    print("\n######################################## END : Concepts-based explainability method - Training HI-Concept  ########################################")



# Concepts discovery method proposed in ConceptSHAP
class ConceptNet(nn.Module):

    def __init__(self, n_concepts, embedding_dim, hidden_dim,thres):
        super(ConceptNet, self).__init__()
        # random init using uniform dist
        self.concept = nn.Parameter(self.init_concept(embedding_dim, n_concepts), requires_grad=True)
        self.n_concepts = n_concepts

        self.rec_vector_1 = nn.Parameter(self.init_concept(n_concepts, hidden_dim), requires_grad = True)
        self.rec_vector_2 = nn.Parameter(self.init_concept(hidden_dim, embedding_dim), requires_grad = True)

        self.thres = thres
    

    def init_concept(self, embedding_dim, n_concepts):
        r_1 = -0.5
        r_2 = 0.5
        concept = (r_2 - r_1) * torch.rand(embedding_dim, n_concepts) + r_1
        return concept


    ## R1 in ConceptSHAP 
    def concept_sim(self, concept_score_n):
        # maximize the top k 
        batch_size = concept_score_n.shape[0]
        res = torch.reshape(concept_score_n,(-1,self.n_concepts))
        res = torch.transpose(res,0,1)
        k = max(1, batch_size // 4)
        res = torch.topk(res,k=k,sorted=True).values
        res = torch.mean(res)
        return - res
    
    
    ## L2 in ConceptSHAP. The concepts are L2 normalizes, so it means we compute the mean cosine similarity
    def concept_far(self, concept_vector_n):
        # topic_vector_n: #hidden_dim, n_concept
        # after norm: n_concept, n_concept
        return torch.mean(torch.mm(torch.transpose(concept_vector_n, 0, 1), concept_vector_n) - torch.eye(self.n_concepts).to(concept_vector_n.device))
    

    # Reconstruct the activations from the concepts activations (by calling a MLP)
    def reconstruct_from_concepts_activations(self,concept_score_thres_prob):

        rec_layer_1 = F.relu(torch.mm(concept_score_thres_prob, self.rec_vector_1))
        rec_layer_2 = torch.mm(rec_layer_1, self.rec_vector_2) 

        return rec_layer_2
        
    # Reconstruct the activations from the original hidden state activations
    def reconstruct(self,train_embedding):

        concept_score_thres_prob, concept_score_n, concept_vector_n = self.concepts_activations(train_embedding)

        reconstructed_activations = self.reconstruct_from_concepts_activations(concept_score_thres_prob)

        return reconstructed_activations, concept_score_thres_prob, concept_score_n, concept_vector_n
    

    # Compute the concepts activations from the original hidden state activations
    def concepts_activations(self,train_embedding):

        concept_vector = self.concept
        train_embedding = train_embedding.type_as(concept_vector)

        concept_vector_n = F.normalize(concept_vector, dim = 0, p=2)
        train_embedding_n = F.normalize(train_embedding, dim = 1, p=2)
        
        concept_score = torch.mm(train_embedding, concept_vector_n)
        concept_score_n = torch.mm(train_embedding_n, concept_vector_n)

        concept_score_mask = torch.gt(concept_score_n, self.thres)
        concept_score_thres = concept_score_mask * concept_score

        concept_score_sum = torch.sum(concept_score_thres, axis = 1, keepdim=True)+1e-3
        concept_score_thres_prob = concept_score_thres / concept_score_sum

        return concept_score_thres_prob, concept_score_n, concept_vector_n
        
    
    def forward(self, f_input, model, hook_layer):
        """
        f_input: shape (bs, embedding_dim)
        """

        for p in model.parameters():
            p.requires_grad = False

        rec_layer_2,_,concept_score_n, concept_vector_n = self.reconstruct(f_input)
        
        rec_sentence_embedding = rec_layer_2.unsqueeze(1)
        # rec_sentence_embedding : (bs,1,d_in)
    
        logits_reconstruction = model.run_with_hooks(
                rec_sentence_embedding,
                start_at_layer=hook_layer,
                return_type="logits",
        )
        logits_reconstruction = logits_reconstruction.squeeze(1)


        # Regularizer 1 in ConceptSHAP
        L_sparse_1_new = self.concept_sim(concept_score_n) 
        
        # Regularize 2 in ConceptSHAP
        L_sparse_2_new = self.concept_far(concept_vector_n) 

        return logits_reconstruction, L_sparse_1_new, L_sparse_2_new

    
    def loss(self, train_embedding, train_y_true, regularize, labels_tokens_id, r_1, r_2, nb_classes, model, hook_layer):

        y_pred, L_sparse_1_new, L_sparse_2_new = self.forward(train_embedding, model, hook_layer)
    
        train_y_true = train_y_true.to(y_pred.device)
        

        if labels_tokens_id is not None:
            # Filter predictions based on the logits corresponding to an accepted answer
            ordered_old_idxs = [old for old, new in sorted(labels_tokens_id.items(),
                                                key=lambda kv: kv[1])]
        
            y_pred = y_pred[:,ordered_old_idxs]


        ce_loss = nn.CrossEntropyLoss()
        loss_new = ce_loss(y_pred,train_y_true)
        pred_loss = torch.mean(loss_new)

    
        if regularize:
            final_loss = pred_loss + (r_1 * L_sparse_1_new ) + (r_2 * L_sparse_2_new)
        else:
            final_loss = pred_loss

        return final_loss, pred_loss, L_sparse_1_new, L_sparse_2_new


class ConceptNetEncoder(ConceptNet):

    def classifier(self,hidden,f_input,model,hook_layer):

        hidden_norm = hidden.norm(p=2,dim=-1).clamp_min(1e-8)
        f_input_norm = f_input.norm(p=2,dim=-1)
        factor = f_input_norm / hidden_norm.squeeze(1)
        hidden = hidden * factor[:,None,None]


        base = getattr(model, model.base_model_prefix)
        encoder = base.encoder                         # BERT->bert.encoder, debrta->deberta.encoder

        # Build attention mask locally to avoid NameError
        attn_mask = torch.ones(hidden.size()[:2], dtype=torch.long, device=hidden.device)
        extended = model.get_extended_attention_mask(
            attention_mask=attn_mask,
            input_shape=attn_mask.shape,
            device=hidden.device,
        )
        
        for idx in range(hook_layer, model.config.num_hidden_layers):
            hidden = encoder.layer[idx](hidden, attention_mask=extended, output_attentions=False)[0]
            
        pooler = getattr(model, "pooler", None) or getattr(base, "pooler", None)
        pooled = pooler(hidden) if pooler is not None else hidden[:, 0]

        y_pred = model.classifier(model.dropout(pooled))

        return y_pred

    def forward(self, f_input, model, hook_layer):
        """
        f_input: shape (bs, embedding_dim)
        """

        for p in model.parameters():
            p.requires_grad = False

        rec_layer_2,_,concept_score_n, concept_vector_n = self.reconstruct(f_input)
        
        rec_sentence_embedding = rec_layer_2.unsqueeze(1)
        hidden = rec_sentence_embedding

        # Prediction 
        logits_reconstruction = self.classifier(hidden,f_input,model,hook_layer)
    
        # Regularizer 1 in ConceptSHAP
        L_sparse_1_new = self.concept_sim(concept_score_n) 
        
        # Regularize 2 in ConceptSHAP
        L_sparse_2_new = self.concept_far(concept_vector_n) 

        return logits_reconstruction, L_sparse_1_new, L_sparse_2_new


def save_plot(list_values,title,path):
    
    os.makedirs(path, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(list_values, marker='o', linestyle='-', color='b', label='Values')
    plt.title(title, fontsize=14)
    plt.xlabel('Batch Iterations', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    file_path = os.path.join(path,f'{title}.pdf')
    plt.savefig(file_path, format='pdf', bbox_inches='tight')
    plt.close()


def concept_shap_train(cfg_baseline_method: BaselineMethodConfig, cfg_model : LLMLoadConfig):

    print("\n######################################## BEGIN : Concepts-based explainability method - Training ConceptSHAP  ########################################")
   
    layer_activations = get_layer_activations(cfg_baseline_method)
    layer_activations_without_labels = layer_activations[:,0,:-1]
    labels = layer_activations[:,0,-1].long()

    r_1 = cfg_baseline_method.baseline_method_args['r1']
    r_2 = cfg_baseline_method.baseline_method_args['r2']
    batch_size = cfg_baseline_method.baseline_method_args['batch_size']
    epochs = cfg_baseline_method.baseline_method_args['epochs']
    hidden_dim = cfg_baseline_method.baseline_method_args['hidden_dim']
    thres = cfg_baseline_method.baseline_method_args['thres']
    nb_classes = cfg_baseline_method.baseline_method_args['nb_classes']

    loss_reg_epoch = cfg_baseline_method.baseline_method_args['loss_reg_epoch']

    n_concepts = cfg_baseline_method.baseline_method_args['n_concepts']
    hook_layer = cfg_baseline_method.hook_layer

    tokenizer, decoder = _init_tokenizer(cfg_model)

    if decoder:
        model = get_hook_model(cfg_model,tokenizer)
        vocab = tokenizer.get_vocab()
        unique_labels = np.unique(np.array(labels))
        keys_labels = set(unique_labels)
        labels_tokens_id = { vocab[str(key)] : key for key in keys_labels if str(key) in vocab}
        device = model.cfg.device
    else:
        model = get_model(cfg_model,decoder,nb_classes) 
        labels_tokens_id = None
        device = model.device

    if decoder:
        concept_net = ConceptNet(n_concepts, layer_activations_without_labels.shape[-1], hidden_dim,thres).to(device)
    else:
        concept_net = ConceptNetEncoder(n_concepts, layer_activations_without_labels.shape[-1], hidden_dim,thres).to(device)

    
    optimizer = torch.optim.Adam(concept_net.parameters(), lr=3e-4)
    train_size = layer_activations_without_labels.shape[0]

    n_iter = 0

    sum_losses = []
    pred_losses = []
    l1_losses = []
    l2_losses = []
    
    model.eval()
    concept_net.train()
    
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
          train_embeddings_narrow = permuted_train_embeddings.narrow(0, batch_start, batch_end - batch_start).to(device)
          train_y_true_narrow = permuted_train_y_true.narrow(0, batch_start, batch_end - batch_start).to(device)  

          final_loss, pred_loss, l1, l2 = concept_net.loss(train_embeddings_narrow,
                                                                            train_y_true_narrow,
                                                                            regularize=regularize,
                                                                            labels_tokens_id=labels_tokens_id,
                                                                            r_1=r_1, r_2=r_2,
                                                                            nb_classes=nb_classes,
                                                                            model=model,
                                                                            hook_layer=hook_layer)        

   
        
          # update gradients
          optimizer.zero_grad()
          final_loss.backward()
          optimizer.step()

        
          sum_losses.append(final_loss.data.item())
          pred_losses.append(pred_loss.data.item())
          l1_losses.append(l1.data.item())
          l2_losses.append(l2.data.item())
    
          # update batch indices
          batch_start += batch_size
          batch_end += batch_size
          n_iter += 1


    save_plot(sum_losses,'Total loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(pred_losses,'Prediction loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(l1_losses,'L1 loss',cfg_baseline_method.metrics_reconstruction)
    save_plot(l2_losses,'L2 loss',cfg_baseline_method.metrics_reconstruction)    

    # Create directory and save the ConceptNet model 
    os.makedirs(f"{cfg_baseline_method.path_to_baseline_methods}_{thres:.1f}", exist_ok=True)
    torch.save(concept_net.state_dict(), os.path.join(f'{cfg_baseline_method.path_to_baseline_methods}_{thres:.1f}',f'conceptshap_weights.pth') )
    logger.info(f"Fitted {cfg_baseline_method.method_name} weights are saved in {cfg_baseline_method.path_to_baseline_methods}_{thres:.1f}")
        
    print("\n######################################## END : Concepts-based explainability method - Training ConceptSHAP  ########################################")


def get_layer_activations(cfg_baseline_method: BaselineMethodConfig):

    tensors = []
    # Load the classifier LLM cached activations
    logger.info(f"Loading the activations from {cfg_baseline_method.activations_path}")
    for file_name in sorted(os.listdir(cfg_baseline_method.activations_path)):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(cfg_baseline_method.activations_path, file_name)
            # Load the safetensors file (assuming a single tensor per file)
            tensor_data = load_file(file_path)  # Returns a dictionary
            for key, tensor in tensor_data.items():
                tensors.append(tensor)

    
    layer_activations = torch.cat(tensors, dim=0)  # (N,1,d+1)

    return layer_activations



def ica_fitting(cfg_baseline_method: BaselineMethodConfig):
 
    print("\n######################################## BEGIN : Concepts-based explainability method - Fitting ICA  ########################################")


    layer_activations = get_layer_activations(cfg_baseline_method)
    
    # No need to get the predicted labels for ICA, the method is unsupervised
    if cfg_baseline_method.with_label:
        layer_activations = layer_activations[:,0,:-1].cpu().numpy()
    else:
        layer_activations = layer_activations.squeeze(1).cpu().numpy()    

    
    ica_method = FastICA(**cfg_baseline_method.baseline_method_args)

    # ICA fitting ~ Find candidate concept vectors
    independent_components = ica_method.fit_transform(layer_activations)
    logger.info(f"{independent_components.shape[1]} independent components have been fitted by ICA.")

    # Get the mixing matrix
    mixing_matrix = ica_method.mixing_

    # Create directory and save the ICA model 
    os.makedirs(cfg_baseline_method.path_to_baseline_methods, exist_ok=True)
    np.save(os.path.join(cfg_baseline_method.path_to_baseline_methods,'mixing_matrix.npy'), mixing_matrix) 
    joblib.dump(ica_method,  os.path.join(cfg_baseline_method.path_to_baseline_methods,'ica.pkl'))
    logger.info(f"Fitted {cfg_baseline_method.method_name} is saved in {cfg_baseline_method.path_to_baseline_methods}")

    print("\n######################################## END : Concepts-based explainability method - Fitting ICA  ########################################")



def baseline_concept_method_train(config_baseline_method: str, config_llm_classifier : Optional[str] = None):

    # Retrieve the config of the baseline method
    cfg_baseline_method = BaselineMethodConfig.autoconfig(config_baseline_method)

    seed = next((k for k in ("seed", "random_state") if k in cfg_baseline_method.baseline_method_args), None)
    if seed is not None:
        set_seed( cfg_baseline_method.baseline_method_args[seed])

    if cfg_baseline_method.method_name == 'concept_shap':

        # ConceptSHAP needs to load the LLM classifier since the training requires to compare the original predictions (already cached) with the predictions made by the classifier when the original hidden state goes through the concepts layer bottleneck 
        if config_llm_classifier is None:
            raise ValueError(f"You use the ConceptSHAP baseline which requires to load the original classifier. You should pass a model config with the --config-model option.")
        
        # Retrieve the config of the model
        cfg_model = LLMLoadConfig.autoconfig(config_llm_classifier)

        concept_shap_train(cfg_baseline_method, cfg_model)
    
    elif cfg_baseline_method.method_name == 'hi_concept':
        
        # HI-Concept needs to load the LLM classifier since the method contains a cross-entropy loss between the model original's label predicted and the probability distribution obtained after branching the reconstructed hidden state
        if config_llm_classifier is None:
            raise ValueError(f"You use the HI-Concept baseline which requires to load the original classifier. You should pass a model config with the --config-model option.")
        
        # Retrieve the config of the model
        cfg_model = LLMLoadConfig.autoconfig(config_llm_classifier)

        hi_concept_train(cfg_baseline_method, cfg_model)
    
    elif cfg_baseline_method.method_name == 'ica':

        # ICA does not require to load a LLM classifier config. We assume the activations of the studied layer are already cached.
        ica_fitting(cfg_baseline_method)
    
    else:
        raise ValueError(f"Unknown baseline method: {cfg_baseline_method.method_name}")