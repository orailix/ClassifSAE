import os
import torch
from safetensors.torch import load_file
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
from loguru import logger
from ..utils import DRMethodsConfig, LLMLoadConfig
import pickle
from .sae_evaluation import compute_loss_last_token_classif, update_metrics, compute_same_predictions, ActivationDataset, cache_activations_with_labels, display_cosine_similarity_stats, cosine_similarity_concepts, knn_distance_metric
from .concept_shap import ConceptNet
from datasets import Dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from ..model_training import process_dataset,get_hook_model
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
import pickle
import time
import json
from functools import partial
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge






def reconstr_hook_classification_token(activation, hook, replacement):
    n,m,d = activation.shape
    new_activations = replacement.view(n,m,d)
    return new_activations



def reconstr_hook_classification_token_single_element(activation, hook, replacement):
    return replacement.unsqueeze(1)



def dr_fit(config_dr_method: str):
 
    #Retrieve the config of the DR method
    cfg_dr_method = DRMthodsConfig.autoconfig(config_dr_method)

    
    tensors = []
    for file_name in os.listdir(cfg_dr_method.activations_path):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(cfg_dr_method.activations_path, file_name)
            # Load the safetensors file (assuming a single tensor per file)
            tensor_data = load_file(file_path)  # Returns a dictionary
            for key, tensor in tensor_data.items():
                tensors.append(tensor)

   
    layer_activations = torch.cat(tensors, dim=0)  # Concatenate along the first axis (n1+n2+...+nN)


    layer_activations = layer_activations.squeeze(1).numpy()
    
    if cfg_dr_method.method_name == 'pca':
         
        dr_method = PCA(**cfg_dr_method.dr_methods_args)
    
        # Fit PCA on the data and transform it
        pca_vectors = dr_method.fit_transform(layer_activations)
    
        # explained_variance_ratio = pca.explained_variance_ratio_


    elif cfg_dr_method.method_name == 'ica':

        dr_method = FastICA(**cfg_dr_method.dr_methods_args)
    
        independent_components = dr_method.fit_transform(layer_activations)
        print(f"independent_components shape : {independent_components.shape}")

        # Get the mixing matrix
        mixing_matrix = dr_method.mixing_
        np.save(f'{cfg_dr_method.path_to_dr_methods}.npy', mixing_matrix) 


 
    joblib.dump(dr_method,  f'{cfg_dr_method.path_to_dr_methods}.pkl')
    logger.info(f"Fitted {cfg_dr_method.method_name} is saved in {cfg_dr_method.path_to_dr_methods}.pkl")




            
def eval_hook_loss(hook_model,
                   dr_model,
                   method_name,
                   activations_dataset,
                   labels_dataset,
                   tokenizer,
                   hook_name,
                   hook_layer,
                   is_eos,
                   path_to_dr_methods):

    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(np.array(labels_dataset))
    keys_labels = set(unique_labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}

    # Evaluation loop
    hook_model.eval()

    activations_dataloader = DataLoader(activations_dataset,batch_size=1)
    dict_metrics_original = {}
    dict_metrics_reconstruction = {}
    total_matches_original = 0
    total_matches_reconstruction = 0
    total_same_predictions = 0
    original_total_loss = 0
    reconstruction_total_loss = 0

    feature_activations_list = []
    original_activation_list = []
    prompt_labels_list = []

    bs = next(iter(activations_dataloader))["input_ids"].shape[0]

    #Save the different activations to speed the causality calculations done after
    new_activations_dataset = ActivationDataset()
     

    with torch.no_grad():

        for batch in tqdm(activations_dataloader, desc="Forward Passes with reconstruction with the DR method", unit="batch"):

                
                inputs = batch["input_ids"].to(hook_model.cfg.device)
                cache = batch["cache"].to(hook_model.cfg.device)
                original_output = batch["output"].to(hook_model.cfg.device)
                labels = batch["label"]
                attention_mask = batch['attention_mask'].to(dtype=int)

                
                a,c = cache.shape
                #print(f"cache shape : {cache.shape}")
                #cache_flatten = cache.view(a*b,-1)
                cache_sentence = cache

                
                if method_name == 'pca':

                    cache_numpy = cache_sentence.cpu().numpy()
                    dr_activations = torch.from_numpy(dr_model.transform(cache_numpy)).to(hook_model.cfg.device) #shape : [batch size, num_components]
                    #print(f"dr_activations shape : {dr_activations.shape}")
                    
                    #For the PCA
                    reconstruct_from_dr = torch.matmul(dr_activations, torch.from_numpy(dr_model.components_).to(hook_model.cfg.device)) + torch.from_numpy(dr_model.mean_).to(hook_model.cfg.device)
                
                elif method_name=='ica':

                    cache_numpy = cache_sentence.cpu().numpy()
                    dr_activations = torch.from_numpy(dr_model.transform(cache_numpy)).to(hook_model.cfg.device) #shape : [batch size, num_components]
                    #print(f"dr_activations shape : {dr_activations.shape}")

                    try : 
                        mixing_matrix = np.load(f'{path_to_dr_methods}.npy')
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Error: '{path_to_dr_methods}.npy' not found. The mixing matrix for ICA has not been saved.")
                    except Exception as e:
                        raise RuntimeError(f"An unexpected error occurred while loading the mixing matrix for ICA: {str(e)}")

                    dr_mean = torch.from_numpy(dr_model.mean_).to(hook_model.cfg.device)
                    mixing_matrix = torch.tensor(mixing_matrix).to(hook_model.cfg.device)
                    
                    reconstruct_from_dr = dr_activations @ mixing_matrix.T + dr_mean

                elif method_name=='concept_shap':

                    reconstruct_from_dr, dr_activations = dr_model.reconstruct(cache_sentence)
                    dr_activations = dr_activations.cpu()
                    #print(f"dr_activations : {dr_activations}")
                    
                else:
                    raise ValueError("Currently only 'pca','ica' and 'concept_shap' are supported as dimensionalinity reduction methods")


                reconstruct_from_dr = reconstruct_from_dr.view(a,c)

                #Save the activations and labels
                bs = inputs.shape[0]
                d_in = cache.shape[-1]
                inputs_to_save = inputs.cpu()
                cache_to_save = cache.cpu()
                original_output_to_save = original_output.cpu()  #shape : [batch size, vocab size]
                attention_mask_to_save = attention_mask.cpu()
                dr_activations_to_save = dr_activations
                new_activations_dataset.append(inputs_to_save, cache_to_save, original_output_to_save, labels,attention_mask_to_save,dr_activations_to_save)

                feature_activations_list.append(dr_activations_to_save)
                original_activation_list.append(cache_sentence.cpu())
                prompt_labels_list.append(labels)

                # logits_reconstruction = hook_model.run_with_hooks(
                #     reconstruct_from_dr,
                #     start_at_layer=hook_layer,
                #     return_type="logits"
                # )

                
                # logits_reconstruction = hook_model.run_with_hooks(eval_hook_loss
                #     cache,
                #     start_at_layer=hook_layer,
                #     fwd_hooks=[
                #         (
                #             hook_name,
                #             partial(reconstr_hook_classification_token, replacement=reconstruct_from_dr),
                #         ) ],
                #     return_type="logits",
                # )

            
                cache_sentence = cache_sentence.unsqueeze(1)
                logits_reconstruction = hook_model.run_with_hooks(
                    cache_sentence,
                    start_at_layer=hook_layer,
                    fwd_hooks=[
                        (
                            hook_name,
                            partial(reconstr_hook_classification_token_single_element, replacement=reconstruct_from_dr),
                        ) ],
                    return_type="logits",
                )

                logits_reconstruction = logits_reconstruction.contiguous().view(-1, logits_reconstruction.shape[-1])

                #We compute the original classification cross-entropy loss and the same loss obtained by plugging the reconstructed activations at hook_name 
                original_loss = compute_loss_last_token_classif(inputs,original_output,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[-1])
                original_total_loss  += original_loss
                reconstruction_loss = compute_loss_last_token_classif(inputs,logits_reconstruction,labels_tokens_id,is_eos=is_eos,vocab_size=logits_reconstruction.shape[-1])
                reconstruction_total_loss += reconstruction_loss

            
                #We compute the variation in true accuracy
                acc_original = update_metrics(inputs,original_output,labels_tokens_id,dict_metrics_original,is_eos)
                acc_reconstruction = update_metrics(inputs,logits_reconstruction,labels_tokens_id,dict_metrics_reconstruction,is_eos)
                total_matches_original += acc_original.item()
                total_matches_reconstruction += acc_reconstruction.item()
                #We compute the recovering accuracy metric
                count_same_predictions = compute_same_predictions(original_output,logits_reconstruction,is_eos)
                total_same_predictions += count_same_predictions.item()


        accuracy_original = total_matches_original / (len(activations_dataset)*bs)
        accuracy_reconstruction = total_matches_reconstruction /  (len(activations_dataset)*bs)
        recovering_accuracy = total_same_predictions /  (len(activations_dataset)*bs)

        reconstruction_mean_loss = reconstruction_total_loss / len(activations_dataloader)
        original_mean_loss = original_total_loss / len(activations_dataloader)

        print(f"\nOriginal classification crossentropy mean loss : {original_mean_loss} ({total_matches_original} matches) - DR reconstruction classification crossentropy mean loss : {reconstruction_mean_loss} (Computed over {len(activations_dataset)} sentences) ")
        print(f'\nRecovering accuracy : {recovering_accuracy}')
        print(f"\nOriginal accuracy of the model : {accuracy_original} - Accuracy of the model when plugging the SAE reconstruction hidden states : {accuracy_reconstruction}")

        feature_activations_tensor = torch.cat(feature_activations_list,dim=0)
        original_activation_tensor = torch.cat(original_activation_list,dim=0)
        prompt_labels_tensor = torch.cat(prompt_labels_list,dim=0)

        return dict_metrics_original, dict_metrics_reconstruction, new_activations_dataset, feature_activations_tensor, original_activation_tensor, prompt_labels_tensor




def run_model_to_get_pred(
    hook_model,
    dr_model,
    method_name,
    path_to_dr_methods,
    activations_dataset,
    labels_dataset,
    tokenizer,
    hook_name,
    hook_layer,
    is_eos,
    perturb=None
):
    dict_metrics_reconstruction = {}
    
   
    #Retrieve a dictionary matching the labels to their tokens ids
    vocab = tokenizer.get_vocab()
    unique_labels = np.unique(np.array(labels_dataset))
    keys_labels = set(unique_labels)
    labels_tokens_id = {str(key) : vocab[str(key)] for key in keys_labels if str(key) in vocab}

    # Evaluation loop
    hook_model.eval()

    activations_dataloader = DataLoader(activations_dataset,batch_size=1)

    bs = next(iter(activations_dataloader))["input_ids"].shape[0]

    #This is where we store the prediction logits specific to classification. One for each category plus one that sums up all the logits associated to tokens which do not belong to the tokens of a class
    classification_predicted_probs = torch.zeros((len(labels_dataset),len(unique_labels)+1)).cpu()

    #For accuracy 
    number_matches = 0

    with torch.no_grad():
      
        for num_batch, batch in tqdm(enumerate(activations_dataloader), desc="Forward Passes with the dr method with ablation on feature(s)", unit="batch"):
                
                inputs = batch["input_ids"].to(hook_model.cfg.device)
                cache = batch["cache"].to(hook_model.cfg.device)
                activations_reconstruct = batch["activations_reconstruct"]
                #print(f"activations_reconstruct shape : {activations_reconstruct.shape}")
                
                a,c = cache.shape
                #activations_reconstruct = activations_reconstruct.view(a*b,-1)
                
            
                if perturb is not None:
                    activations_reconstruct[:,perturb] = 0

                if method_name == 'pca':
                    #For the PCA
                    reconstruct_from_dr = torch.matmul(activations_reconstruct.to(hook_model.cfg.device), torch.from_numpy(dr_model.components_).to(hook_model.cfg.device)) + torch.from_numpy(dr_model.mean_).to(hook_model.cfg.device)
                    #print(f"reconstruct_from_dr shape : {reconstruct_from_dr.shape}")
                    reconstruct_from_dr = reconstruct_from_dr.view(a,b,c)
                    #print(f"reconstruct_from_dr shape : {reconstruct_from_dr.shape}")
                elif method_name == 'ica':
                    try : 
                        mixing_matrix = np.load(f'{path_to_dr_methods}.npy')
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Error: '{path_to_dr_methods}.npy' not found. The mixing matrix for ICA has not been saved.")
                    except Exception as e:
                        raise RuntimeError(f"An unexpected error occurred while loading the mixing matrix for ICA: {str(e)}")

                    dr_mean = torch.from_numpy(dr_model.mean_).to(hook_model.cfg.device)
                    mixing_matrix = torch.tensor(mixing_matrix).to(hook_model.cfg.device)
                    
                    reconstruct_from_dr = activations_reconstruct @ mixing_matrix.T + dr_mean
                    
                elif method_name=='concept_shap':

                    activations_reconstruct = activations_reconstruct.to(hook_model.cfg.device)
                    rec_layer_1 = F.relu(torch.mm(activations_reconstruct, dr_model.rec_vector_1))
                    reconstruct_from_dr = torch.mm(rec_layer_1, dr_model.rec_vector_2) 
                    
            
                else:
                    raise ValueError("Currently only 'pca' and 'ica' are supported as dimensionalinity reduction methods")
                        
            
                # logits_reconstruction = hook_model.run_with_hooks(
                #     cache,
                #     start_at_layer=hook_layer,
                #     fwd_hooks=[
                #         (
                #             hook_name,
                #             partial(reconstr_hook_classification_token, replacement=reconstruct_from_dr),
                #         ) ],
                #     return_type="logits",
                # )

                cache_sentence = cache.unsqueeze(1)
                logits_reconstruction = hook_model.run_with_hooks(
                    cache_sentence,
                    start_at_layer=hook_layer,
                    fwd_hooks=[
                        (
                            hook_name,
                            partial(reconstr_hook_classification_token_single_element, replacement=reconstruct_from_dr),
                        ) ],
                    return_type="logits",
                )

                
                #We adapt the logits so that we extract the logits for the class and sum up all the other logits to a category that we could see as undecised
                predicted_logits = logits_reconstruction.contiguous().view(-1, logits_reconstruction.shape[-1])
                class_probs = torch.zeros((predicted_logits.shape[0],len(unique_labels)+1))
                probs = F.softmax(predicted_logits,dim=1)
                prob_alternative = 1
                for i , (key,value) in enumerate(labels_tokens_id.items()):
                      #Predictions
                      class_probs[:,i] = probs[:,value]
                      prob_alternative -= class_probs[:,i]
                class_probs[:,-1] = prob_alternative
    
                if num_batch==(len(activations_dataloader)-1):
                    classification_predicted_probs[num_batch*inputs.shape[0] : ] = class_probs.cpu()
                else:
                    classification_predicted_probs[num_batch*inputs.shape[0] : (num_batch+1)*inputs.shape[0] ] = class_probs.cpu()

                number_matches += update_metrics(inputs,predicted_logits,labels_tokens_id,dict_metrics_reconstruction,is_eos)

    del cache
    del logits_reconstruction
    del predicted_logits
    del class_probs
    
    accuracy = number_matches /  (len(activations_dataset)*bs)
    print(f"accuracy : {accuracy}")

    return  classification_predicted_probs , accuracy, dict_metrics_reconstruction 
    




def eval_causal_effect_model(probs_pred, 
                             accuracy,
                             hook_model,
                             dr_model,
                             activations_dataset,
                             labels_dataset,
                             tokenizer,
                             hook_name,
                             hook_layer,
                             is_eos,
                             features_to_ablate,
                             method_name,
                             path_to_dr_methods):

    #This is where we save the impact and change of accuracy resulting from the ablation of one feature
    overall_effects = []
    overall_accs_change = []
    overall_dict_metrics = []

    #We re-run the model with the perturbation
    probs_pred_without_selected, accuracy_without_selected, dict_metrics_without_selected = run_model_to_get_pred(hook_model,dr_model,method_name,path_to_dr_methods,activations_dataset,labels_dataset,tokenizer,hook_name,hook_layer,is_eos,perturb=features_to_ablate)

    accuracy_change_relative = (accuracy_without_selected - accuracy) / accuracy
    accuracy_change_absolute = (accuracy_without_selected - accuracy)
    without_selected_tvd = 0.5 * torch.sum(torch.abs(probs_pred - probs_pred_without_selected), dim=1).mean().item()  
    print(f'When we desactivate the top {len(features_to_ablate.tolist())} features by the metric of mean activation, it results in the following effects : \n')
    print(f'TVD : {without_selected_tvd}; Relative Accuracy change: {accuracy_change_relative}; Absolute Accuracy change: {accuracy_change_absolute} \n')

    return dict_metrics_without_selected


def design_figure(components_pca, sizes, np_keys_prototypes, np_prototypes_pca, colors, feature_colors, normalized_class_scores) :

    feature_colors = np.array(['#636EFA','#EF553B','#00CC96','#3D2B1F'])
    #feature_colors = np.array(['#636EFA','#EF553B'])
    # Extracting the x and y components of the vectors
    x = components_pca[:, 0]
    y = components_pca[:, 1]
  
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels_names = np.array(['World','Sport', 'Business','Sci/Tech'])
    #labels_names = np.array(['Non Toxic','Toxic'])
    
    # Plot the special points with yellow triangles
    ax.scatter(np_prototypes_pca[:,0], np_prototypes_pca[:,1], color='orange', marker='^', s=400, zorder=6)
    
    # Add placards with labels for special points
    for i, label in enumerate(labels_names):
      ax.text(np_prototypes_pca[:,0][i] - 4.5, np_prototypes_pca[:,1][i] + 0.1, label, fontsize=15,bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'),zorder=100)
    
    # Add grid, labels, and title
    ax.axhline(0, color='grey', lw=1,zorder=10)
    ax.axvline(0, color='grey', lw=1,zorder=10)
    ax.grid(True, linestyle='--', alpha=0.5,zorder=10)

    def draw_pie(ax, center, sizes, colors, radius=0.5, alpha=0.8):
    
        # Starting angle
        start_angle = 0
        
        # Iterate through sizes and corresponding colors to draw pie slices
        for size, color in zip(sizes, colors):
          # Calculate the end angle of the wedge
          end_angle = start_angle + size * 360
          # Create a wedge patch for each slice
          wedge = Wedge(
              center, radius, start_angle, end_angle,
              facecolor=color, alpha=alpha, zorder=2
          )
          # Add the wedge to the axes
          ax.add_patch(wedge)
          # Update the start angle for the next slice
          start_angle = end_angle

    # print(f"sizes : {sizes}")

    # print(f"components : {components_pca[:,:2]}")
    
    for coord, prop,radius in zip(components_pca[:,:2], normalized_class_scores.T,sizes):
      draw_pie(ax,coord, prop, feature_colors, radius=(5*radius+1e-8))


    components_pca[:,:2]
    
    # Set axis limits to give some padding around the vectors
    range_x = np.concatenate((x,np_prototypes_pca[:,0]))
    range_y = np.concatenate((y,np_prototypes_pca[:,1]))
    
    xlim_min = min(range_x) - 1
    xlim_max = max(range_x) + 1
    ylim_min = min(range_y) - 1
    ylim_max = max(range_y) + 1
    
    
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(ylim_min, ylim_max)
    
    
    ax.set_xlabel('PCA component 1')
    ax.set_ylabel('PCA component 2')
    ax.set_aspect('equal', 'box')
    
    return ax, fig 

    




def pca_activations_projection(feature_activations_tensor, 
                               original_activation_tensor, 
                               prompt_labels_tensor,
                               mean_activations,
                               components):
    logger.info(f"Compute the PCA plan of the original activations and projection the features/decoder rows on it")
    
    labels = prompt_labels_tensor.cpu().numpy()
    original_activations = original_activation_tensor.cpu().numpy()
    sae_activations = feature_activations_tensor.cpu().numpy()
    unique_labels = np.unique(labels) #sorted unique elements of the array

     #Put each vector to norm 1
    norms = np.linalg.norm(original_activations, axis=1, keepdims=True)
    original_activations_norm = original_activations / norms

    #Compute class prototype activations by using test data
    hidden_size_prototype_class = {}
    for label in unique_labels:
        indices = np.where(labels==label)[0]
        if indices.size > 0: #Is at least one sample is associated to this label
            hidden_size_prototype_class[label] = np.mean(original_activations_norm[indices],axis=0)

    #Assign a score to each feature with regard to each class
    feature_score_class = {}
    for label in unique_labels:
        indices = np.where(labels==label)[0]
        if indices.size > 0: #Is at least one sample is associated to this label
            feature_score_class[label] = np.sum(sae_activations[indices],axis=0)

    color_labels = np.arange(unique_labels.size)
    feature_score_class_array = np.zeros((unique_labels.size,sae_activations.shape[1]))
    #Concatenate features scores of each class for normalization
    for i,label in enumerate(unique_labels):
        feature_score_class_array[i,:] = feature_score_class[label]
    feature_sums = feature_score_class_array.sum(axis=0)
    indices_dead_features = (feature_sums==0.)
    normalized_class_scores = np.zeros((unique_labels.size,feature_sums.size))
    normalized_class_scores[:,~indices_dead_features] = feature_score_class_array[:,~indices_dead_features] / feature_sums[~indices_dead_features]
    normalized_class_scores[:,indices_dead_features] = (np.ones(unique_labels.size) / unique_labels.size).reshape(-1,1) 

    feature_colors = px.colors.qualitative.Plotly[:unique_labels.size]
    # Compute colors for all points
    point_colors_indices = np.argmax(normalized_class_scores,axis=0) 
    colors = [feature_colors[idx] for idx in point_colors_indices]

    #On a normalisé les activations avant de faire la PCA dessus car les vecteurs du decoder sont eux-mêmes normalisés
    
    scaler = StandardScaler()
    #print(f"original_activations_norm shape : {original_activations_norm.shape}")
    activations_scaled = scaler.fit_transform(original_activations_norm)    
    pca = PCA(n_components=2)  
    pca.fit(activations_scaled)

    print(f"components shape : {components.shape}")
    
    components_scaled = scaler.transform(components)
    components_pca = pca.transform(components_scaled)

    #Display the hidden representation prototype for each class
    np_prototypes = np.array(list(hidden_size_prototype_class.values()))  # Convert dictionary values to a NumPy array
    np_keys_prototypes = list(hidden_size_prototype_class.keys())  # Get dictionary keys for hover text
    np_prototypes_scaled = scaler.transform(np_prototypes)
    np_prototypes_pca = pca.transform(np_prototypes_scaled)

    sizes = mean_activations.cpu().numpy()

    ax, fig = design_figure(components_pca, sizes, np_keys_prototypes, np_prototypes_pca, colors, feature_colors, normalized_class_scores)  
    fig.savefig('./layer5_30000.pdf', dpi=300, bbox_inches='tight')

    return normalized_class_scores



def analyze_features(
    hook_model,
    feature_activations_tensor,
    original_activation_tensor, 
    prompt_labels_tensor,
    components):

    mean_activations = torch.clamp(feature_activations_tensor, min=0).mean(dim=0).cpu()

    normalized_class_scores = pca_activations_projection(feature_activations_tensor, original_activation_tensor, prompt_labels_tensor, mean_activations ,components)

    #Top p actived features per F_c
    nb_labels = normalized_class_scores.shape[0]
    top_indice = np.argmax(normalized_class_scores,axis=0)
    j_select_list = []
    values_select_list = []
    
    p=20
    
    j_select_tensor = torch.zeros((nb_labels,p),dtype=torch.int)
    values_select_tensor = torch.zeros((nb_labels,p))

    
    for c in range(nb_labels):
        features_most_related_to_c =  torch.from_numpy( np.where(top_indice==c)[0] )
        top_mean_activations_values, top_p_indices  = torch.topk(mean_activations[features_most_related_to_c],k=p)
        #Map the top_p indices back to the original tensor 'mean_activations'
        j_select_c = features_most_related_to_c[top_p_indices]
        j_select_tensor[c,:] = j_select_c
        values_select_tensor[c,:] = top_mean_activations_values


    torch.save(j_select_tensor,'j_select_tensor_dr_method.pt')
    torch.save(values_select_tensor,'values_select_tensor_dr_method.pt')

    return j_select_tensor, mean_activations
    


def dr_methods_investigation(config_dr_method: str, config_model : str):

    #Retrieve the config of the model, dataset and tokenizer
    cfg_model = LLMLoadConfig.autoconfig(config_model)

    #Retrieve the config of the DR method
    cfg_dr_method = DRMthodsConfig.autoconfig(config_dr_method)

    #Load our local tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg_model.tokenizer_path)
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else "</s>"
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"


    #Process the dataset on which we will do the forward passes
    dataset_tokenized = process_dataset(cfg_model,split=cfg_model.split,tokenizer=tokenizer)

    cfg_model.task_args['prompt_tuning'] = False
    hook_model = get_hook_model(cfg_model,tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
    
    #Check if the acivations and labels have already been cached
    dir_activations_with_labels = os.path.join(cfg_dr_method.dir_acts_with_labels,cfg_model.split,cfg_model.model_name,cfg_model.dataset_name)

    cache_activations_file_path = os.path.join(dir_activations_with_labels,f'layer_{cfg_dr_method.hook_layer}.pkl')
    original_text_file_path = os.path.join(dir_activations_with_labels,f'layer_{cfg_dr_method.hook_layer}.json')
    if not os.path.exists(dir_activations_with_labels):
            os.makedirs(dir_activations_with_labels)

    if not os.path.isfile(cache_activations_file_path):
            activations_dataset, original_text_used = cache_activations_with_labels(hook_model,
                                                                                   dataset_tokenized,
                                                                                   data_collator,
                                                                                   tokenizer,
                                                                                   hook_name = cfg_dr_method.hook_name,
                                                                                   is_eos = cfg_model.task_args['is_eos'])
    
            torch.save(activations_dataset.data_block,cache_activations_file_path)

            with open(original_text_file_path, 'w') as f:
                json.dump(original_text_used, f)
    else:
        data_block = torch.load(cache_activations_file_path)
        activations_dataset = ActivationDataset()
        activations_dataset.data_block = data_block
                
        with open(original_text_file_path, 'r') as f:
            original_text_used = json.load(f)
    
    labels_dataset = dataset_tokenized["token_labels"]


    if cfg_dr_method.method_name=='concept_shap':

        
        n_concepts = cfg_dr_method.dr_methods_args['n_concepts']
        hidden_dim = cfg_dr_method.dr_methods_args['hidden_dim']
        thres = cfg_dr_method.dr_methods_args['thres']
        activations_dataloader = DataLoader(activations_dataset,batch_size=1)
        embeddings_sentences = []
        for batch in activations_dataloader:
            cache_sentence = batch["cache"].to(hook_model.cfg.device)
            embeddings_sentences.append(cache_sentence)
        embeddings_sentences_tensor = torch.cat(embeddings_sentences, dim=0)
            
        dr_model = ConceptNet(n_concepts, embeddings_sentences_tensor, hidden_dim, thres).to(hook_model.cfg.device)
        dr_model.load_state_dict(torch.load(f'{cfg_dr_method.path_to_dr_methods}_{thres}.pth'))

        #Display the cosine similarity properties between the learned concepts 
        cosine_similarity_concepts(dr_model.concept,0.9)
    
    else:
        #Load DR method
        dr_model = joblib.load(f'{cfg_dr_method.path_to_dr_methods}.pkl')

    
    
    #Run the reconstruction forward pass
    dict_metrics_original, dict_metrics_reconstruction, new_activations_dataset, feature_activations_tensor, original_activation_tensor, prompt_labels_tensor = eval_hook_loss(hook_model, dr_model, cfg_dr_method.method_name, activations_dataset,labels_dataset, tokenizer , hook_name = cfg_dr_method.hook_name, hook_layer=cfg_dr_method.hook_layer, is_eos=cfg_model.task_args['is_eos'], path_to_dr_methods=cfg_dr_method.path_to_dr_methods)

    with open(os.path.join(cfg_dr_method.metrics_reconstruction,'original.json'), 'w') as file:
            json.dump(dict_metrics_original, file, indent=4)

    with open(os.path.join(cfg_dr_method.metrics_reconstruction,'reconstruction.json'), 'w') as file:
            json.dump(dict_metrics_reconstruction, file, indent=4)


    #Compute the cosine similarity between the features along with their frequency of activations 
    display_cosine_similarity_stats(feature_activations_tensor,torch.arange(0,feature_activations_tensor.shape[1]))

   

    method_name=cfg_dr_method.method_name
    if method_name == 'concept_shap':
        dr_model_components = dr_model.concept.detach().cpu().numpy().T #(n_concepts,embedding_dim)
        print(f"dr_model_components shape : {dr_model_components.shape}")
    elif method_name == 'pca' or  method_name == 'ica':
        dr_model_components = dr_model.components_
        print(f"dr_model_components shape : {dr_model_components.shape}") #(n_concepts,embedding_dim)
    else:
        raise ValueError("Currently only 'pca','ica' and 'concept_shap' are supported as dimensionalinity reduction methods")
    

    #Analyze features
    j_select_tensor, mean_activations = analyze_features(hook_model,feature_activations_tensor, original_activation_tensor, prompt_labels_tensor,dr_model_components)
    print(f"j_select_tensor : {j_select_tensor}")
    cosine_similarity_concepts(torch.from_numpy(dr_model_components.T),0.9)
    mean_knn_distance = knn_distance_metric(torch.from_numpy(dr_model_components),k=5)
    print(f"Mean k-NN Distance across the concepts : {mean_knn_distance}")

    #Causality eval (add an if)
    # total_to_select = [4,20,40,80]
    total_to_select = [1,5,10,15,20]    
    nb_classes = np.unique(np.array(labels_dataset))

     #We first run the model with all the features on, then with one of the feature disable one at a time
    probs_pred, accuracy, dict_metrics_original = run_model_to_get_pred(
        hook_model,
        dr_model,
        cfg_dr_method.method_name,
        cfg_dr_method.path_to_dr_methods,
        new_activations_dataset,
        labels_dataset,
        tokenizer,
        hook_name=cfg_dr_method.hook_name,
        hook_layer=cfg_dr_method.hook_layer,
        is_eos=cfg_model.task_args['is_eos'],
        perturb=None)
    
    for num_features in total_to_select:

            #_, features_indices = torch.topk(mean_activations, num_features)
            features_indices = j_select_tensor[0][:num_features]
        
            dict_metrics_without_selected = eval_causal_effect_model(probs_pred, accuracy, hook_model, dr_model, new_activations_dataset, labels_dataset, tokenizer, hook_name = cfg_dr_method.hook_name, hook_layer=cfg_dr_method.hook_layer, is_eos=cfg_model.task_args['is_eos'], features_to_ablate=features_indices, method_name=cfg_dr_method.method_name, path_to_dr_methods=cfg_dr_method.path_to_dr_methods)

            
            with open(os.path.join(cfg_dr_method.metrics_reconstruction,f'ablation_{num_features}_components.json'), 'w') as file:
                json.dump(dict_metrics_without_selected, file, indent=4)


    


















    
    
    
    