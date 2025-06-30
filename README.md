### Unveiling Decision-Making in LLMs for Text Classification : Extraction of influential and interpretable concepts with Sparse Autoencoders

Mathis Le Bail<sup>1</sup>, Jérémie Dentan<sup>1</sup>, Davide Buscaldi<sup>1,2</sup>, Sonia Vanier<sup>1</sup>

<sup>1</sup>LIX (École Polytechnique, IP Paris, CNRS)  <sup>2</sup>LIPN (Sorbonne Paris Nord)

# Presentation of the repository

## Abstract of the paper

Sparse Autoencoders (SAEs) have been successfully used to probe Large Language Models (LLMs) and extract interpretable concepts from their internal representations. These concepts are linear combinations of neuron activations that correspond to human-interpretable features. In this paper, we investigate the effectiveness of SAE-based explainability approaches for sentence classification, a domain where such methods have not been extensively explored. We present a novel SAE-based architecture tailored for text classification, leveraging a specialized classifier head and incorporating an activation rate sparsity loss. We benchmark this architecture against established methods such as ConceptShap, Independent Component Analysis, and other SAE-based concept extraction techniques. Our evaluation covers two classification benchmarks and four fine-tuned LLMs from the Pythia family. We further enrich our analysis with two novel metrics for measuring the precision of concept-based explanations, using an external sentence encoder. Our empirical results show that our architecture improves both the causality and interpretability of the extracted features.

## License and Copyright

Copyright 2025-present Laboratoire d'Informatique de Polytechnique. Apache Licence v2.0. (MIT License ?)

Please cite this work as follows:

# Overview of the repository

## Third-Party Code  

The module `sae_implementation` contains all source code for implementing the activation‐caching process on the training datasets, the SAE architecture and its training procedure. This part of the code is based on [SAELens](https://github.com/jbloomAus/SAELens) (v 3.13.0), which is distributed under the MIT License. You can find the original LICENSE text in `SAELens_License/LICENSE`.
 We modified the original library code to adapt it to our sentence classification variant ClassifSAE.  The SAE architecture is enriched with an additional linear classifier layer (`sae_implementation/sae.py`) and the training (`sae_implementation/sae_trainer.py`) and activations caching (`sae_implementation/activations_store.py` & `sae_implementation/caching_for_sae.py`) methods are adapted to handle only the hidden state associated with the sentence classification decision. Additionally, we incorporate the activation rate sparsity mechanism into the SAE training loss (`sae_implementation/training_sae.py`).

## Concepts extraction from the LLM classifiers 

The folder `sae_classification` contains the source code for the following modules:

- `sae_classification.llm_classifier_tuning` implements fine-tuning of the [Pythia](https://github.com/EleutherAI/pythia) backbones on the two sentences classification datasets (AG News and IMDB) using the [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) method from HuggingFace Transformers.  

- Module `concepts_extraction` contains the scripts to train the ICA and ConceptSHAP baselines (`concepts_extraction/baseline_method.py`), train ClassifSAE and the regular SAE (`concepts_extraction/training.py`), evaluate all methods’ concepts on the test splits using the metrics introduced in the paper (`concepts_extraction/concept_model_evaluation.py` & `concepts_extraction/interpret_concepts.py`) and implement the automatic features segmentation strategy (`concepts_extraction/concepts_selection_segmentation.py`).

<!-- The `local_datasets` folder contains the two sentence classification datasets we used to evaluate our concepts discovery methods, stored locally (AG News and IMDB).  -->

## Usage

Our repo enables the user to reproduce the steps undertaken in the paper to compare the sentence-level concepts learned in the two benchmarks when inspecting the internals of the fine-tuned Pythia models. 

The 4 considered concepts discovery methods are: ICA, ConceptSHAP, SAE and ClassifSAE. 

Scripts in the folder `scripts` launch the procedures to fine-tune the LLM classifiers, train the 4 concepts discovery methods and compute the completeness, causality and interpretability metrics for the concepts learned by each method. 
The Slurm scripts were designed for deployment on an HPC cluster. We used Jean-Zay HPC cluster from IDRIS. We used Nvidia A100 80G GPUs and Intel Xeon 6248 CPUs with 40 cores. 
Below, we detail the scripts used to run the full procedure for the AG News classification task. The process for IMDB is identical, requiring only the replacement of `agnews` with `imdb` in the provided filenames.

### 0. Environment

Create a conda virtual environnement with the required dependencies:
```bash
conda env create -f environment.yml
conda activate ClassifSAE_env
```

### 1. Fine-tune the Pythia backbones for sentence classification task

`LLM_classifiers_fine_tuning_evaluation_agnews`: Fine-tune the Pythia backbones on the AG News classification task thanks to a template. After training each model, it evaluates its accuracy separately on the training and test splits. It also saves the labels predicted by the LLM to provide them as training inputs for the concepts discovery methods that contain a supervised component.  

### 2. Cache the LLMs internal activations

`LLM_classifiers_caching_activations_agnews`: Cache the sentence-level hidden-state activations extracted from the residual stream of the penultimate transformer block of each LLM fine-tuned on AG News. 

### 3. Train the 4 concepts discovery methods

`ClassifSAE_training_agnews`: Train our method ClassifSAE.
`baselines_training_agnews`: Train the comparison baselines: ICA, ConceptSHAP and the SAE.

For each fine-tuned LLM, the methods are trained on the activations from the penultimate layer. We consider only one hidden state per sentence, sentence-level information is aggregated at the token just before the classification decision. ClassifSAE and ConceptSHAP are the only two methods that contain a supervised component with regard to the predicted labels of the inspected LLM.

### 4. Evaluation of the learned concepts' "quality"

`concepts_evaluation_agnews`: Compute the metrics of completeness, causality and explainability introduced in the paper for the concepts learned by each method by leveraging the test split of AG News. The evaluation is duplicated 4*4=16 times, one for each pair (LLM fine-tuned classifier, concepts discovery method)

1. Selection of $\mathbf{z}_\text{class}$ and partitionning of the features into class-specific features segments.
2. Compute the recovery accuracy and the individual conditional causality metrics for the learned concepts.
3. Compute the interpretability metrics `ConceptSim` and `SentenceSim`. Save the top activating sentences for each concept.

# Results

Results of the concepts analysis are saved in folder `results\concepts_metrics`. For each investigated LLM and interpretability method, details can be found in subfolder `<n_concepts>/test`, which contains:

- `metrics.json` file that provides general metrics on reconstruction quality, such as recovery accuracy, class-wise recall and precision, and the average and maximum activation rates of the learned features when evaluated on the test split of the examined dataset.
- `causality` folder that includes all metrics related to the concepts impact on the model's decision. 
- `interpretability` folder that includes both interpretability metrics (Weighted and Unweighted Averaged ConceptSim, SentenceSim, coherence measures) and the top activating sentences for each concept (concepts are separated in class related files depending on the results of the featrues segmentation scheme).
- For SAE-based methods, since the directions captured inhabit the same latent space as the hidden state activations, we include the 2D PCA projection of the learned concepts into this space in folder `pca_figures`.

Remark: `sae_logistic_regression` corresponds to the regular SAE approach (selection of $\mathbf{z}\_{\text{class}}$ using a learned logistic probe) and `sae_truncation` to our ClassifSAE framework ( $\mathbf{z}_\text{class}$ is selected as the first `n_concepts` dimensions of the SAE hidden layer).

# References

# Acknowledgements

This work received financial support from Crédit Agricole SA through the research chair “Trustworthy and responsible AI” with École Polytechnique. This work was performed using HPC resources from GENCI-IDRIS 2025-AD011015063R1. 
