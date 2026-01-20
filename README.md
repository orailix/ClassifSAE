# Unveiling Decision-Making in LLMs for Text Classification : Extraction of influential and interpretable concepts with Sparse Autoencoders

Mathis Le Bail<sup>1</sup>, Jérémie Dentan<sup>1</sup>, Davide Buscaldi<sup>1,2</sup>, Sonia Vanier<sup>1</sup>

<sup>1</sup>LIX (École Polytechnique, IP Paris, CNRS)  <sup>2</sup>LIPN (Sorbonne Paris Nord)

![](concepts_wordclouds.png)


## Presentation of the repository

### Abstract of the paper

Sparse Autoencoders (SAEs) have been successfully used to probe Large Language Models (LLMs) and extract interpretable concepts from their internal representations. These concepts are linear combinations of neuron activations that correspond to human-interpretable features. In this paper, we investigate the effectiveness of SAE-based explainability approaches for sentence classification, a domain where such methods have not been extensively explored. We present a novel SAE-based architecture tailored for text classification, leveraging a specialized classifier head and incorporating an activation rate sparsity loss. We benchmark this architecture against established methods such as ConceptShap, Independent Component Analysis, and other SAE-based concept extraction techniques. Our evaluation covers two classification benchmarks and four fine-tuned LLMs from the Pythia family. We further enrich our analysis with two novel metrics for measuring the precision of concept-based explanations, using an external sentence encoder. Our empirical results show that our architecture improves both the causality and interpretability of the extracted features.

## License and Copyright

Copyright 2025-present Laboratoire d'Informatique de Polytechnique. Apache Licence v2.0.

### Third-Party Code  

The module `sae_implementation` contains all source code for implementing the activation‐caching process on the training datasets, the SAE architecture and its training procedure. This part of the code is based on [SAELens](https://github.com/jbloomAus/SAELens) (v 3.13.0), which is distributed under the MIT License. You can find the original LICENSE text in `SAELens_License/LICENSE`.
 We modified the original library code to adapt it to our sentence classification variant ClassifSAE.  The SAE architecture is enriched with an additional linear classifier layer (`sae_implementation/sae.py`) and the training (`sae_implementation/sae_trainer.py`) and activations caching (`sae_implementation/activations_store.py` & `sae_implementation/caching_for_sae.py`) methods are adapted to handle only the hidden state associated with the sentence classification decision. Additionally, we incorporate the activation rate sparsity mechanism into the SAE training loss (`sae_implementation/training_sae.py`).

## Overview of the repository

### Concepts extraction from the LLM classifiers 

The folder `sae_classification` contains the source code for the following modules:

- `sae_classification.llm_classifier_tuning` implements fine-tuning of seven LLM backbones on one of the four sentences classification datasets (AG News, IMDB, TweetEval Offensive and TweetEval Sentiment) using the [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) method from HuggingFace Transformers.  

- Module `concepts_extraction` contains the scripts to train the ICA, ConceptSHAP and HI-Concept baselines (`concepts_extraction/baseline_method.py`), train ClassifSAE and the regular SAE (`concepts_extraction/training.py`), evaluate all methods’ concepts on the test splits using the metrics introduced in the paper (`concepts_extraction/concept_model_evaluation.py` & `concepts_extraction/interpret_concepts.py`) and implement the automatic features selection and segmentation strategy (`concepts_extraction/concepts_selection_segmentation.py`).

### Usage

This repository allows users to reproduce all steps from the paper to compare the sentence-level concepts learned by the interpretability approaches. They are trained on internal activations from fine-tuned LLM classifiers.

We considered five concept-based discovery methods: ICA, ConceptSHAP, HI-Concept, SAE and our proposed method ClassifSAE. 

The scripts in `scripts/slurm` launch the full procedure including LLM fine-tuning (or soft-prompt tuning), concept-discovery training and computation of completeness, causality and interpretability metrics for all sets of concepts.
These scripts were designed for deployment on HPC clusters. Our experiments were conducted on the Jean-Zay HPC cluster (IDRIS), using NVIDIA A100 80GB GPUs and Intel Xeon 6248 (40-core) CPUs.

For convenience, we also provide standalone Bash scripts in s`scripts/bash` to reproduce all results for any given <LLM MODEL, DATASET> pair.
The following sections explain how to use them.


#### 0. Environment

Create a conda virtual environnement with the required dependencies:
```bash
conda env create -f environment.yml
conda activate ClassifSAE_env
```

Make the .sh files executable:

```bash
chmod +x scripts/bash/*.sh
```

#### 1. Align a given backbone LLM for the sentence classification task

```bash
bash scripts/bash/llm_classification_alignment.sh <MODEL> <DATASET>
```

For example, to align the Pythia-410M backbone on AG News, run:

```bash
bash scripts/bash/llm_classification_alignment.sh pythia_410m agnews
```

This script tunes the LLM on the training split of the dataset, then evaluate it on both the train and test sets to collect predicted labels. These predictions can subsequently be used by supervised concept-based explainability methods. Finally, the script also caches sentence-level hidden-state activations extracted from the model’s residual stream. These activations (optionally along the predicted labels) serve as the training data for the explainability methods.


In the paper, all backbones are fine-tuned except Mistral-7B-Instruct and LLama-8B-Instruct. These two models are aligned to the classification task via soft-prompt tuning rather than full fine-tuning. The computed prompt embeddings are stored in the folder `prompt_tuned_embeddings` and (when soft-prompt tuning is enabled) are automatically prepended to every input during evaluation for the corresponding model.

For further customization, you may adjust the configuration files used by the src module.
All configs are located in the `config/` directory.

#### 2. Benchmarking Concept-Based Explainability Methods

Once the LLM has been aligned with the classification task, you can evaluate the quality and influence of the learned internal concepts extracted by the different concept-based explainability methods.

The script below trains each concept-discovery method on the specified (tuned-model, dataset) pair and computes all completeness, causality and interpretability metrics introduced in the paper:

```bash
bash scripts/bash/comparison_extracted_concepts.sh <MODEL> <DATASET>
```

For example, run:

```bash
bash scripts/bash/comparison_extracted_concepts.sh pythia_410m agnews
```

#### 3. Results

After running the two scripts above, a summary table of the benchmark metrics will be saved in `concepts_comparison_<MODEL>_<DATASET>` . This directory also includes a plot showing the __SentenceSim__ metric evolution for each method, and qualitative visualizations of the discovered concepts such as word clouds and top-activating sentences.

For more detailed metrics, the complete results for the concept-analysis experiments are stored in  `results/concepts_metrics`. Within this directory, each investigated LLM and interpretability method has its own subfolder `<n_concepts>/test`, which contains a:

- `metrics.json` file that provides general reconstruction-quality metrics, including recovery accuracy, class-wise recall/precision and average/max activation rates of the learned features.
- `causality` folder that includes all metrics assessing the causal influence of the discovered concepts on the model’s predictions.
- `interpretability` folder that includes both 
  - interpretability metrics (Weighted/Unweighted Averaged ConceptSim, SentenceSim, coherence measures),
  - qualitative representations of the concepts (wordclouds based on the TF-IDF scores and top activating sentences).
- a `pca_figures` folder (SAE-based methods only) that contains a 2D PCA projections of the learned concepts into the hidden-state activation space, since SAE directions reside in the same latent space.


Note:
- `sae_logistic_regression` corresponds to the standard SAE pipeline, where the class-specific vector $\mathbf{z}\_{\text{class}}$ is selected via a trained learned logistic probe.
- `sae_truncation` corresponds to our __ClassifSAE__ framework, where $\mathbf{z}_\text{class}$ is taken as the first `n_concepts` dimensions of the SAE hidden layer.

## Citation

If you find our work useful, please consider citing our paper:

```bash
@misc{bail2025unveilingdecisionmakingllmstext,
      title={Unveiling Decision-Making in LLMs for Text Classification : Extraction of influential and interpretable concepts with Sparse Autoencoders}, 
      author={Mathis Le Bail and Jérémie Dentan and Davide Buscaldi and Sonia Vanier},
      year={2025},
      eprint={2506.23951},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.23951}, 
}
```

## Acknowledgements

This work received financial support from Crédit Agricole SA through the research chair “Trustworthy and responsible AI” with École Polytechnique. This work was performed using HPC resources from GENCI-IDRIS 2025-AD011015063R1. 


