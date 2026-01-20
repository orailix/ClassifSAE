#!/bin/bash
#SBATCH --account=dun@a100
#SBATCH -C a100
#SBATCH --job-name=sim_eval
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # One task per array job
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1            # Request 1 GPU per task
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --hint=nomultithread
#SBATCH --output=./logs/output/slurm-%A_%a.out.log
#SBATCH --error=./logs/output/slurm-%A_%a.out.log

# Ensure the script stops on errors
set -e

# Initialize conda functions in this shell
eval "$(conda shell.bash hook)"

# Load Conda environment
conda activate ClassifSAE_env

# Evaluation of the learned concepts' "quality". Compute the metrics of completeness, causality and explainability introduced in the paper for the concepts learned by each method by leveraging the test split of AG News.
# The evaluation is duplicated for each pair (LLM classifier, concepts discovery method)

# 1. Selection of z_class and partitionning of the features into class-specific features segments.
# 2. Compute the recovery accuracy and the individual conditional causality metrics of the learned concepts
# 3. Compute the metrics of explainability ConceptSim for each concept and SentenceSim. Save the top activating sentences for each concept.


# AG NEWS

#### ICA ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ICA/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ICA/gpt_j/eval_concepts_agnews.cfg  --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ICA/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg

#### ConceptSHAP ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/gpt_j/eval_concepts_agnews.cfg  --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg

#### HIConcept ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/HIConcept/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/HIConcept/gpt_j/eval_concepts_agnews.cfg  --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/HIConcept/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg


#### SAE ####


python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/SAE/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/SAE/gpt_j/eval_concepts_agnews.cfg  --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/SAE/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg


#### ClassifSAE ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/gpt_j/eval_concepts_agnews.cfg  --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/gpt_j/eval_concepts_agnews.cfg --config-classifier=configs/load_LLM_classifier/gpt_j/LLM_classifier_agnews_test.cfg

