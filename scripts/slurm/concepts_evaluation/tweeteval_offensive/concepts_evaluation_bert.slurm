#!/bin/bash
#SBATCH --account=dun@v100
#SBATCH -C v100
#SBATCH --job-name=sim_eval
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # One task per array job
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1            # Request 1 GPU per task
#SBATCH --time=2:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH --hint=nomultithread
#SBATCH --output=./logs/output/slurm-%A_%a.out.log
#SBATCH --error=./logs/output/slurm-%A_%a.out.log

# Ensure the script stops on errors
set -e

# Initialize conda functions in this shell
eval "$(conda shell.bash hook)"

# Load Conda environment
conda activate ClassifSAE_env

# Evaluation of the learned concepts' "quality". Compute the metrics of completeness, causality and explainability introduced in the paper for the concepts learned by each method by leveraging the test split of tweeteval offensive.
# The evaluation is duplicated for each pair (LLM classifier, concepts discovery method)

# 1. Selection of z_class and partitionning of the features into class-specific features segments.
# 2. Compute the recovery accuracy and the individual conditional causality metrics of the learned concepts
# 3. Compute the metrics of explainability ConceptSim for each concept and SentenceSim. Save the top activating sentences for each concept.


# tweeteval offensive

# #### ICA ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ICA/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ICA/bert/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ICA/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ICA/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ICA/deberta/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ICA/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg

# #### ConceptSHAP ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/bert/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/deberta/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg

# #### HIConcept ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/HIConcept/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/HIConcept/bert/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/HIConcept/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/HIConcept/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/HIConcept/deberta/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/HIConcept/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg


# #### SAE ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/SAE/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/SAE/bert/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/SAE/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/SAE/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/SAE/deberta/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/SAE/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg


#### ClassifSAE ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/bert/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/bert/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_offensive_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/deberta/eval_concepts_tweeteval_offensive.cfg  --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/deberta/eval_concepts_tweeteval_offensive.cfg --config-classifier=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_offensive_test.cfg

