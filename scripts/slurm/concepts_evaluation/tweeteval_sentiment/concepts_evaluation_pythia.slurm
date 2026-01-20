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

# Evaluation of the learned concepts' "quality". Compute the metrics of completeness, causality and explainability introduced in the paper for the concepts learned by each method by leveraging the test split of tweeteval sentiment.
# The evaluation is duplicated for each pair (LLM classifier, concepts discovery method)

# 1. Selection of z_class and partitionning of the features into class-specific features segments.
# 2. Compute the recovery accuracy and the individual conditional causality metrics of the learned concepts
# 3. Compute the metrics of explainability ConceptSim for each concept and SentenceSim. Save the top activating sentences for each concept.


# tweeteval sentiment

#### ICA ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ICA/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ICA/pythia_410m/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ICA/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ICA/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ICA/pythia_1b/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ICA/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg

#### ConceptSHAP ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/pythia_410m/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/pythia_1b/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg


#### HIConcept ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/HIConcept/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/HIConcept/pythia_410m/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/HIConcept/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/HIConcept/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/HIConcept/pythia_1b/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/HIConcept/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg


#### SAE ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/SAE/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/SAE/pythia_410m/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/SAE/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/SAE/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/SAE/pythia_1b/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/SAE/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg


#### ClassifSAE ####

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/pythia_410m/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/pythia_410m/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_test.cfg

python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/pythia_1b/eval_concepts_tweeteval_sentiment.cfg  --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/pythia_1b/eval_concepts_tweeteval_sentiment.cfg --config-classifier=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_test.cfg

