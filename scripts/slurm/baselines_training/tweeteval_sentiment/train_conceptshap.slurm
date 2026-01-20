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


# TWEETEVAL SENTIMENT 

#### ConceptSHAP ####

python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/pythia_410m/train_conceptshap_tweeteval_sentiment.cfg --config-model=configs/load_LLM_classifier/pythia_410m/LLM_classifier_tweeteval_sentiment_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/pythia_1b/train_conceptshap_tweeteval_sentiment.cfg --config-model=configs/load_LLM_classifier/pythia_1b/LLM_classifier_tweeteval_sentiment_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/bert/train_conceptshap_tweeteval_sentiment.cfg --config-model=configs/load_LLM_classifier/bert/LLM_classifier_tweeteval_sentiment_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/deberta/train_conceptshap_tweeteval_sentiment.cfg --config-model=configs/load_LLM_classifier/deberta/LLM_classifier_tweeteval_sentiment_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/gpt_j/train_conceptshap_tweeteval_sentiment.cfg --config-model=configs/load_LLM_classifier/gpt_j/LLM_classifier_tweeteval_sentiment_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/mistral_instruct/train_conceptshap_tweeteval_sentiment.cfg --config-model=configs/load_LLM_classifier/mistral_instruct/LLM_classifier_tweeteval_sentiment_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/llama_instruct/train_conceptshap_tweeteval_sentiment.cfg --config-model=configs/load_LLM_classifier/llama_instruct/LLM_classifier_tweeteval_sentiment_train.cfg
