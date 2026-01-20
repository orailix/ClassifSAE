#!/bin/bash
#SBATCH --account=dun@v100
#SBATCH -C v100
#SBATCH --job-name=sim_eval
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # One task per array job
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1            # Request 1 GPU per task
#SBATCH --time=12:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=./logs/output/slurm-%A_%a.out.log
#SBATCH --error=./logs/output/slurm-%A_%a.out.log

# Ensure the script stops on errors
set -e

# Initialize conda functions in this shell
eval "$(conda shell.bash hook)"

# Load Conda environment
conda activate ClassifSAE_env


# IMDB

#### HIConcept ####

python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/pythia_410m/train_hiconcept_imdb.cfg --config-model=configs/load_LLM_classifier/pythia_410m/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/pythia_1b/train_hiconcept_imdb.cfg --config-model=configs/load_LLM_classifier/pythia_1b/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/bert/train_hiconcept_imdb.cfg --config-model=configs/load_LLM_classifier/bert/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/deberta/train_hiconcept_imdb.cfg --config-model=configs/load_LLM_classifier/deberta/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/gpt_j/train_hiconcept_imdb.cfg --config-model=configs/load_LLM_classifier/gpt_j/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/llama_instruct/train_hiconcept_imdb.cfg --config-model=configs/load_LLM_classifier/llama_instruct/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/mistral_instruct/train_hiconcept_imdb.cfg --config-model=configs/load_LLM_classifier/mistral_instruct/LLM_classifier_imdb_train.cfg
