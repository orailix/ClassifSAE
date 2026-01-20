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

# Cache the sentence-level hidden-state activations extracted from the residual stream 
# of the penultimate transformer block of the BERT models fine-tuned on IMDB.
# The inference is done on the training split of IMDB 

# IMDB
python -m sae_classification save-activations-classifier --config=configs/cache_LLM_classifier_activations/bert/cache_activations_classifier_imdb.cfg
python -m sae_classification save-activations-classifier --config=configs/cache_LLM_classifier_activations/deberta/cache_activations_classifier_imdb.cfg


