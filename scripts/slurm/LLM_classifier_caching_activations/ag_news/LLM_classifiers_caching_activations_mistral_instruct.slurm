#!/bin/bash
#SBATCH --account=dun@a100
#SBATCH -C a100
#SBATCH --job-name=sim_eval
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # One task per array job
#SBATCH --cpus-per-task=8   
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
# of the penultimate transformer block of the pre-trained model Mistral-7B instruct
# The inference is done on the training split of AG News 

# AG NEWS
python -m sae_classification save-activations-classifier --config=configs/cache_LLM_classifier_activations/mistral_instruct/cache_activations_classifier_agnews.cfg

