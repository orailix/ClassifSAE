#!/bin/bash
#SBATCH --account=dun@a100
#SBATCH -C a100
#SBATCH --job-name=sim_eval
#SBATCH --nodes=1               # Request 1 node
#SBATCH --ntasks=1              # One task per array job
#SBATCH --cpus-per-task=8   # Assign 8 CPU cores per task
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

# Train ClassifSAE on the cached sentence-level hidden-state activations extracted from the residual stream of the penultimate transformer block for each of the 4 LLMs fine-tuned on AG News.
# It also takes as inputs the predicted labels assigned by the LLM classfier for each sentence.


# AG NEWS

python -m sae_classification train-sae --config=configs/train_ClassifSAE/pythia_70m/train_classif_sae_agnews.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/pythia_160m/train_classif_sae_agnews.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/pythia_410m/train_classif_sae_agnews.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/pythia_1b/train_classif_sae_agnews.cfg