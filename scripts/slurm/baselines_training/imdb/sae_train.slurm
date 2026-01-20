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

#### SAE ####

python -m sae_classification train-sae --config=configs/train_baselines/train_sae/pythia_410m/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/pythia_1b/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/bert/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/deberta/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/gpt_j/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/mistral_instruct/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/llama_instruct/train_sae_imdb.cfg
