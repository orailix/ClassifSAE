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

#### ICA ####

python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/pythia_410m/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/pythia_1b/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/bert/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/deberta/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/gpt_j/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/mistral_instruct/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/llama_instruct/fit_ica_imdb.cfg