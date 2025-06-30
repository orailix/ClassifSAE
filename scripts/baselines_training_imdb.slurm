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

# Train the 3 baselines (SAE, ICA, ConceptSHAP) on the cached sentence-level hidden-state activations extracted from the residual stream of the penultimate transformer block for each of the 4 LLMs fine-tuned on IMDB. 
# SAE and ICA are unsupervised, they only use the inspected layer activations. ConceptSHAP includes a supervised component like ClassifSAE, leveraging the predictions made by the LLM classifier, therefore we also provides the predicted labels for this method.

# IMDB

#### ICA ####

python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/pythia_70m/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/pythia_160m/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/pythia_410m/fit_ica_imdb.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/pythia_1b/fit_ica_imdb.cfg

#### ConceptSHAP ####

python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/pythia_70m/train_conceptshap_imdb.cfg --config-model=configs/load_LLM_classifier/pythia_70m/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/pythia_160m/train_conceptshap_imdb.cfg --config-model=configs/load_LLM_classifier/pythia_160m/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/pythia_410m/train_conceptshap_imdb.cfg --config-model=configs/load_LLM_classifier/pythia_410m/LLM_classifier_imdb_train.cfg
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/pythia_1b/train_conceptshap_imdb.cfg --config-model=configs/load_LLM_classifier/pythia_1b/LLM_classifier_imdb_train.cfg

#### SAE ####

python -m sae_classification train-sae --config=configs/train_baselines/train_sae/pythia_70m/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/pythia_160m/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/pythia_410m/train_sae_imdb.cfg
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/pythia_1b/train_sae_imdb.cfg



