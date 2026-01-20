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

# Train ClassifSAE on the cached sentence-level hidden-state activations extracted from the residual stream of the penultimate transformer block of all investigated LLMs classifiers on tweeteval sentiment.
# It also takes as inputs the predicted labels assigned by the LLM classfier for each sentence.


# tweeteval sentiment

python -m sae_classification train-sae --config=configs/train_ClassifSAE/pythia_410m/train_classif_sae_tweeteval_sentiment.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/pythia_1b/train_classif_sae_tweeteval_sentiment.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/bert/train_classif_sae_tweeteval_sentiment.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/deberta/train_classif_sae_tweeteval_sentiment.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/gpt_j/train_classif_sae_tweeteval_sentiment.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/mistral_instruct/train_classif_sae_tweeteval_sentiment.cfg
python -m sae_classification train-sae --config=configs/train_ClassifSAE/llama_instruct/train_classif_sae_tweeteval_sentiment.cfg
