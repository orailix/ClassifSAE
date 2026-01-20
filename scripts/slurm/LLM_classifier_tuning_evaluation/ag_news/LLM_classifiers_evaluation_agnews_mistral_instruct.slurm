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



# Evaluate directly Mistral-instruct v0.1 on the classification task. To optimize the prompt task, we directly do soft-prompt tuning at the embedding level
# Then, evaluate the classifier on the train and test split of AG News.
# The predicted labels are used in addition of the cached activations to train ClassifSAE, adding a supervised component in the learning process. 


# AG News
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/mistral_instruct/evaluation_agnews_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/mistral_instruct/evaluation_agnews_test.cfg


