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

# Fine-tune the GPT-J backbone on the sentences classification dataset TweetEval Offensive. This resluts in 1 fine-tuned model.
# After each training, the fine-tuned classifier is evaluated on the train and test split of the associated dataset. 
# Evaluating the fine-tuned classifier on the train split is important for the rest of the pipeline as this enables us to retrieve the labels predicted by the LLM for each training sentence. 
# The predicted labels are used in addition of the cached activations to train ClassifSAE, adding a supervised component in the learning process. 


# Tweeteval Offensive
python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/gpt_j/fine_tune_tweeteval_offensive.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/gpt_j/evaluation_tweeteval_offensive_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/gpt_j/evaluation_tweeteval_offensive_test.cfg

