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

# Fine-tune the 4 autoregressive Pythia backbones on the sentences classification dataset AG News. This resluts in 4 fine-tuned models.
# After each training, the fine-tuned classifier is evaluated on the train and test split of the associated dataset. 
# Evaluating the fine-tuned classifier on the train split is important for the rest of the pipeline as this enables us to retrieve the labels predicted by the LLM for each training sentence. 
# The predicted labels are used in addition of the cached activations to train ClassifSAE, adding a supervised component in the learning process. 


# AG NEWS
python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/pythia_70m/fine_tune_agnews.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_70m/evaluation_agnews_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_70m/evaluation_agnews_test.cfg

python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/pythia_160m/fine_tune_agnews.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_160m/evaluation_agnews_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_160m/evaluation_agnews_test.cfg

python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/pythia_410m/fine_tune_agnews.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_410m/evaluation_agnews_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_410m/evaluation_agnews_test.cfg

python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/pythia_1b/fine_tune_agnews.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_1b/evaluation_agnews_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_1b/evaluation_agnews_test.cfg

