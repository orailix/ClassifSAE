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

# Fine-tune the encoder-only backbones on the sentences classification dataset TweetEval sentiment. This resluts is 2 fine-tuned models.
# After each training, the fine-tuned classifier is evaluated on the train and test split of the associated dataset. 
# Evaluating the fine-tuned classifier on the train split is important for the rest of the pipeline as this enables us to retrieve the labels predicted by the LLM for each training sentence. 
# The predicted labels are used in addition of the cached activations to train ClassifSAE, adding a supervised component in the learning process. 


# Tweeteval sentiment
python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/bert/fine_tune_tweeteval_sentiment.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/bert/evaluation_tweeteval_sentiment_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/bert/evaluation_tweeteval_sentiment_test.cfg

python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/deberta/fine_tune_tweeteval_sentiment.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/deberta/evaluation_tweeteval_sentiment_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/deberta/evaluation_tweeteval_sentiment_test.cfg

python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/pythia_410m/fine_tune_tweeteval_sentiment.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_410m/evaluation_tweeteval_sentiment_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_410m/evaluation_tweeteval_sentiment_test.cfg

python -m sae_classification tune-llm-classifier --config=configs/fine_tune_LLM_classifier/pythia_1b/fine_tune_tweeteval_sentiment.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_1b/evaluation_tweeteval_sentiment_train.cfg
python -m sae_classification eval-classifier --config=configs/evaluate_LLM_classifier/pythia_1b/evaluation_tweeteval_sentiment_test.cfg

