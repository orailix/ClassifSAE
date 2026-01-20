# Either fine-tune or learn prompt-embedding to align the evaluted LLM backbone with the classification dataset
# Usage:
#   bash scripts/llm_classification_alignment.sh pythia_410m agnews


set -euo pipefail

########################
#  Read LLM classifier Model and dataset
########################

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL DATASET"
    echo "Example: $0 pythia_410m agnews "
    exit 1
fi

MODEL="$1"
DATASET="$2"
CONDA_ENV=${CONDA_ENV:-"ClassifSAE_env"}

# CONFIGURATION USED IN THE PAPER (Modify if used different parameters in the config files)
FINE_TUNE_MODELS=("bert" "deberta" "gpt_j" "pythia_410m" "pythia_1b")
SOFT_TUNE_MODELS=("mistral_instruct" "llama_instruct")


########################
# Environment setup
########################


# Initialize conda (only if conda is available)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
else
    echo "[WARN] conda not found on PATH. Assuming dependencies are already installed."
fi

mkdir -p logs/output

if [[ " ${FINE_TUNE_MODELS[*]} " == *" $MODEL "* ]]; then
    echo "Fine-tuning LLM backbone: $MODEL on dataset: $DATASET"
    python -m sae_classification tune-llm-classifier --config="configs/fine_tune_LLM_classifier/${MODEL}/fine_tune_${DATASET}.cfg"
    echo "Evaluating fine-tuned LLM backbone: $MODEL on dataset: $DATASET for train and test splits + collect LLM predicted labels"
    python -m sae_classification eval-classifier --config="configs/evaluate_LLM_classifier/${MODEL}/evaluation_${DATASET}_train.cfg"
    python -m sae_classification eval-classifier --config="configs/evaluate_LLM_classifier/${MODEL}/evaluation_${DATASET}_test.cfg"
elif [[ " ${SOFT_TUNE_MODELS[*]} " == *" $MODEL "* ]]; then
    echo "Evaluating LLM backbone: $MODEL on dataset: $DATASET for train and test splits + collect LLM predicted labels"
    python -m sae_classification eval-classifier --config="configs/evaluate_LLM_classifier/${MODEL}/evaluation_${DATASET}_train.cfg"
    python -m sae_classification eval-classifier --config="configs/evaluate_LLM_classifier/${MODEL}/evaluation_${DATASET}_test.cfg"
else
    echo "Model $MODEL not recognized for fine-tuning or soft-prompt tuning."
    exit 1
fi

# Evaluating the fine-tuned classifier on the train split is important for the rest of the pipeline as this enables us to retrieve the labels predicted by the LLM for each training sentence. 
# The predicted labels are used in addition of the cached activations to train ClassifSAE, adding a supervised component in the learning process. 

echo "LLM classification alignment for model: $MODEL on dataset: $DATASET completed."

# Cache the sentence-level hidden-state activations extracted from the residual stream of the inspected transformer block of the investigated LLM classifier model on the provided dataset.
# The inference is done on the training split of the dataset.

python -m sae_classification save-activations-classifier --config=configs/cache_LLM_classifier_activations/${MODEL}/cache_activations_classifier_${DATASET}.cfg
echo "Cached activations for model: $MODEL on dataset: $DATASET completed."


# Deactivate conda environment
if command -v conda &> /dev/null; then
    conda deactivate
fi

