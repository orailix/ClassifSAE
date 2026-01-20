# Evaluation metrics + interpretations for the paper
# Usage:
#   bash scripts/comparison_extracted_concepts.sh pythia_410m agnews

set -euo pipefail

########################
#  Read LLM classifier Model and dataset
########################

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 MODEL DATASET"
    echo "Example: $0 pythia_410m agnews"
    exit 1
fi

MODEL="$1"
DATASET="$2"
CONDA_ENV=${CONDA_ENV:-"ClassifSAE_env"}

# CONFIGURATION USED IN THE PAPER (Modify if used different parameters in the config files)
declare -A HOOK_LAYER
HOOK_LAYER["pythia_410m"]=23
HOOK_LAYER["pythia_1b"]=15
HOOK_LAYER["bert"]=12
HOOK_LAYER["deberta"]=24
HOOK_LAYER["gpt_j"]=27
HOOK_LAYER["mistral_instruct"]=31
HOOK_LAYER["llama_instruct"]=31

declare -A D_SAE
D_SAE["pythia_410m"]=2048
D_SAE["pythia_1b"]=4096
D_SAE["bert"]=1536
D_SAE["deberta"]=2048
D_SAE["gpt_j"]=8192
D_SAE["mistral_instruct"]=8192
D_SAE["llama_instruct"]=8192


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


# Assume the LLM backbone have already been either fine-tuned or soft-prompting tuning to align the inspected model with the classification task


#################### TRAIN THE CONCEPTS BASED EXPLAINABILITY METHODS #####################

####### FIT ICA #######
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/fit_ica/${MODEL}/fit_ica_${DATASET}.cfg

####### TRAIN CONCEPTSHAP #######
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_conceptshap/${MODEL}/train_conceptshap_${DATASET}.cfg --config-model=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_train.cfg

####### TRAIN HICONCEPT #######
python -m sae_classification train-baseline --config-baseline=configs/train_baselines/train_hiconcept/${MODEL}/train_hiconcept_${DATASET}.cfg --config-model=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_train.cfg

####### TRAIN SAE #######
python -m sae_classification train-sae --config=configs/train_baselines/train_sae/${MODEL}/train_sae_${DATASET}.cfg

####### TRAIN ClassifSAE #######
python -m sae_classification train-sae --config=configs/train_ClassifSAE/${MODEL}/train_classif_sae_${DATASET}.cfg


##################### EVALUATE THE CONCEPTS BASED EXPLAINABILITY METHODS #####################

####### EVALUATE CONCEPTS EXTRACTED BY ICA #######
python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ICA/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ICA/${MODEL}/eval_concepts_${DATASET}.cfg  --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ICA/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg

####### EVALUATE CONCEPTS EXTRACTED BY CONCEPTSHAP #######
python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/${MODEL}/eval_concepts_${DATASET}.cfg  --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ConceptSHAP/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg

####### EVALUATE CONCEPTS EXTRACTED BY HICONCEPT #######
python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/HIConcept/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/HIConcept/${MODEL}/eval_concepts_${DATASET}.cfg  --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/HIConcept/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg

####### EVALUATE CONCEPTS EXTRACTED BY SAE #######
python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/SAE/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/SAE/${MODEL}/eval_concepts_${DATASET}.cfg  --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/SAE/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg

####### EVALUATE CONCEPTS EXTRACTED BY ClassifSAE #######
python -m sae_classification post-process-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification evaluate-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/${MODEL}/eval_concepts_${DATASET}.cfg  --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg
python -m sae_classification interpret-concepts --config-concept=configs/evaluate_concepts/ClassifSAE/${MODEL}/eval_concepts_${DATASET}.cfg --config-classifier=configs/load_LLM_classifier/${MODEL}/LLM_classifier_${DATASET}_test.cfg


##################### RETRIEVE THE MAIN CONCEPTS METRICS FOR DISPLAY IN EXPERIMENTS FOLDER #####################
python compile_results.py --model="${MODEL}" --dataset="${DATASET}" --hook-layer="${HOOK_LAYER[$MODEL]}" --d-sae="${D_SAE[$MODEL]}" 