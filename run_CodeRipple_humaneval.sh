#!/bin/bash

models=("gpt-4-turbo-preview" "gpt-3.5-turbo" "claude-3-opus" "claude-3-sonnet" "gemini-1.0-pro")
task=("Code")
dataset_type="human_eval"
wavelet_type="db6"
method="zero_sampen_K_biscope_codellamaSwtCodellamaK"

for task in "${task[@]}"; do
for model in "${models[@]}"; do
echo "Running task=$task with model=$model"
python CodeRipple_humaneval.py \
    --dataset_type "$dataset_type" \
    --task "$task" \
    --wavelet_type "$wavelet_type" \
    --generative_model "$model" >>CodeRipplehumaneval.log
done
done