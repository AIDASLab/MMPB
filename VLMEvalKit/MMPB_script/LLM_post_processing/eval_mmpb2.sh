#!/bin/bash
models=(
    # "Ovis2-34B"
    "InternVL2_5-78B-MPO"
    # "Qwen2.5-VL-72B-Instruct"
    # "Qwen2-VL-72B-Instruct"
)

prompt_types=("hard_moderate")

for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        echo "Running model: $model, prompt type: $prompt_type, total concept: $number"
        CUDA_VISIBLE_DEVICES=4 python run.py --category inconsistency --data MMPB --model "$model" --verbose --llm_post_processing True --blind Yes --wandb_exp_name LLM_post_processing
    done
done