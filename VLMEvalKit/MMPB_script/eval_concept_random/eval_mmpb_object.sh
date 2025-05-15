#!/bin/bash
models=(
    "llava-onevision-qwen2-7b-ov-hf"
    "Ovis2-8B"
    "Qwen2-VL-7B-Instruct"
    "Qwen2.5-VL-7B-Instruct"
    "Ovis2-16B"
    # "llava_v1.5_13b"
    # "InternVL2_5-8B-MPO"
)

prompt_types=("hard_moderate")
concept_numbers=(2 5 7 10 15 25)


for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for number in "${concept_numbers[@]}"; do
            echo "Running model: $model, prompt type: $prompt_type, total concept: $number"
            torchrun --nproc-per-node=4 run.py --data MMPB --category object --model "$model" --verbose --injection_prompt_n_concepts "$number" --wandb_exp_name object_multi_concepts_random_results   
        done
    done
done
