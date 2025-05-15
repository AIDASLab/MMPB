#!/bin/bash

models=(
    "llava-onevision-qwen2-7b-ov-hf"
    "Ovis2-8B"
    "Qwen2-VL-7B-Instruct"
    "Qwen2.5-VL-7B-Instruct"
    "Ovis2-16B"
    # "InternVL2_5-8B-MPO"
)


prompt=("hard_moderate")
turns=(0)

for model in "${models[@]}"; do
    for loc in "${prompt[@]}"; do
        for turn in "${turns[@]}"; do
            echo "Running model: $model, prompt type: $prompt_type, conversation turns: $turn"
            torchrun --nproc-per-node=3 run.py --data MMPB --model "$model" --verbose --generic_conversation_n_turn "$turn" --injection_description_prompt_type "$loc" --generation_type open-ended --wandb_exp_name gen_test
        done
    done
done
