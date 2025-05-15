#!/bin/bash

models=(
    # "llava-onevision-qwen2-7b-ov-hf"
    # "Ovis2-8B"
    # "Qwen2-VL-7B-Instruct"
    # "Qwen2.5-VL-7B-Instruct"
    # "Ovis2-16B"
    # "llava_v1.5_13b"
    "InternVL2_5-8B-MPO"
)

methods=(
        "remind_zero_shot"
        # "zero_shot_cot"
        # "few_shot"
        # "few_shot_cot"
        )

prompt=("hard_moderate")
turns=(0 1 5 6 8 20 25)

for model in "${models[@]}"; do
    for loc in "${prompt[@]}"; do
        for turn in "${turns[@]}"; do
            for method in "${methods[@]}"; do
                echo "Running model: $model, prompt type: $prompt_type, conversation turns: $turn"
                torchrun --nproc-per-node=4 run.py --data MMPB --model "$model" --prompting_methods "$method" --verbose --generic_conversation_n_turn "$turn" --injection_description_prompt_type "$loc" --wandb_exp_name remind_rev_results
            done
        done
    done
done
