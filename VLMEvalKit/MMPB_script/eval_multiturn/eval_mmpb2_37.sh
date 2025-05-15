#!/bin/bash

models=(
    # "llava-onevision-qwen2-7b-ov-hf"
    # "Ovis2-8B"
    # "Qwen2-VL-7B-Instruct"
    # "Qwen2.5-VL-7B-Instruct"
    # "Ovis2-16B"
    # "llava_v1.5_13b"
    "InternVL2_5-8B-MPO"
    # "Ovis2-8B"
)

prompt_types=("hard_moderate")
# conversation_turns=(20) 
conversation_turns=(1 5 10 25 50 117) # (0), 1, 5, (10) 25, 50 ,100


for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for turn in "${conversation_turns[@]}"; do
            echo "Running model: $model, prompt type: $prompt_type, conversation turns: $turn"
            CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model "$model" --verbose --category human_recognition --generic_conversation_type pref --generic_conversation_n_turn "$turn" --wandb_exp_name pref_to_rec_results
        done
    done
done