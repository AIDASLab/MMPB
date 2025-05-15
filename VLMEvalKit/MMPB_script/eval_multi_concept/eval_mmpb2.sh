#!/bin/bash

# models=(
#     # "llava-onevision-qwen2-7b-ov-hf"
#     # "Ovis2-8B"
#     # "Qwen2-VL-7B-Instruct"
#     # "Qwen2.5-VL-7B-Instruct"
#     # "Ovis2-16B"
#     "llava_v1.5_13b"
#     # "InternVL2_5-8B-MPO"
#     # "Ovis2-8B"
# )

# prompt_types=("hard_moderate")
# conversation_turns=(10) 
# concept_numbers=(10) # (0), 1, 5, (10) 25, 50 ,100


# for model in "${models[@]}"; do
#     for prompt_type in "${prompt_types[@]}"; do
#         for turn in "${conversation_turns[@]}"; do
#             for number in "${concept_numbers[@]}"; do
#                 echo "Running model: $model, prompt type: $prompt_type, conversation turns: $turn, total concept: $number"
#                 torchrun --nproc-per-node=4 run.py --data MMPB --model "$model" --verbose --injection_prompt_n_concepts "$number" --generic_conversation_type text --generic_conversation_n_turn "$turn" --wandb_exp_name multi_concepts_results
#             done        
#         done
#     done
# done

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

prompt_types=("hard_simple" "hard_moderate" "hard_detailed")
concept_numbers=(2 5 10 25 50)


for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for number in "${concept_numbers[@]}"; do
            echo "Running model: $model, prompt type: $prompt_type, total concept: $number"
            torchrun --nproc-per-node=4 run.py --data MMPB --model "$model" --verbose --injection_description_prompt_type "$prompt_type" --injection_prompt_n_concepts "$number" --wandb_exp_name ex_multi_concepts_results   
        done
    done
done