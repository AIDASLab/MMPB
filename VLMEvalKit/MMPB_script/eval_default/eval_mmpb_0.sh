#!/bin/bash
# # When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior).
# # That is recommended for evaluating very large VLMs (like IDEFICS-80B-Instruct).

# # IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG, Inference and Evalution
# python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose
# # IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG, Inference only
# python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose --mode infer

# # When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference.
# # However, that is only suitable for VLMs that consume small amounts of GPU memory.

# # IDEFICS-9B-Instruct, Qwen-VL-Chat, mPLUG-Owl2 on MMBench_DEV_EN, MME, and SEEDBench_IMG. On a node with 8 GPU. Inference and Evaluation.
# torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose
# # Qwen-VL-Chat on MME. On a node with 2 GPU. Inference and Evaluation.
# torchrun --nproc-per-node=2 run.py --data MME --model qwen_chat --verbose

### For our evaluation

# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-38B-MPO --verbose --injection_description_prompt_type image --generic_conversation_n_turn 10 --wandb_exp_name default_main_results

# llava 1.5 (vlm_37)
models=(
    # "llava_v1.5_7b"
    # "llava_v1.5_13b"
    # "InternVL2_5-8B-MPO"
    # "InternVL3-38B"
    # "InternVL3-14B"

    # # "InternVL2_5-26B-MPO"
    # # "InternVL2_5-38B-MPO"
    # "InternVL2_5-78B-MPO"
    # "Qwen2.5-VL-72B-Instruct"
    # "Qwen2-VL-72B-Instruct"
    # "llava_onevision_qwen2_72b_ov"
    # "llava_next_qwen_32b"
)

prompt_types=("hard_moderate")

conversation_turns=(10)

for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for turn in "${conversation_turns[@]}"; do
            echo "Running model: $model, prompt type: $prompt_type, conversation turns: $turn"
            torchrun --nproc-per-node=3 run.py --data MMPB --model "$model" --verbose --injection_description_prompt_type "$prompt_type" --generic_conversation_n_turn "$turn" 
        done
    done
done


# models=(
#     # "llava_v1.5_7b"
#     # "llava_v1.5_13b"
#     # "InternVL2_5-8B-MPO"
#     # "InternVL3-8B"
#     "InternVL3-38B"

#     # # "InternVL2_5-26B-MPO"
#     # # "InternVL2_5-38B-MPO"
#     # "InternVL2_5-78B-MPO"
#     # "Qwen2.5-VL-72B-Instruct"
#     # "Qwen2-VL-72B-Instruct"
#     # "llava_onevision_qwen2_72b_ov"
#     # "llava_next_qwen_32b"
# )

# prompt_types=("image" "hard_moderate")

# conversation_turns=(0 10)

# for model in "${models[@]}"; do
#     for prompt_type in "${prompt_types[@]}"; do
#         for turn in "${conversation_turns[@]}"; do
#             echo "Running model: $model, prompt type: $prompt_type, conversation turns: $turn"
#             torchrun --nproc-per-node=3 run.py --data MMPB --model "$model" --verbose --injection_description_prompt_type "$prompt_type" --generic_conversation_n_turn "$turn" --wandb_exp_name default_main_results
#         done
#     done
# done


# # qwen2.5 (vlm_latest)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Qwen2.5-VL-7B-Instruct --verbose

# # gpt4o (vlm_latest)
# CUDA_VISIBLE_DEVICES=0,1,2 python run.py --data MMPB --model llava_onevision_qwen2_72b_ov --verbose

# # ovis2 (vlm_latest)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Ovis2-8B --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Ovis2-16B --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Ovis2-34B --verbose

# # deepseekvl2 (vlm_38)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model deepseek_vl2 --verbose

# # paligemma (vlm_latest)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model paligemma-3b-mix-448 --verbose

# # llama3.2 (vlm_latest)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Llama-3.2-11B-Vision-Instruct --verbose

# python run.py --data MMPB --model Claude3-5V_Haiku --verbose --injection_description_prompt_type hard_moderate --generic_conversation_n_turn 0 --wandb_exp_name default_main_results
# python run.py --data MMPB --model Claude3-7V_Sonnet --verbose --injection_description_prompt_type hard_moderate --generic_conversation_n_turn 0 --wandb_exp_name default_main_results