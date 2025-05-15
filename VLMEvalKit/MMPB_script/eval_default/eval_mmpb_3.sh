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
# torchrun --nproc-per-node=2 run.py --data MMPB --model deepseek_vl2 --verbose
### For our evaluation

# # llava 1.5 (vlm_37)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_v1.5_7b --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_v1.5_13b --verbose

# # llava next (vlm_latest)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_vicuna_7b --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_vicuna_13b --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_yi_34b --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_qwen_32b --verbose

# # llava onevision (vlm_latest)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava-onevision-qwen2-7b-ov-hf --verbose

# # internvl2.5 (vlm_37)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-8B-MPO --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-26B-MPO --verbose
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-38B-MPO --verbose

# qwen2.5 (vlm_latest)
# CUDA_VISIBLE_DEVICES=3 python run.py --data MMPB --model llava_next_mistral_7b --verbose --injection_description_prompt_type hard_moderate --generic_conversation_n_turn 10 --wandb_exp_name default_main_results
# CUDA_VISIBLE_DEVICES=3 python run.py --data MMPB --model llava_next_mistral_7b --verbose --injection_description_prompt_type hard_moderate --generic_conversation_n_turn 0 --wandb_exp_name default_main_results
# python run.py --data MMPB --model InternVL2_5-78B-MPO --verbose --injection_description_prompt_type hard_moderate --generic_conversation_n_turn 0 --wandb_exp_name default_main_results
models=(
    # "Ovis2-16B"
    "Ovis2-34B"
    # "InternVL2_5-78B-MPO"
)

prompt_types=("image")

conversation_turns=(0)

for model in "${models[@]}"; do
    for prompt_type in "${prompt_types[@]}"; do
        for turn in "${conversation_turns[@]}"; do
            echo "Running model: $model, prompt type: $prompt_type, conversation turns: $turn"
            CUDA_VISIBLE_DEVICES=3 python run.py --data MMPB --model "$model" --verbose --injection_description_prompt_type "$prompt_type" --generic_conversation_n_turn "$turn" --wandb_exp_name default_main_results
        done
    done
done
# # deepseekvl2 (vlm_38)
# CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model deepseek_vl2 --verbose

# llama3.2 (vlm_latest)
