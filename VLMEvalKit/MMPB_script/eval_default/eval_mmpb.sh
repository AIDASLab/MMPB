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
# torchrun --nproc-per-node=3 run.py --data MMPB --model Qwen2.5-VL-7B-Instruct --verbose

### For our evaluation

# llava 1.5 (vlm_37)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_v1.5_7b --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_v1.5_13b --verbose

# llava next (vlm_latest)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_vicuna_7b --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_vicuna_13b --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_yi_34b --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava_next_qwen_32b --verbose

# llava onevision (vlm_latest)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model llava-onevision-qwen2-7b-ov-hf --verbose

# internvl2.5 (vlm_37)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-8B-MPO --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-26B-MPO --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-38B-MPO --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model InternVL2_5-78B-MPO --verbose --wandb_exp_name="0306"

# qwen2.5 (vlm_latest)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Qwen2.5-VL-7B-Instruct --verbose

# gpt4o (vlm_latest)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model GPT4o --verbose

# ovis2 (vlm_latest)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Ovis2-8B --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Ovis2-16B --verbose
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Ovis2-34B --verbose

# deepseekvl2 (vlm_38)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model deepseek_vl2 --verbose

# paligemma (vlm_latest)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model paligemma-3b-mix-448 --verbose

# llama3.2 (vlm_latest)
CUDA_VISIBLE_DEVICES=0 python run.py --data MMPB --model Llama-3.2-11B-Vision-Instruct --verbose