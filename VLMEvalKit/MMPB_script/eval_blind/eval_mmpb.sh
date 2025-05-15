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

for model in "${models[@]}"; do
    echo "Running model: $model"
    torchrun --nproc-per-node=4 run.py --data MMPB --model "$model" --verbose --blind Yes --wandb_exp_name blind_results
done
