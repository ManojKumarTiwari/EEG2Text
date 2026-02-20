#!/bin/bash
SCRIPT_DIR=$(dirname "$0")
LOG_DIR="log"
mkdir -p "$LOG_DIR"
LOG_FILE_NAME=$(basename "$0" .sh)
LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}.log"

echo "Job started at $(date)" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=0 python finetune_eeg_llm.py \
    --downstream_dataset FACED \
    --datasets_dir <path_to_faced_lmdb> \
    --num_of_classes 9 \
    --model_dir "pth_downtasks/${LOG_FILE_NAME}" \
    --foundation_dir "pth/CSBrain.pth" \
    --model CSBrain \
    --use_pretrained_weights \
    --n_layer 12 \
    --llm_model_name "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --llm_dim 2048 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-4 \
    --weight_decay 0.01 \
    --epochs 20 \
    --warmup_epochs 5 \
    --dropout 0.1 \
    --max_target_len 128 \
    --temporal_pool_stride 2 \
    --seed 42 \
    2>&1 | tee -a "$LOG_FILE"

echo "Job completed at $(date)" | tee -a "$LOG_FILE"
