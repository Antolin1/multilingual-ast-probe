#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python run.py \
    --model_type roberta \
    --output_dir saved_models/CSN/codebert \
    --model_name_or_path microsoft/codebert-base  \
    --do_train \
    --do_multilingual_training \
    --train_data_file dataset/CSN/all_train_250k.jsonl \
    --num_train_epochs 5 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456
