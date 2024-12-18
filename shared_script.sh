#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --do_train_all_languages \
    --run_name multilingual_CodeBERT \
    --pretrained_model_name_or_path microsoft/codebert-base \
    --model_type roberta \
    --layer 5 \
    --rank 128 \
    --epochs 20
