#!/bin/bash

# list of tuples (model_name_or_path, model_type, model_name)
declare -a models=(
  # "microsoft/codebert-base roberta codebert"
  # "microsoft/graphcodebert-base roberta graphcodebert"
  # "Salesforce/codet5-base t5 codet5"
  # "huggingface/CodeBERTa-small-v1 roberta codeberta"
  # "roberta-base roberta roberta"
  # "bert-base-uncased bert bert"
  # "distilbert-base-uncased distilbert distilbert"
  # "distilroberta-base roberta distilroberta"
  "microsoft/unixcoder-base roberta unixcoder"
  "microsoft/unixcoder-base-nine roberta unixcoder-nine"
)

for model in "${models[@]}"; do
  read -a strarr <<< "$model"
  echo "${strarr[0]}" "${strarr[1]}" "${strarr[2]}"

  CUDA_VISIBLE_DEVICES=2,3 python code/run.py \
    --output_dir="./saved_models/${strarr[2]}" \
    --model_type="${strarr[1]}" \
    --tokenizer_name="${strarr[0]}" \
    --model_name_or_path="${strarr[0]}" \
    --do_train \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee "${strarr[2]}_train.log"

  CUDA_VISIBLE_DEVICES=2 python code/run.py \
    --output_dir="./saved_models/${strarr[2]}" \
    --model_type="${strarr[1]}" \
    --tokenizer_name="${strarr[0]}" \
    --model_name_or_path="${strarr[0]}" \
    --do_eval \
    --do_test \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --epoch 5 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1 | tee "${strarr[2]}_test.log"
done
