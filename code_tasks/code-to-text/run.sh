#!/bin/bash

# list of tuples (model_name_or_path, model_type, model_name)
declare -a models=(
  "microsoft/unixcoder-base roberta unixcoder"
  "microsoft/unixcoder-base-nine roberta unixcoder-nine"
  "microsoft/codebert-base roberta codebert"
  "microsoft/graphcodebert-base roberta graphcodebert"
  "Salesforce/codet5-base t5 codet5"
  "huggingface/CodeBERTa-small-v1 roberta codeberta"
  "roberta-base roberta roberta"
  "bert-base-uncased bert bert"
  "distilbert-base-uncased distilbert distilbert"
  "distilroberta-base roberta distilroberta"
)

for model in "${models[@]}"; do
  read -a strarr <<< "$model"
  echo "${strarr[0]}" "${strarr[1]}" "${strarr[2]}"

  CUDA_VISIBLE_DEVICES=2,3 python code/run.py \
    --output_dir "./saved_models/${strarr[2]}" \
    --model_type "${strarr[1]}" \
    --model_name_or_path "${strarr[0]}" \
    --tokenizer_name "${strarr[0]}" \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --beam_size 10 \
    --max_source_length 256 \
    --max_target_length 128 \
    --train_filename "./dataset/all_train_250k.jsonl" \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --do_train \
    --do_eval \
    --seed 123456 2>&1 | tee "${strarr[2]}_train.log"
done