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

  # java to C# training
  CUDA_VISIBLE_DEVICES=0,1 python code/run.py \
    --output_dir="./saved_models/java-to-cs/${strarr[2]}" \
    --model_type="${strarr[1]}" \
    --model_name_or_path="${strarr[0]}" \
    --tokenizer_name="${strarr[0]}" \
    --do_train \
    --do_eval \
    --train_filename=./data/train.java-cs.txt.java,./data/train.java-cs.txt.cs \
    --dev_filename=./data/valid.java-cs.txt.java,./data/valid.java-cs.txt.cs \
    --max_source_length 512 \
    --max_target_length 512 \
    --beam_size 5 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --train_steps 5000 \
    --eval_steps 1000 \
    --seed 123456  2>&1 | tee "${strarr[2]}_train.log"

  # inference
  CUDA_VISIBLE_DEVICES=0 python code/run.py \
    --output_dir="./saved_models/java-to-cs/${strarr[2]}" \
    --model_type="${strarr[1]}" \
    --model_name_or_path="${strarr[0]}" \
    --tokenizer_name="${strarr[0]}" \
    --load_model_path="./saved_models/java-to-cs/${strarr[2]}/checkpoint-best-bleu/pytorch_model.bin" \
    --do_test \
    --dev_filename=./data/valid.java-cs.txt.java,./data/valid.java-cs.txt.cs \
    --test_filename=./data/test.java-cs.txt.java,./data/test.java-cs.txt.cs \
    --max_source_length 512 \
    --max_target_length 512 \
    --beam_size 5 \
    --eval_batch_size 16 \
    --seed 123456 2>&1 | tee "${strarr[2]}_test.log"
done;