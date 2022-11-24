#!/bin/bash


langs=("python" "java" "ruby" "go" "javascript" "php")

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

  for lang in "${langs[@]}"; do
    echo "${lang}"

    CUDA_VISIBLE_DEVICES=2,3 python code/run.py \
      --output_dir "./saved_models/${strarr[2]}/${lang}" \
      --model_type "${strarr[1]}" \
      --model_name_or_path "${strarr[0]}" \
      --tokenizer_name "${strarr[0]}" \
      --load_model_path "./saved_models/${strarr[2]}/checkpoint-best-ppl/pytorch_model.bin" \
      --eval_batch_size 128 \
      --beam_size 10 \
      --max_source_length 256 \
      --max_target_length 128 \
      --test_filename "./dataset/${lang}/test.jsonl" \
      --do_test \
      --seed 123456 2>&1 | tee "${strarr[2]}_${lang}_test.log"
  done
done
