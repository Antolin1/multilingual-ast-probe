#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/main.py --do_train --run_name test_distilroberta --pretrained_model_name_or_path distilroberta-base --model_type roberta --lang python --layer 5 --rank 128
CUDA_VISIBLE_DEVICES=0 python src/main.py --do_train --run_name test_distilbert --pretrained_model_name_or_path distilbert-base-uncased --model_type distilbert --lang python --layer 5 --rank 128
CUDA_VISIBLE_DEVICES=0 python src/main.py --do_train --run_name test_bert --pretrained_model_name_or_path bert-base-uncased --model_type bert --lang python --layer 5 --rank 128