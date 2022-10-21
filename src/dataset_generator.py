import argparse
import os
import random

import networkx as nx
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from data import download_codesearchnet_dataset, download_codexglue_csharp, download_codexglue_c, PARSER_OBJECT_BY_NAME
from data.code2ast import code2ast, has_error, get_tokens_ast
from data.utils import match_tokenized_to_untokenized_roberta
from main import setup_logger

tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
tokenizer_t5 = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
tokenizer_codebert = AutoTokenizer.from_pretrained('microsoft/codebert-base')
tokenizer_graphcodebert = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
tokenizer_codeberta = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

tokenizers = [tokenizer_roberta, tokenizer_t5, tokenizer_codebert, tokenizer_graphcodebert, tokenizer_codeberta]


def filter_samples(code, max_length, lang, parser):
    try:
        G, code_pre = code2ast(code=code, parser=parser, lang=lang)
        assert nx.is_tree(nx.Graph(G))
        assert nx.is_connected(nx.Graph(G))
    except:
        return False
    if has_error(G):
        return False
    code_tokens = get_tokens_ast(G, code_pre)
    for tokenizer in tokenizers:
        t, _ = match_tokenized_to_untokenized_roberta(untokenized_sent=code_tokens, tokenizer=tokenizer)
        if len(t) + 2 > max_length:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating the dataset for probing')
    parser.add_argument('--dataset_dir', default='./dataset', help='Path to save the dataset')
    parser.add_argument('--lang', help='Language.', choices=['javascript', 'python', 'go', 'php',
                                                             'java', 'ruby', 'csharp', 'c'],
                        default='python')
    parser.add_argument('--max_code_length', help='Maximum code length.', default=512)
    parser.add_argument('--download_csn', help='If download the csn', action='store_true')
    parser.add_argument('--download_cxg', help='If download the cxg csharp', action='store_true')
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    args = parser.parse_args()

    logger = setup_logger()

    # seed everything
    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # download dataset
    if args.download_csn:
        download_codesearchnet_dataset(dataset_dir=args.dataset_dir)

    if args.download_cxg:
        download_codexglue_csharp(dataset_dir=args.dataset_dir)
        download_codexglue_c(dataset_dir=args.dataset_dir)

    dataset_path = os.path.join(args.dataset_dir, args.lang, 'dataset.jsonl')
    logger.info('Loading dataset.')
    dataset = load_dataset('json', data_files=dataset_path, split='train')

    # select the parser
    parser_lang = PARSER_OBJECT_BY_NAME[args.lang]

    # filter dataset
    logger.info('Filtering dataset.')
    dataset = dataset.filter(
        lambda e: filter_samples(e['original_string'], args.max_code_length, args.lang, parser_lang), num_proc=8,
        load_from_cache_file=False)
    dataset = dataset.shuffle(seed=args.seed)
    dataset = dataset.select(range(0, min(20000, len(dataset))))
    logger.info(f'Dataset points {len(dataset)}')

    logger.info('Splitting dataset 70/20/10.')
    ds_split_train_test = dataset.train_test_split(test_size=0.20, seed=args.seed, shuffle=True)
    train_ds, test_ds = ds_split_train_test["train"], ds_split_train_test["test"]
    ds_split_train_val = train_ds.train_test_split(test_size=0.1 / 0.8, seed=args.seed, shuffle=True)
    train_ds, val_ds = ds_split_train_val["train"], ds_split_train_val["test"]

    logger.info('Storing dataset.')
    logger.info(f'Train datapoints {len(train_ds)}')
    logger.info(f'Test datapoints {len(test_ds)}')
    logger.info(f'Val datapoints {len(val_ds)}')
    train_ds.to_json(os.path.join(args.dataset_dir, args.lang, 'train.jsonl'))
    test_ds.to_json(os.path.join(args.dataset_dir, args.lang, 'test.jsonl'))
    val_ds.to_json(os.path.join(args.dataset_dir, args.lang, 'valid.jsonl'))
