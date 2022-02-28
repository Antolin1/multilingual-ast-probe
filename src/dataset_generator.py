import logging
import argparse
import random
import os

import numpy as np
import networkx as nx
import torch
from transformers import AutoTokenizer
from tree_sitter import Parser
from datasets import load_dataset

from data import download_codesearchnet_dataset, PY_LANGUAGE, JS_LANGUAGE, GO_LANGUAGE
from data.code2ast import code2ast, get_tokens_ast, has_error
from data.utils import match_tokenized_to_untokenized_roberta


tokenizer_roberta = AutoTokenizer.from_pretrained('roberta-base')
tokenizer_t5 = AutoTokenizer.from_pretrained('Salesforce/codet5-base')
tokenizer_codebert = AutoTokenizer.from_pretrained('microsoft/codebert-base')
tokenizer_graphcodebert = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')
tokenizer_codeberta = AutoTokenizer.from_pretrained('huggingface/CodeBERTa-small-v1')

tokenizers = [tokenizer_roberta, tokenizer_t5,
              tokenizer_codebert, tokenizer_graphcodebert,
              tokenizer_codeberta]


def filter_by_tokens_subtokens(code, lang, parser):
    try:
        G, code_pre = code2ast(code=code, parser=parser, lang=lang)
        assert nx.is_tree(nx.Graph(G))
        assert nx.is_connected(nx.Graph(G))
    except:
        return False
    if has_error(G):
        return False
    tokens = get_tokens_ast(G, code_pre)
    #filter for roberta, codebert, t5, graphcodebert, codeberta
    for tokenizer in tokenizers:
        to_convert, _ = match_tokenized_to_untokenized_roberta(tokens, tokenizer)
        if (len(to_convert) + 2) > 512:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Script for generating the dataset for probing')
    parser.add_argument("--dir", default="dataset", help="Path to save the dataset")
    parser.add_argument("--seed", help="seed.", type=int, default=123)
    parser.add_argument("--lang", help="Language.", choices=['javascript', 'python', 'go'],
                        default="python")
    parser.add_argument("--download", help="If download the csn", action="store_true")

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    #parse arguments
    args = parser.parse_args()
    dataset_dir = args.dir
    seed = args.seed
    lang = args.lang
    
    #seed everything
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    #download dataset
    if args.download:
        download_codesearchnet_dataset(dataset_dir=dataset_dir)
    dataset_path = os.path.join(dataset_dir, lang, 'dataset.jsonl')
    logger.info('Loading huggingface dataset.')
    hugg_dataset = load_dataset('json', data_files=dataset_path, split='train')

    # select the parser
    parser_lang = Parser()
    if lang == 'python':
        parser_lang.set_language(PY_LANGUAGE)
    elif lang == 'javascript':
        parser_lang.set_language(JS_LANGUAGE)
    elif lang == 'go':
        parser_lang.set_language(GO_LANGUAGE)

    #filter dataset
    logger.info('Filtering dataset.')
    hugg_dataset = hugg_dataset.filter(lambda e: filter_by_tokens_subtokens(e['original_string'], lang,
                                                                            parser_lang))
    logger.info('Shuffling dataset.')
    hugg_dataset = hugg_dataset.shuffle(seed)

    logger.info('Splitting dataset.')
    train_dataset = hugg_dataset.select(range(0, 20000))
    test_dataset = hugg_dataset.select(range(20000, 24000))
    val_dataset = hugg_dataset.select(range(24000, 26000))

    logger.info('Storing dataset.')
    train_dataset.to_json(os.path.join(dataset_dir, lang, 'train.jsonl'))
    test_dataset.to_json(os.path.join(dataset_dir, lang, 'test.jsonl'))
    val_dataset.to_json(os.path.join(dataset_dir, lang, 'valid.jsonl'))


if __name__ == '__main__':
    main()
