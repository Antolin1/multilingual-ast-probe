import json
import logging
import os
import pathlib
import random
import re
import shutil
from collections import Counter

import gdown
from tqdm import tqdm
from tree_sitter import Language, Parser

from .binary_tree import ast2binary, tree_to_distance
from .code2ast import code2ast, get_tokens_ast
from .utils import download_url, unzip_file

logger = logging.getLogger(__name__)

CSN_DATASET_SPLIT_PATH = 'https://github.com/guoday/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip'
CSN_DATASET_BASE_PATH = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/'

LANGUAGES = (
    'python',
    'java',
    'ruby',
    'javascript',
    'go',
    'c',
    'csharp',
    'php'
)

LANGUAGES_CSN = (
    'python',
    'java',
    'ruby',
    'javascript',
    'go',
    'php'
)

PY_LANGUAGE = Language('grammars/languages.so', 'python')
JS_LANGUAGE = Language('grammars/languages.so', 'javascript')
GO_LANGUAGE = Language('grammars/languages.so', 'go')
PHP_LANGUAGE = Language('grammars/languages.so', 'php')
JAVA_LANGUAGE = Language('grammars/languages.so', 'java')
RUBY_LANGUAGE = Language('grammars/languages.so', 'ruby')
CSHARP_LANGUAGE = Language('grammars/languages.so', 'c_sharp')
C_LANGUAGE = Language('grammars/languages.so', 'c')

PY_PARSER = Parser()
PY_PARSER.set_language(PY_LANGUAGE)
JS_PARSER = Parser()
JS_PARSER.set_language(JS_LANGUAGE)
GO_PARSER = Parser()
GO_PARSER.set_language(GO_LANGUAGE)
PHP_PARSER = Parser()
PHP_PARSER.set_language(PHP_LANGUAGE)
JAVA_PARSER = Parser()
JAVA_PARSER.set_language(JAVA_LANGUAGE)
RUBY_PARSER = Parser()
RUBY_PARSER.set_language(RUBY_LANGUAGE)
CSHARP_PARSER = Parser()
CSHARP_PARSER.set_language(CSHARP_LANGUAGE)
C_PARSER = Parser()
C_PARSER.set_language(C_LANGUAGE)

LANG_OBJECT_BY_NAME = {
    'python': PY_LANGUAGE,
    'java': JAVA_LANGUAGE,
    'ruby': RUBY_LANGUAGE,
    'javascript': JS_LANGUAGE,
    'go': GO_LANGUAGE,
    'c': C_LANGUAGE,
    'csharp': CSHARP_LANGUAGE,
    'php': PHP_LANGUAGE
}

PARSER_OBJECT_BY_NAME = {
    'python': PY_PARSER,
    'java': JAVA_PARSER,
    'ruby': RUBY_PARSER,
    'javascript': JS_PARSER,
    'go': GO_PARSER,
    'c': C_PARSER,
    'csharp': CSHARP_PARSER,
    'php': PHP_PARSER
}


def download_codexglue_c(dataset_dir):
    url = 'https://drive.google.com/uc?id=1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF'
    name_file = 'function.json'
    gdown.download(url, name_file, quiet=False)

    with open(name_file) as functions:
        js_all = json.load(functions)
    filename = os.path.join(dataset_dir, 'c', 'dataset.jsonl')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        for j, entry in enumerate(js_all):
            new_entry = {'original_string': entry['func']}
            json.dump(new_entry, outfile)
            if j < len(js_all) - 1:
                outfile.write('\n')
    os.remove(name_file)


def download_codexglue_csharp(dataset_dir):
    links = [
        'https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/test.java-cs.txt.cs',
        'https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/train.java-cs.txt.cs',
        'https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/Code-Code/code-to-code-trans/data/valid.java-cs.txt.cs']
    names_files = ['test.cs', 'train.cs', 'valid.cs']

    for l, name in zip(links, names_files):
        download_url(l, name)

    data = []
    for name in names_files:
        with open(name) as fp:
            lines = fp.readlines()
            data += [{'original_string': l} for l in lines]
    filename = os.path.join(dataset_dir, 'csharp', 'dataset.jsonl')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as outfile:
        for j, entry in enumerate(data):
            json.dump(entry, outfile)
            if j < len(data) - 1:
                outfile.write('\n')

    for name in names_files:
        os.remove(name)


def download_codesearchnet_dataset(dataset_dir):
    """Download CodeSearchNet dataset and clean it using GraphCodeBERT cleaning splits (Guo et al's)

    Return:
        dataset_dir (str): the path containing the downloaded dataset.
    """
    zip_file_path = 'dataset.zip'

    if not os.path.exists(zip_file_path):
        logger.info('Downloading CodeSearchNet dataset...')
        download_url(CSN_DATASET_SPLIT_PATH, zip_file_path)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    unzip_file(zip_file_path, './')

    os.chdir(dataset_dir)
    for lang in LANGUAGES_CSN:
        logger.info(f'Creating {lang} dataset.')
        try:
            os.remove(os.path.join(lang, 'codebase.txt'))
            os.remove(os.path.join(lang, 'test.txt'))
            os.remove(os.path.join(lang, 'valid.txt'))
        except:
            pass
        if not os.path.exists(os.path.join(lang, 'final')):
            logger.info(f'Downloading CodeSearchNet {lang} dataset.')
            download_url(os.path.join(CSN_DATASET_BASE_PATH, f'{lang}.zip'), f'{lang}.zip')
            unzip_file(f'{lang}.zip', './')
        # we care about the training set that we can further split into train/val/test
        if os.path.exists(os.path.join(lang, 'final/jsonl/test')):
            shutil.rmtree(os.path.join(lang, 'final/jsonl/test'))
        if os.path.exists(os.path.join(lang, 'final/jsonl/valid')):
            shutil.rmtree(os.path.join(lang, 'final/jsonl/valid'))

    for lang in LANGUAGES_CSN:
        logger.info(f'Cleaning {lang} dataset.')
        data = {}
        # gzip all .gz files and add them to `data` with their url as key
        for file in tqdm(pathlib.Path(f'./{lang}').rglob('*.gz')):
            unzip_file(str(file), '', str(file)[:-3])
            os.remove(file)
            with open(str(file)[:-3]) as f:
                for line in f:
                    js = json.loads(line)
                    data[js['url']] = js
        with open(f'./{lang}/dataset.jsonl', 'w') as f1, open(f'./{lang}/train.txt', encoding='utf-8') as f2:
            for line in f2:
                line = line.strip()
                # we only keep code snippets that are clean (based on GraphCodeBERT cleaning)
                #   by matching the url with a key in `data`.
                if line in data:
                    # we only extract the original code and the code tokens to filter
                    js = {'original_string': data[line]['original_string'],
                          'code_tokens': data[line]['code_tokens']}
                    f1.write(json.dumps(js) + '\n')
        os.remove(os.path.join(lang, 'train.txt'))
        shutil.rmtree(os.path.join(lang, 'final'))
    # clean folders
    for file in os.listdir('.'):
        if re.match('.*.(zip|pkl|py|sh)', file):
            os.remove(file)
    os.chdir('../')


def create_splits(dataset_path, split):
    dataset_dir = os.path.dirname(dataset_path)
    if (os.path.isfile(os.path.join(dataset_dir, 'train.jsonl'))
            and os.path.isfile(os.path.join(dataset_dir, 'test.jsonl'))
            and os.path.isfile(os.path.join(dataset_dir, 'valid.jsonl'))):
        logger.info('Splits already created.')
        return
    with open(dataset_path, 'r') as f:
        data = list(f)
        # we already seeded random package in the main
        random.shuffle(data)
        total_count = len(data)
        train_count = int(split[0] * total_count)
        valid_count = int(split[1] * total_count)
        train_data = data[:train_count]
        valid_data = data[train_count:train_count + valid_count]
        test_data = data[train_count + valid_count:]
    with open(os.path.join(dataset_dir, 'train.jsonl'), 'w') as f1, \
            open(os.path.join(dataset_dir, 'valid.jsonl'), 'w') as f2, \
            open(os.path.join(dataset_dir, 'test.jsonl'), 'w') as f3:
        logger.info('Creating training split.')
        for line in tqdm(train_data):
            js = json.loads(line)
            f1.write(json.dumps(js) + '\n')
        logger.info('Creating validation split.')
        for line in tqdm(valid_data):
            js = json.loads(line)
            f2.write(json.dumps(js) + '\n')
        logger.info('Creating test split.')
        for line in tqdm(test_data):
            js = json.loads(line)
            f3.write(json.dumps(js) + '\n')


def convert_sample_to_features(code, parser, lang):
    G, pre_code = code2ast(code, parser, lang)
    binary_ast = ast2binary(G)
    d, c, _, u = tree_to_distance(binary_ast, 0)
    code_tokens = get_tokens_ast(G, pre_code)

    return {
        'd': d,
        'c': c,
        'u': u,
        'num_tokens': len(code_tokens),
        'code_tokens': code_tokens
    }


def get_non_terminals_labels(train_set_labels, valid_set_labels, test_set_labels):
    all_labels = [label for seq in train_set_labels for label in seq] + \
                 [label for seq in valid_set_labels for label in seq] + \
                 [label for seq in test_set_labels for label in seq]
    # use a Counter to constantly get the same order in the labels
    ct = Counter(all_labels)
    labels_to_ids = {}
    for i, label in enumerate(ct):
        labels_to_ids[label] = i
    return labels_to_ids


def convert_to_ids(c, column_name, labels_to_ids):
    labels_ids = []
    for label in c:
        labels_ids.append(labels_to_ids[label])
    return {column_name: labels_ids}


def convert_to_ids_multilingual(c, column_name, labels_to_ids, lang):
    labels_ids = []
    for label in c:
        labels_ids.append(labels_to_ids[label + '--' + lang])
    return {column_name: labels_ids}


def get_mask_multilingual(ids_to_labels, lang):
    result = []
    for i, _ in enumerate(ids_to_labels.keys()):
        label = ids_to_labels[i]
        if label.endswith('--' + lang):
            result.append(0)
        else:
            result.append(-float("Inf"))
    return result
