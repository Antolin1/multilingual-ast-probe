import os
import re
import shutil
import pathlib
import json
import random
import logging

from tqdm import tqdm
from tree_sitter import Language, Parser

from .utils import download_url, unzip_file
from .code2ast import code2ast, enrichAstWithDeps, getDependencyTree, getMatrixAndTokens


logger = logging.getLogger(__name__)

CSN_DATASET_SPLIT_PATH = 'https://github.com/guoday/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip'
CSN_DATASET_BASE_PATH = 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/'

LANGUAGES = (
    'python',
    'java',
    'ruby',
    'javascript',
    'go',
    'php'
)
PY_LANGUAGE = Language('grammars/languages.so', 'python')


def download_codesearchnet_dataset():
    """Download CodeSearchNet dataset and clean it using GraphCodeBERT cleaning splits (Guo et al's)

    Return:
        dataset_dir (str): the path containing the downloaded dataset.
    """
    zip_file_path = 'dataset.zip'
    dataset_dir = './dataset'

    if not os.path.exists(zip_file_path):
        logger.info('Downloading CodeSearchNet dataset...')
        download_url(CSN_DATASET_SPLIT_PATH, zip_file_path)
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    unzip_file(zip_file_path, './')

    os.chdir(dataset_dir)
    for lang in LANGUAGES:
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

    for lang in LANGUAGES:
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
                    # we only extract the original code
                    js = {'original_string': data[line]['original_string']}
                    f1.write(json.dumps(js) + '\n')
        os.remove(os.path.join(lang, 'train.txt'))
        shutil.rmtree(os.path.join(lang, 'final'))
    # clean folders
    for file in os.listdir('.'):
        if re.match('.*.(zip|pkl|py|sh)', file):
            os.remove(file)
    os.chdir('../')
    return dataset_dir


def create_splits(dataset_path: str, split: list[float] = (.8, .1)):
    dataset_dir = os.path.dirname(dataset_path)
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


def convert_sample_to_features(code, parser):
    G, pre_code = code2ast(code, parser)
    enrichAstWithDeps(G)
    T = getDependencyTree(G)
    matrix, code_tokens = getMatrixAndTokens(T, pre_code)

    return {'tokens': code_tokens, 'matrix': matrix}
