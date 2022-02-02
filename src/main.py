import os
import sys
import random
import logging

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, AutoModel, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from prettytable import PrettyTable
from tree_sitter import Parser

from args import ProgramArguments
from data import download_codesearchnet_dataset, create_splits, convert_sample_to_features, PY_LANGUAGE, collator_fn
from probe.probe import TwoWordPSDProbe
from probe.loss import L1DistanceLoss
from probe.run_probe import train_probe


def main(args):
    if args.download_csn:
        dataset_dir = download_codesearchnet_dataset()
        args.dataset_path_or_name = os.path.join(dataset_dir, args.lang, 'dataset.jsonl')
        create_splits(args.dataset_path_or_name)

    if args.dataset_path_or_name is None:
        raise ValueError('A dataset path or name must be provided.')

    logger.info('Loading dataset from local file.')
    data_files = {'train': os.path.join(args.dataset_path_or_name, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_path_or_name, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_path_or_name, 'test.jsonl')}
    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')

    # @todo: case when the model is loaded locally
    logger.info('Loading model and tokenizer.')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path_or_name)
    model = AutoModel.from_pretrained(args.tokenizer_path_or_name, output_hidden_states=True)

    # @todo: create a function to encapsulate everything
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser))
    test_set = test_set.remove_columns('original_string')
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=64,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer))
    
    #for b in tqdm(test_dataloader):
    #    pass

    probe = TwoWordPSDProbe(128, 768, 'cpu')
    criterion = L1DistanceLoss('cpu')
    train_probe(test_dataloader, test_dataloader, probe, model, criterion)


if __name__ == '__main__':
    parser = HfArgumentParser(ProgramArguments)
    args = parser.parse_args()

    if args.seed > 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logger.addHandler(console)

    if args.run_base_path is not None and args.run_name is not None:
        args.output_path = os.path.join(args.run_base_path, args.run_name)
        os.mkdir(args.output_path)

        file = logging.FileHandler(os.path.join(args.output_path, 'info.log'))
        file.setLevel(level=logging.INFO)
        formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
        file.setFormatter(formatter)
        logger.addHandler(file)

    logger.info('COMMAND: {}'.format(' '.join(sys.argv)))

    config_table = PrettyTable()
    config_table.field_names = ["Configuration", "Value"]
    config_table.align["Configuration"] = "l"
    config_table.align["Value"] = "l"
    for config, value in vars(args).items():
        config_table.add_row([config, str(value)])
    logger.info('Configuration:\n{}'.format(config_table))

    main(args)
