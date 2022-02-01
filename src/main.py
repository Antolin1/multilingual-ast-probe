import os
import sys
import random
import logging

import torch
import numpy as np
from transformers import HfArgumentParser
from datasets import load_dataset
from prettytable import PrettyTable

from args import ProgramArguments
from data import download_codesearchnet_dataset, create_splits


def main(args):
    if args.download_csn:
        dataset_dir = download_codesearchnet_dataset()
        args.dataset_path_or_name = os.path.join(dataset_dir, args.lang, 'dataset.jsonl')
        create_splits(args.dataset_path_or_name)

    if args.dataset_path_or_name is not None:
        logger.info('Loading dataset from local file.')
        data_files = {'train': os.path.join(args.dataset_path_or_name, 'train.jsonl'),
                      'valid': os.path.join(args.dataset_path_or_name, 'valid.jsonl'),
                      'test': os.path.join(args.dataset_path_or_name, 'test.jsonl')}
        train_set = load_dataset('json', data_files=data_files, split='train')
        valid_set = load_dataset('json', data_files=data_files, split='valid')
        test_set = load_dataset('json', data_files=data_files, split='test')


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
