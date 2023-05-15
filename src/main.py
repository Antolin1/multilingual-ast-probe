import logging
import os
import sys

import torch
from prettytable import PrettyTable
from transformers import HfArgumentParser

from args import ProgramArguments
from run_probing import run_probing_train, run_probing_test, run_probing_all_languages
from utils import setup_logger, set_seed


def main(args):
    if args.do_train:
        args.dataset_name_or_path = os.path.join(args.dataset_name_or_path, args.lang)
        run_probing_train(args=args)
    elif args.do_test:
        args.dataset_name_or_path = os.path.join(args.dataset_name_or_path, args.lang)
        run_probing_test(args=args)
    elif args.do_train_all_languages:
        run_probing_all_languages(args=args)
    else:
        raise ValueError('--do_train, --do_test, or do_train_all_languages should be provided.')


if __name__ == '__main__':
    parser = HfArgumentParser(ProgramArguments)
    args = parser.parse_args()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.seed > 0:
        set_seed(args.seed)

    logger = setup_logger()

    if args.run_base_path is not None and args.run_name is not None:
        args.output_path = os.path.join(args.run_base_path, args.run_name)
        os.makedirs(args.output_path, exist_ok=True)

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
