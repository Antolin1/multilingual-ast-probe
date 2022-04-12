import argparse
import os


def main(args):
    run_dir = 'runs_monolingual'
    for lang in ['python', 'javascript', 'go', 'php', 'ruby', 'java']:
        folder = lang
        layer = 5
        if args.baseline:
            folder = lang + '_baseline'
            layer = 0
        os.system(f"CUDA_VISIBLE_DEVICES=2 python src/main.py --do_train --run_base_path {run_dir} --run_name {folder} "
                  f"--lang {lang} --layer {layer}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating the dataset for probing')
    parser.add_argument('--baseline', help='To run the baseline', action='store_true')
    args = parser.parse_args()
    main(args)
