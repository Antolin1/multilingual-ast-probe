import argparse
import glob
import os
import pickle

import pandas as pd

from src.data import LANGUAGES


def read_results(args):
    data = {'model': [], 'lang': [], 'recall': [], 'f1': [], 'precision': []}
    for file in glob.glob(args.run_dir + "/*/metrics.log"):
        parent = os.path.dirname(file).split('/')[-1]
        if 'multilingual' not in parent:
            continue
        _, model = parent.split('_')
        with open(file, 'rb') as f:
            results = pickle.load(f)
        for lang in LANGUAGES:
            data['model'].append(model)
            data['lang'].append(lang)
            data['precision'].append(results[f'test_precision_{lang}'])
            data['recall'].append(results[f'test_recall_{lang}'])
            data['f1'].append(results[f'test_f1_{lang}'])
    df = pd.DataFrame(data)
    return df


def main(args):
    results = read_results(args)
    print(results.to_latex(index=False, columns=['model', 'lang', 'f1']))
    results.to_csv(args.out_csv_rq2, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for analyzing the results')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    parser.add_argument('--out_csv_rq2', default='rq2_all_data.csv', help='Csv name for the first rq2')
    args = parser.parse_args()
    main(args)
