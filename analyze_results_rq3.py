import argparse
import json

import pandas as pd
from scipy import stats


def compute_correlation_multilingual(rq2_dataframe, results_finetuning):
    input_lang = results_finetuning["input_lang"]
    results_finetuning_pd = pd.DataFrame.from_dict({"model": results_finetuning["model"],
                                                    "performance": results_finetuning["performance"]})
    if len(input_lang) == 1:
        rq2_dataframe_lang = rq2_dataframe[rq2_dataframe["lang"] == input_lang[0]]
    else:
        pass

    df_cd = pd.merge(results_finetuning_pd, rq2_dataframe_lang, how='inner', on='model')
    f1s = list(df_cd["f1"])
    performances = list(df_cd["performance"])
    print(f1s)
    print(performances)
    print(f'Multilingual: {stats.spearmanr(f1s, performances)}')


def main(args):
    rq1_dataframe = pd.read_csv(args.out_csv_rq1)
    rq2_dataframe = pd.read_csv(args.out_csv_rq2)
    with open(args.out_best_layer_per_model_rq1) as json_file:
        best_layer_per_model = json.load(json_file)
    with open(args.results_finetuning) as json_file:
        results_finetuning = json.load(json_file)
    compute_correlation_multilingual(rq2_dataframe, results_finetuning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for analyzing the results')
    parser.add_argument('--out_csv_rq1', default='rq1_all_data.csv', help='Csv name for the first rq1')
    parser.add_argument('--out_csv_rq2', default='rq2_all_data.csv', help='Csv name for the first rq2')
    parser.add_argument('--out_best_layer_per_model_rq1', default='best_layer_per_model.json',
                        help='Json for the best layer per model')
    parser.add_argument('--results_finetuning', help='Json for results of finetuning')
    args = parser.parse_args()
    main(args)
