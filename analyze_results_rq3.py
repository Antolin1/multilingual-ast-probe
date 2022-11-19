import argparse
import json

import pandas as pd
from scipy import stats


def get_table_monolingual(rq1_dataframe):
    group_by_model = rq1_dataframe.groupby(['model', 'layer'])['f1'].mean().reset_index()
    best_layer_per_model = (
        group_by_model
        .groupby(['model'])
        .apply(lambda group: group.loc[group['f1'] == group['f1'].max()])
        .reset_index(level=-1, drop=True)
    )
    return best_layer_per_model


def compute_correlation_multilingual(rq_dataframe, results_finetuning, mono_or_multi):
    input_lang = results_finetuning["input_lang"]
    results_finetuning_pd = pd.DataFrame.from_dict({"model": results_finetuning["model"],
                                                    "performance": results_finetuning["performance"]})
    if len(input_lang) == 1:
        rq2_dataframe_lang = rq_dataframe[rq_dataframe["lang"] == input_lang[0]]
    else:
        rq2_dataframe_lang = rq_dataframe[rq_dataframe["lang"].isin(input_lang)] \
            .groupby(['model'])['f1'].mean().reset_index()

    df_cd = pd.merge(results_finetuning_pd, rq2_dataframe_lang, how='inner', on='model')
    f1s = list(df_cd["f1"])
    performances = list(df_cd["performance"])
    print(f1s)
    print(performances)
    print(f'{mono_or_multi}: {stats.spearmanr(f1s, performances)}')


def main(args):
    rq1_dataframe = pd.read_csv(args.out_csv_rq1)
    rq1_dataframe = get_table_monolingual(rq1_dataframe)
    rq2_dataframe = pd.read_csv(args.out_csv_rq2)
    with open(args.results_finetuning) as json_file:
        results_finetuning = json.load(json_file)
    compute_correlation_multilingual(rq1_dataframe, results_finetuning, 'Monolingual')
    compute_correlation_multilingual(rq2_dataframe, results_finetuning, 'Multilingual')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for analyzing the results')
    parser.add_argument('--out_csv_rq1', default='rq1_all_data.csv', help='Csv name for the first rq1')
    parser.add_argument('--out_csv_rq2', default='rq2_all_data.csv', help='Csv name for the first rq2')
    parser.add_argument('--results_finetuning', help='Json for results of finetuning')
    args = parser.parse_args()
    main(args)
