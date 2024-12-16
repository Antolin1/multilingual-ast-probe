import argparse
import glob
import os
import pickle

import pandas as pd
from plotnine import ggplot, aes, geom_line, scale_x_continuous, labs, theme, element_text, theme_light
from scipy.stats import stats

from src.data import LANGUAGES

ELEGANT_NAMES = {'codebert': 'CodeBERT',
                 'codebert-baseline': 'CodeBERTrand',
                 'codeberta': 'CodeBERTa',
                 'codet5': 'CodeT5',
                 'graphcodebert': 'GraphCodeBERT',
                 'roberta': 'RoBERTa',
                 'distilbert': 'DistilBERT',
                 'bert': 'BERT',
                 'distilroberta': 'DistilRoBERTa'
                 }


def read_results(args):
    data = {'model': [], 'lang': [], 'layer': [], 'rank': [],
            'precision': [], 'recall': [], 'f1': []}
    for file in glob.glob(args.run_dir + "/*/metrics.log"):
        parent = os.path.dirname(file).split('/')[-1]
        if 'multilingual' in parent:
            continue
        print(parent)
        model, lang, layer, rank = parent.split('_')
        with open(file, 'rb') as f:
            results = pickle.load(f)
        data['model'].append(model)
        data['lang'].append(lang)
        data['layer'].append(int(layer))
        data['rank'].append(int(rank))
        data['precision'].append(results['test_precision'])
        data['recall'].append(results['test_recall'])
        data['f1'].append(results['test_f1'])
    df = pd.DataFrame(data)
    df_renamed = df.replace(ELEGANT_NAMES)
    return df_renamed


def plot_results_layer_vs_f1(results):
    for lang in ['python', 'java', 'ruby', 'javascript', 'go', 'c', 'csharp', 'php']:
        layer_vs_f1 = (
                ggplot(results[(results['lang'] == lang)])
                + aes(x="layer", y="f1", color='model')
                + geom_line()
                + scale_x_continuous(breaks=range(0, 13, 1))
                + labs(x="Layer", y="F1", color="Model")
                + theme(text=element_text(size=16))
                + theme_light()
        )
        layer_vs_f1.save(f"layer_vs_f1_{lang}.pdf", dpi=600)
    for model in ELEGANT_NAMES.values():
        layer_vs_f1 = (
                ggplot(results[(results['model'] == model)])
                + aes(x="layer", y="f1", color='lang')
                + geom_line()
                + scale_x_continuous(breaks=range(0, 13, 1))
                + labs(x="Layer", y="F1", color="Lang")
                + theme(text=element_text(size=16))
                + theme_light()
        )
        layer_vs_f1.save(f"layer_vs_f1_{model}.pdf", dpi=600)


def best_layer_for_each_model(results):
    group_by_model = results.groupby(['model', 'layer'])['f1'].mean().reset_index()
    print(group_by_model.head(20))
    layer_vs_f1 = (
            ggplot(group_by_model)
            + aes(x="layer", y="f1", color='model')
            + geom_line()
            + scale_x_continuous(breaks=range(0, 13, 1))
            + labs(x="Layer", y="F1", color="Model")
            + theme(text=element_text(size=16))
    )
    layer_vs_f1.save(f"layer_vs_f1_global.pdf", dpi=600)

    best_layer_per_model = (
        group_by_model
        .groupby(['model'])
        .apply(lambda group: group.loc[group['f1'] == group['f1'].max()])
        .reset_index(level=-1, drop=True)
    )
    return best_layer_per_model


def get_table(results, best_layer_per_model):
    best_layer_per_model_dict = best_layer_per_model.to_dict('records')
    results_dict = results.to_dict('records')
    results_dict_filtered = [r for r in results_dict
                             if {'model': r['model'], 'layer': r['layer']} in best_layer_per_model_dict]
    results_filtered = pd.DataFrame.from_records(results_dict_filtered)
    print(results_filtered.to_latex(index=False, columns=['model', 'layer', 'lang', 'f1']))


def statistical_test(results, best_layer_per_model):
    best_layer_per_model_dict = best_layer_per_model.to_dict('records')
    list_f1_langs = []
    for m_dict in best_layer_per_model_dict:
        model = m_dict['model']
        if model == 'UniXcoder' or model == 'UniXcoder-9':
            continue
        layer = m_dict['layer']
        f1_langs = [results[(results['model'] == model) &
                            (results['layer'] == layer) &
                            (results['lang'] == l)]['f1'].values[0]
                    for l in LANGUAGES]
        print(f'for {model} is {f1_langs}')
        list_f1_langs.append(f1_langs)
    print(f'Friedman test: {stats.friedmanchisquare(*list_f1_langs)}')
    # p_adjust = 'holm'
    # print(f'\n{posthoc_wilcoxon(list_f1_langs, p_adjust=p_adjust)}')
    # print(f'\n{posthoc_ttest(list_f1_langs, p_adjust=p_adjust)}')
    # print(f'\n{posthoc_tukey(list_f1_langs)}')
    print('Not enough samples to get significance! Future work more languages?')


def main(args):
    results = read_results(args)
    plot_results_layer_vs_f1(results)
    best_layer_per_model = best_layer_for_each_model(results)
    results.to_csv(args.out_csv_rq1, index=False)
    best_layer_per_model = best_layer_per_model.drop(columns=['f1'])
    best_layer_per_model.to_json(args.out_best_layer_per_model_rq1, orient="records")
    get_table(results, best_layer_per_model)
    statistical_test(results, best_layer_per_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for analyzing the results')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    parser.add_argument('--out_csv_rq1', default='rq1_all_data.csv', help='Csv name for the first rq1')
    parser.add_argument('--out_best_layer_per_model_rq1', default='best_layer_per_model.json',
                        help='Json for the best layer per model')
    args = parser.parse_args()
    main(args)
