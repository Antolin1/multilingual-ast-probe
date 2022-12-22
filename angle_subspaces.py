import argparse
import json
import os.path
from collections import defaultdict

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

from analyze_results_rq1 import ELEGANT_NAMES
from src.data import LANGUAGES_CSN, LANGUAGES
from visualization_multilingual import load_vectors
from plotnine import *

DEVANBU_RESULTS = {
    'ruby': {
        'ruby': 12.53,
        'javascript': 11.84,
        'java': 13.42,
        'go': 12.32,
        'php': 13.84,
        'python': 14.09
    },
    'javascript': {
        'ruby': 11.98,
        'javascript': 13.86,
        'java': 14.16,
        'go': 12.55,
        'php': 13.90,
        'python': 14.09
    },
    'java': {
        'ruby': 13.38,
        'javascript': 14.57,
        'java': 18.72,
        'go': 14.20,
        'php': 16.27,
        'python': 16.20
    },
    'go': {
        'ruby': 11.68,
        'javascript': 11.24,
        'java': 13.61,
        'go': 18.15,
        'php': 12.70,
        'python': 13.53
    },
    'php': {
        'ruby': 17.52,
        'javascript': 19.95,
        'java': 22.11,
        'go': 18.67,
        'php': 25.48,
        'python': 21.65
    },
    'python': {
        'ruby': 14.10,
        'javascript': 14.44,
        'java': 16.77,
        'go': 14.92,
        'php': 16.41,
        'python': 18.25
    }
}


def compute_angle_model(args, model):
    with open(args.out_best_layer_per_model_rq1) as json_file:
        best_layer_per_model = json.load(json_file)
    layer = None
    for dict_model in best_layer_per_model:
        if dict_model['model'] == ELEGANT_NAMES[model]:
            layer = dict_model['layer']
    subspaces = {}
    for lang in LANGUAGES:
        name_folder = '_'.join([model, lang, str(layer), '128'])
        run_folder = os.path.join(args.run_dir, name_folder)
        _, _, proj = load_vectors(run_folder)
        subspaces[lang] = proj

    table_sim_ang = PrettyTable()
    table_sim_ang.field_names = ["----"] + list(LANGUAGES)

    data = {'lang1': [], 'lang2': [], 'angle': [], 'text': [], 'model': []}
    for i, x in enumerate(LANGUAGES):
        row_ang = [x]
        for j, y in enumerate(LANGUAGES):
            subspace_sim_ang = np.rad2deg(np.mean(subspace_angles(subspaces[x], subspaces[y])))
            row_ang.append(round(subspace_sim_ang, 2))
            if x != y and j < i:
                data['lang1'].append(x)
                data['lang2'].append(y)
                data['angle'].append(subspace_sim_ang)
                data['text'].append(str(round(subspace_sim_ang, 2)))
                data['model'].append(model)
        table_sim_ang.add_row(row_ang)

    df = pd.DataFrame.from_dict(data)
    return df, table_sim_ang


def compare_rankings(args):
    dfs = []
    for model in ELEGANT_NAMES:
        df, table_sim_ang = compute_angle_model(args, model)
        dfs.append(df)
    df_concat = pd.concat(dfs, ignore_index=True)
    data = {'m1': [], 'm2': [], 'correlation': [], 'text': []}
    representatives = ['codebert', 'codebert-baseline']
    for m1 in representatives:
        correlations = defaultdict(list)
        for j, m2 in enumerate(ELEGANT_NAMES):
            for l in LANGUAGES:
                corr_1 = [df_concat[(df_concat['model'] == m1)
                                    & ((df_concat['lang1'] == l) | (df_concat['lang2'] == l))
                                    & ((df_concat['lang1'] == l2) | (df_concat['lang2'] == l2))]['angle'].values[0]
                          for l2 in LANGUAGES if l2 != l]
                corr_2 = [df_concat[(df_concat['model'] == m2)
                                    & ((df_concat['lang1'] == l) | (df_concat['lang2'] == l))
                                    & ((df_concat['lang1'] == l2) | (df_concat['lang2'] == l2))]['angle'].values[0]
                          for l2 in LANGUAGES if l2 != l]
                correlation = spearmanr(corr_1, corr_2).correlation
                correlations[m2].append(correlation)
                # print(f'{m1} vs {m2} in {l}: {correlation}')
        for m2, corrs in correlations.items():
            if m2 not in representatives:
                print(f'{m1} vs {m2}: {np.mean(corrs)}')
                data['m1'].append(m1)
                data['m2'].append(m2)
                data['correlation'].append(np.mean(corrs))
                data['text'].append(str(round(np.mean(corrs), 2)))

    to_plot = pd.DataFrame.from_dict(data)
    corr_encoders = (
            ggplot(mapping=aes("m2", "m1", fill="correlation"),
                   data=to_plot)
            + geom_tile() + geom_label(aes(label="text"), fill="white", size=10)
            + scale_fill_distiller(palette="YlOrRd", direction=1)
            + theme_minimal()
            + scale_x_discrete(limits=[e for e in ELEGANT_NAMES if e not in representatives])
            + scale_y_discrete(limits=representatives)
            + labs(title="", x="", y="")
            + theme(axis_text_x=element_text(rotation=45, hjust=1, size=12),
                    axis_text_y=element_text(size=12),
                    legend_text=element_text(size=10))
    )
    corr_encoders.save(f"corr_encoders.pdf", dpi=600)


def main(args):
    df, table_sim_ang = compute_angle_model(args, args.model)
    print(table_sim_ang)
    print(df.head(10))
    angles_p9 = (
            ggplot(mapping=aes("lang1", "lang2", fill="angle"),
                   data=df)
            + geom_tile() + geom_label(aes(label="text"), fill="white", size=10)
            + scale_fill_distiller()
            + theme_minimal()
            + scale_x_discrete(limits=LANGUAGES)
            + scale_y_discrete(limits=LANGUAGES)
            + labs(title="", x="", y="")
            + theme(axis_text_x=element_text(rotation=45, hjust=1, size=12),
                    axis_text_y=element_text(size=12),
                    legend_text=element_text(size=10))
    )
    angles_p9.save(f"angle_{args.model}.pdf", dpi=600)

    compare_rankings(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for computing angle between subspaces')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    parser.add_argument('--model', help='Model name', choices=list(ELEGANT_NAMES.keys()), default='codebert')
    parser.add_argument('--out_best_layer_per_model_rq1', default='best_layer_per_model.json',
                        help='Json for the best layer per model')
    args = parser.parse_args()
    main(args)
