import argparse
import os
import pickle

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from scipy.linalg import subspace_angles

from analyze_results_rq1 import ELEGANT_NAMES
from src.data import LANGUAGES
from visualization_multilingual import load_vectors


def load_f1(transfer_dir, source, target):
    path = os.path.join(transfer_dir, f"{source}_{target}.log")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['test_f1']

def compute_angle_model_tranfer(args, model, layer, transfer_dir):
    subspaces = {}
    for lang in LANGUAGES:
        name_folder = '_'.join([model, lang, str(layer), '128'])
        run_folder = os.path.join(args.run_dir, name_folder)
        _, _, proj = load_vectors(run_folder)
        subspaces[lang] = proj

    table_sim_ang = PrettyTable()
    table_sim_ang.field_names = ["----"] + list(LANGUAGES)

    table_transfer = PrettyTable()
    table_transfer.field_names = ["----"] + list(LANGUAGES)

    data = {'lang1': [], 'lang2': [], 'angle': [], 'text': [], 'model': [], 'transfer': []}
    for i, x in enumerate(LANGUAGES):
        row_ang = [x]
        row_transfer = [x]
        for j, y in enumerate(LANGUAGES):
            subspace_sim_ang = np.rad2deg(np.mean(subspace_angles(subspaces[x], subspaces[y])))
            row_ang.append(round(subspace_sim_ang, 2))
            try:
                transfer_f1 = load_f1(transfer_dir, x, y)
                row_transfer.append(round(transfer_f1, 2))
            except:
                transfer_f1 = -1
                row_transfer.append('---')
            if x != y:
                data['lang1'].append(x)
                data['lang2'].append(y)
                data['angle'].append(subspace_sim_ang)
                data['text'].append(str(round(subspace_sim_ang, 2)))
                data['model'].append(model)
                data['transfer'].append(transfer_f1)
        table_sim_ang.add_row(row_ang)
        table_transfer.add_row(row_transfer)

    df = pd.DataFrame.from_dict(data)
    return df, table_sim_ang, table_transfer

def compute_spearman(df):
    langs = ['python', 'java', 'ruby', 'javascript', 'go', 'c', 'csharp', 'php']
    df = df[df['lang1'].isin(langs) & df['lang2'].isin(langs)]
    corrs_coef = []
    for target in langs:
        filtered = df[df['lang2'] == target]
        spear = abs(filtered["angle"].corr(filtered["transfer"], method="spearman"))
        print(f'{target}: {spear}')
        corrs_coef.append(spear)
    print(f'Mean: {np.mean(corrs_coef)}')


def main(args):
    df, table_sim_ang, table_transfer = compute_angle_model_tranfer(args, args.model, args.layer, args.tranfer_dir)
    print(table_sim_ang)
    print(table_transfer)
    compute_spearman(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Angle vs transfer')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    parser.add_argument('--tranfer_dir', default='./transfer', help='Path of the transfer logs')
    parser.add_argument('--model', help='Model name', choices=list(ELEGANT_NAMES.keys()), default='codebert')
    parser.add_argument('--layer', help='Layer', type=int, default=5)
    args = parser.parse_args()
    main(args)
