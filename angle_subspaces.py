import argparse
import json
import os.path

import numpy as np
from prettytable import PrettyTable
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

from analyze_results_rq1 import ELEGANT_NAMES
from src.data import LANGUAGES_CSN
from visualization_multilingual import load_vectors, DEVANBU_RESULTS


def main(args):
    with open(args.out_best_layer_per_model_rq1) as json_file:
        best_layer_per_model = json.load(json_file)
    layer = None
    for dict_model in best_layer_per_model:
        if dict_model['model'] == ELEGANT_NAMES[args.model]:
            layer = dict_model['layer']
    subspaces = {}
    for lang in LANGUAGES_CSN:
        name_folder = '_'.join([args.model, lang, str(layer), '128'])
        run_folder = os.path.join(args.run_dir, name_folder)
        _, _, proj = load_vectors(run_folder)
        subspaces[lang] = proj

    table_sim_ang = PrettyTable()
    table_sim_ang.field_names = ["----"] + list(LANGUAGES_CSN)

    for x in LANGUAGES_CSN:
        row_ang = [x]
        angles = []
        bleus = []
        for y in LANGUAGES_CSN:
            subspace_sim_ang = np.rad2deg(np.mean(subspace_angles(subspaces[x], subspaces[y])))
            row_ang.append(round(subspace_sim_ang, 4))
            angles.append(subspace_sim_ang)
            bleus.append(DEVANBU_RESULTS[x][y])
        table_sim_ang.add_row(row_ang)
        print(f'Testing {x}, correlation: {spearmanr(bleus, angles)}')
    print(table_sim_ang)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for computing angle between subspaces')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    parser.add_argument('--model', help='Model name', choices=['codebert'], default='codebert')
    parser.add_argument('--out_best_layer_per_model_rq1', default='best_layer_per_model.json',
                        help='Json for the best layer per model')
    args = parser.parse_args()
    main(args)
