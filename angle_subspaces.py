import argparse
import json
import os.path

import numpy as np
from prettytable import PrettyTable
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr

from analyze_results_rq1 import ELEGANT_NAMES
from src.data import LANGUAGES_CSN, LANGUAGES
from visualization_multilingual import load_vectors

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


def main(args):
    with open(args.out_best_layer_per_model_rq1) as json_file:
        best_layer_per_model = json.load(json_file)
    layer = None
    for dict_model in best_layer_per_model:
        if dict_model['model'] == ELEGANT_NAMES[args.model]:
            layer = dict_model['layer']
    subspaces = {}
    for lang in LANGUAGES:
        name_folder = '_'.join([args.model, lang, str(layer), '128'])
        run_folder = os.path.join(args.run_dir, name_folder)
        _, _, proj = load_vectors(run_folder)
        subspaces[lang] = proj

    table_sim_ang = PrettyTable()
    table_sim_ang.field_names = ["----"] + list(LANGUAGES)

    for x in LANGUAGES:
        row_ang = [x]
        angles = []
        bleus = []
        for y in LANGUAGES:
            subspace_sim_ang = np.rad2deg(np.mean(subspace_angles(subspaces[x], subspaces[y])))
            row_ang.append(round(subspace_sim_ang, 4))
            if x != y and x in LANGUAGES_CSN and y in LANGUAGES_CSN:
                angles.append(subspace_sim_ang)
                bleus.append(DEVANBU_RESULTS[x][y])
        table_sim_ang.add_row(row_ang)
        print(f'Testing {x}, correlation: {spearmanr(bleus, angles)}')
    print(table_sim_ang)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for computing angle between subspaces')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    parser.add_argument('--model', help='Model name', choices=list(ELEGANT_NAMES.keys()), default='codebert')
    parser.add_argument('--out_best_layer_per_model_rq1', default='best_layer_per_model.json',
                        help='Json for the best layer per model')
    args = parser.parse_args()
    main(args)
