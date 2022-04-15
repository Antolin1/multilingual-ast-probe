import glob
import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable
from scipy.linalg import subspace_angles
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

LANGS = ['python', 'javascript', 'go', 'ruby', 'c', 'java', 'csharp']


def report_angle(args):
    # report angles between subspaces
    dic_performances = {}
    dic_models = {}
    for file in glob.glob(args.run_dir_monolingual + "/*/metrics.log"):
        lang = os.path.dirname(file).split('/')[-1]
        checkpoint = torch.load(os.path.join(os.path.dirname(file), 'pytorch_model.bin'),
                                map_location=torch.device('cpu'))
        proj = checkpoint['proj'].cpu().detach().numpy()
        dic_models[lang] = proj
        with open(file, 'rb') as f:
            data = pickle.load(f)
            dic_performances[lang] = data['test_f1']

    table_sim_rad = PrettyTable()
    table_sim_rad.field_names = ["----"] + LANGS
    table_sim_ang = PrettyTable()
    table_sim_ang.field_names = ["----"] + LANGS

    dic_rad = {}
    for l1 in LANGS:
        row_rad = [l1]
        row_ang = [l1]
        for l2 in LANGS:
            p1 = dic_models[l1]
            p2 = dic_models[l2]
            subspace_sim_rad = np.mean(subspace_angles(p1, p2))
            subspace_sim_ang = np.rad2deg(subspace_sim_rad)
            row_rad.append(round(subspace_sim_rad, 4))
            row_ang.append(round(subspace_sim_ang, 4))
            if l1 != l2:
                dic_rad[f'{l1}_transfer_{l2}'] = round(subspace_sim_rad, 4)
            # print(f'{l1} vs {l2}, {np.mean(subspace_angles(p1, p2))}, {np.rad2deg(np.mean(subspace_angles(p1, p2)))}')
        table_sim_rad.add_row(row_rad)
        table_sim_ang.add_row(row_ang)

    print(table_sim_rad)
    print(table_sim_ang)
    return dic_rad, dic_performances


def report_direct_transfer(args, dic_rad, dic_performances):
    dic_results = {}
    for file in glob.glob(args.run_dir_direct_transfer + "/*/metrics.log"):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        name = os.path.dirname(file).split('/')[-1]
        dic_results[name] = data['test_f1']

    table = PrettyTable()
    table.field_names = ["Training/Testing"] + LANGS
    for l1 in LANGS:
        row = [l1]
        for l2 in LANGS:
            if l1 != l2:
                f1 = dic_results[f'{l1}_transfer_{l2}']
            else:
                f1 = dic_performances[l1]
            row.append(f1)
        table.add_row(row)
    print(table)
    correlations = []
    for j in LANGS:
        f1 = []
        angle = []
        for i in LANGS:
            if i != j:
                f1.append(dic_results[f'{i}_transfer_{j}'])
                angle.append(dic_rad[f'{i}_transfer_{j}'])
            else:
                pass
                # angle.append(0)
                # f1.append(dic_performances[i])
        print(f'For {j} the coef is {stats.spearmanr(f1, angle)}')
        correlations.append(stats.spearmanr(f1, angle).correlation)
    print(f'Mean correlations: {np.mean(correlations)}')
    return dic_results


def load_model_elements(checkpoint):
    loaded_model = torch.load(checkpoint,
                              map_location=torch.device('cpu'))
    vectors_c = loaded_model['vectors_c'].cpu().detach().numpy().T
    vectors_u = loaded_model['vectors_u'].cpu().detach().numpy().T
    proj = loaded_model['proj'].cpu().detach().numpy()
    return proj, vectors_c, vectors_u


def projection_direct_transfer(args, dic_results, label='c'):
    dic_vectors_target = {}
    dic_vocab_source = {}
    dic_vocab_target = {}
    dic_vectors_source = {}
    for file in glob.glob(args.run_dir_direct_transfer + "/*/metrics.log"):
        name = os.path.dirname(file).split('/')[-1]
        checkpoint_target = os.path.join(os.path.dirname(file), 'pytorch_model.bin')
        if label == 'c':
            _, vectors_target, _ = load_model_elements(checkpoint_target)
        else:
            _, _, vectors_target = load_model_elements(checkpoint_target)
        dic_vectors_target[name] = vectors_target

        source = name.split('_transfer_')[0]
        checkpoint_source = os.path.join(args.run_dir_monolingual, source, 'pytorch_model.bin')
        if label == 'c':
            _, vectors_source, _ = load_model_elements(checkpoint_source)
        else:
            _, _, vectors_source = load_model_elements(checkpoint_source)
        dic_vectors_source[name] = vectors_source

        target = name.split('_transfer_')[1]
        with open(os.path.join(args.dataset_path, source, 'labels.pkl'), 'rb') as f:
            data = pickle.load(f)
            if label == 'c':
                dic_vocab_source[name] = data['ids_to_labels_c']
            else:
                dic_vocab_source[name] = data['ids_to_labels_u']
        with open(os.path.join(args.dataset_path, target, 'labels.pkl'), 'rb') as f:
            data = pickle.load(f)
            if label == 'c':
                dic_vocab_target[name] = data['ids_to_labels_c']
            else:
                dic_vocab_target[name] = data['ids_to_labels_u']

    dic_sh_score = {}
    for i, (case, source_vectors) in enumerate(dic_vectors_source.items()):
        target_vectors = dic_vectors_target[case]
        n_vectors_source = source_vectors.shape[0]
        n_vectors_target = target_vectors.shape[0]
        vectors = np.concatenate((source_vectors, target_vectors), axis=0)
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        sh_score = silhouette_score(vectors,
                                    np.concatenate((np.zeros(n_vectors_source),
                                    np.ones(n_vectors_target))))
        print(case, sh_score)
        dic_sh_score[case] = sh_score

        vectors = np.concatenate((source_vectors, target_vectors), axis=0)
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        v_c_2d = TSNE(n_components=2, learning_rate='auto',
                      init='random', random_state=args.seed).fit_transform(vectors)
        v_c_source = v_c_2d[0:n_vectors_source, :]
        v_c_target = v_c_2d[n_vectors_source:, :]

        plt.figure(i, figsize=(20, 20))
        plt.scatter(v_c_source[:, 0], v_c_source[:, 1], color='blue')
        plt.scatter(v_c_target[:, 0], v_c_target[:, 1], color='red')

        for ix, l in dic_vocab_source[case].items():
            plt.annotate(l, (v_c_source[ix, 0], v_c_source[ix, 1]))
        for ix, l in dic_vocab_target[case].items():
            plt.annotate(l, (v_c_target[ix, 0], v_c_target[ix, 1]))
        plt.show()
        plt.savefig(f'plots/{case}_scatter_{label}.png')

    table = PrettyTable()
    table.field_names = ["----"] + LANGS
    for l1 in LANGS:
        row = [l1]
        for l2 in LANGS:
            if l1 != l2:
                sh = dic_sh_score[f'{l1}_transfer_{l2}']
            else:
                sh = -1
            row.append(sh)
        table.add_row(row)
    print(table)

    correlations = []
    for j in LANGS:
        f1 = []
        sh = []
        for i in LANGS:
            if i != j:
                f1.append(dic_results[f'{i}_transfer_{j}'])
                sh.append(dic_sh_score[f'{i}_transfer_{j}'])
            else:
                pass
                # angle.append(0)
                # f1.append(dic_performances[i])
        print(f'For {j} the coef is {stats.spearmanr(f1, sh)}')
        correlations.append(stats.spearmanr(f1, sh).correlation)
    print(f'Mean correlations: {np.mean(correlations)}')


def main():
    parser = argparse.ArgumentParser(description='Script for reporting the results')
    parser.add_argument('--run_dir_monolingual', default='./runs_monolingual', help='Path of the run logs')
    parser.add_argument('--run_dir_direct_transfer', default='./runs_direct_transfer', help='Path of the run logs')
    parser.add_argument('--dataset_path', default='./dataset')
    parser.add_argument('--seed', help='seed.', type=int, default=123)
    args = parser.parse_args()

    dic_rad, dic_performances = report_angle(args)
    dic_results = report_direct_transfer(args, dic_rad, dic_performances)
    projection_direct_transfer(args, dic_results, 'c')
    projection_direct_transfer(args, dic_results, 'u')


if __name__ == '__main__':
    main()
