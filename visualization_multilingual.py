import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree

from src.data.binary_tree import SEPARATOR


def load_labels(args):
    labels_file_path_c = os.path.join(args.run_folder, 'global_labels_c.pkl')
    labels_file_path_u = os.path.join(args.run_folder, 'global_labels_u.pkl')
    with open(labels_file_path_c, 'rb') as f:
        labels_to_ids_c = pickle.load(f)
    with open(labels_file_path_u, 'rb') as f:
        labels_to_ids_u = pickle.load(f)
    ids_to_labels_c = {y: x for x, y in labels_to_ids_c.items()}
    ids_to_labels_u = {y: x for x, y in labels_to_ids_u.items()}
    return labels_to_ids_c, ids_to_labels_c, labels_to_ids_u, ids_to_labels_u


def load_vectors(run_folder):
    loaded_model = torch.load(os.path.join(run_folder, f'pytorch_model.bin'),
                              map_location=torch.device('cpu'))
    vectors_c = loaded_model['vectors_c'].cpu().detach().numpy().T
    vectors_u = loaded_model['vectors_u'].cpu().detach().numpy().T
    proj = loaded_model['proj'].cpu().detach().numpy()
    return vectors_c, vectors_u, proj


COLORS = {'java': 'r',
          'javascript': 'b',
          'go': 'g',
          'python': 'c',
          'c': 'm',
          'ruby': 'y',
          'csharp': 'k',
          'php': 'tab:pink'}


def run_tsne(vectors, ids_to_labels, model, perplexity=30, type_labels='constituency'):
    # vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    v_2d = TSNE(n_components=2, learning_rate='auto', perplexity=perplexity,
                init='random', random_state=args.seed).fit_transform(vectors)
    figure, axis = plt.subplots(1, figsize=(20, 20))
    axis.set_title(f"Vectors {type_labels}")
    for ix, label in ids_to_labels.items():
        l = label.split('--')[1]
        axis.scatter(v_2d[ix, 0], v_2d[ix, 1], color=COLORS[l], label=l)
    plt.show()
    plt.savefig(f'vectors_{type_labels}_{model}.png')


def to_tsv(vectors, ids_to_labels, type_labels='constituency'):
    np.savetxt(f"vectors_{type_labels}.tsv", vectors, delimiter="\t")
    langs = []
    consts = []
    for i in range(len(ids_to_labels)):
        label = ids_to_labels[i]
        lang = label.split('--')[1]
        const = label.split('--')[0]
        langs.append(lang)
        consts.append(const)

    df = pd.DataFrame(list(zip(langs, consts)),
                      columns=['Language', 'Constituency'])
    df.to_csv(f'labels_{type_labels}.tsv', index=False, sep='\t')


def compute_clustering_quality(vectors, ids_to_labels, metric='silhouette'):
    # vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    labels = []
    for idx in range(len(ids_to_labels)):
        lang = ids_to_labels[idx].split('--')[1]
        labels.append(lang)
    if metric == 'silhouette':
        print(f'silhouette: {metrics.silhouette_score(vectors, labels)}')
    elif metric == 'calinski':
        print(f'calinski: {metrics.calinski_harabasz_score(vectors, labels)}')
    elif metric == 'davies':
        print(f'davies: {metrics.davies_bouldin_score(vectors, labels)}')


def compute_analogies(vectors, ids_to_labels, source_lang='csharp', target_lang='c'):
    l2id = {y: x for x, y in ids_to_labels.items()}
    # vectors_unit = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    kd_tree = KDTree(vectors)
    list_nonterminals_source = [l.split('--')[0] for l in l2id.keys()
                                if SEPARATOR not in l and
                                l.endswith(f'--{source_lang}')]
    for nonterminal in list_nonterminals_source:
        y_source_mean = np.mean(np.stack([vectors[l2id[x]] for x in l2id if x.endswith(f'--{source_lang}')]), axis=0)
        y_target_mean = np.mean(np.stack([vectors[l2id[x]] for x in l2id if x.endswith(f'--{target_lang}')]), axis=0)
        y_diff_langs = y_target_mean - y_source_mean
        y = y_diff_langs + vectors[l2id[f'{nonterminal}--{source_lang}']]
        # y = y / np.linalg.norm(y)
        _, i = kd_tree.query([y], k=3)
        i = i[0]
        for j, idx in enumerate(i):
            if ids_to_labels[idx].endswith(f'--{target_lang}'):
                print(f'Neig k={j} for analogy {nonterminal}--{source_lang} is {ids_to_labels[idx]}')


def main(args):
    labels_to_ids_c, ids_to_labels_c, labels_to_ids_u, ids_to_labels_u = load_labels(args)
    vectors_c, vectors_u, _ = load_vectors(args.run_folder)
    run_tsne(vectors_c, ids_to_labels_c, args.model, perplexity=30, type_labels='constituency')
    run_tsne(vectors_u, ids_to_labels_u, args.model, perplexity=5, type_labels='unary')
    compute_clustering_quality(vectors_c, ids_to_labels_c, metric=args.clustering_quality_metric)
    to_tsv(vectors_c, ids_to_labels_c, type_labels='constituency')
    if args.compute_analogies:
        compute_analogies(vectors_c, ids_to_labels_c, args.source_lang, args.target_lang)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for visualizing multilingual probes')
    parser.add_argument('--run_folder', help='Run folder of the multilingual probe', required=True)
    parser.add_argument('--model', help='Model name', required=True)
    parser.add_argument('--clustering_quality_metric', help='CLustering quality metric',
                        choices=['silhouette', 'calinski', 'davies'], default='silhouette')
    parser.add_argument('--seed', default=123)
    parser.add_argument('--source_lang', default='csharp')
    parser.add_argument('--target_lang', default='c')
    parser.add_argument('--compute_analogies', action='store_true')
    args = parser.parse_args()
    main(args)
