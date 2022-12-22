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
from plotnine import *

from analyze_results_rq1 import ELEGANT_NAMES
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
    df = pd.DataFrame(v_2d, columns=['tsne1', 'tsne2'])
    langs = []
    const = []
    for ix, label in ids_to_labels.items():
        l = label.split('--')[1]
        langs.append(l)
        const.append(label.split('--')[0])
    df['language'] = langs
    df['constituency'] = const
    scatter_tsne = (
            ggplot(df, aes(x='tsne1', y='tsne2', color='language')) + geom_point()
            + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
    )
    scatter_tsne.save(f"scatter_{model}_{type_labels}.pdf", dpi=600)
    scatter_tsne.save(f"scatter_{model}_{type_labels}.png", dpi=600)

    if type_labels == 'constituency':
        zoom_1 = (
                ggplot(df, aes(x='tsne1', y='tsne2', color='language', label='constituency')) + geom_point()
                + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
                + ylim((-10, 0)) + xlim((-10, 0)) + geom_text()
        )
        zoom_1.save(f"zoom_1_{model}_{type_labels}.pdf", dpi=600)

        zoom_2 = (
                ggplot(df, aes(x='tsne1', y='tsne2', color='language', label='constituency')) + geom_point()
                + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
                + ylim((-5, 0)) + xlim((0, 10)) + geom_text()
        )
        zoom_2.save(f"zoom_2_{model}_{type_labels}.pdf", dpi=600)

        zoom_3 = (
                ggplot(df, aes(x='tsne1', y='tsne2', color='language', label='constituency')) + geom_point()
                + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
                + ylim((5, 10)) + xlim((-10, 0)) + geom_text()
        )
        zoom_3.save(f"zoom_3_{model}_{type_labels}.pdf", dpi=600)

        zoom_4 = (
                ggplot(df, aes(x='tsne1', y='tsne2', color='language', label='constituency')) + geom_point()
                + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
                + ylim((20, 30)) + xlim((-5, 0)) + geom_text()
        )
        zoom_4.save(f"zoom_4_{model}_{type_labels}.pdf", dpi=600)

        zoom_5 = (
                ggplot(df, aes(x='tsne1', y='tsne2', color='language', label='constituency')) + geom_point()
                + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
                + ylim((10, 20)) + xlim((5, 10)) + geom_text()
        )
        zoom_5.save(f"zoom_5_{model}_{type_labels}.pdf", dpi=600)

        zoom_6 = (
                ggplot(df, aes(x='tsne1', y='tsne2', color='language', label='constituency')) + geom_point()
                + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
                + ylim((0, 5)) + xlim((-20, -15)) + geom_text()
        )
        zoom_6.save(f"zoom_6_{model}_{type_labels}.pdf", dpi=600)

        zoom_7 = (
                ggplot(df, aes(x='tsne1', y='tsne2', color='language', label='constituency')) + geom_point()
                + labs(title=f"Non-terminals {ELEGANT_NAMES[model]}", x="", y="")
                + ylim((-10, -20)) + xlim((-10, -0)) + geom_text()
        )
        zoom_7.save(f"zoom_7_{model}_{type_labels}.pdf", dpi=600)


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
        print(f'silhouette: {round(metrics.silhouette_score(vectors, labels), 4)}')
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
    compute_clustering_quality(vectors_c, ids_to_labels_c, metric=args.clustering_quality_metric)
    if args.model == 'codebert' or args.model == 'codebert-baseline':
        run_tsne(vectors_c, ids_to_labels_c, args.model, perplexity=30, type_labels='constituency')
        run_tsne(vectors_u, ids_to_labels_u, args.model, perplexity=5, type_labels='unary')
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
