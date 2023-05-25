import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from plotnine import *
from sklearn import metrics
from sklearn.manifold import TSNE

LANGUAGES = ['java', 'python', 'javascript', 'go', 'php', 'ruby', 'csharp', 'c']
DATASET = 'dataset'


def load_model(path):
    loaded_model = torch.load(path,
                              map_location=torch.device('cpu'))
    vectors_c = loaded_model['vectors_c'].cpu().detach().numpy().T
    vectors_u = loaded_model['vectors_u'].cpu().detach().numpy().T
    proj = loaded_model['proj'].cpu().detach().numpy()
    return vectors_c, vectors_u, proj


def load_labels():
    labels = defaultdict(list)
    for language in LANGUAGES:
        path = os.path.join(DATASET, language, 'labels.pkl')
        with open(path, 'rb') as f:
            ids_to_labels = pickle.load(f)['ids_to_labels_c']
            for i in range(len(ids_to_labels)):
                labels[language].append(ids_to_labels[i])
    return labels


def load_vectors(args):
    vectors = {}
    for language in LANGUAGES:
        model_path = os.path.join(args.runs_folder, '_'.join([args.model, language, str(args.layer), '128']),
                                  'pytorch_model.bin')
        vectors_c, _, proj = load_model(model_path)
        real_vectors_c = np.matmul(vectors_c, proj.T)
        vectors[language] = real_vectors_c
    return vectors


def run_tsne(vectors):
    lang_labels = []
    all_vectors = []
    for language in vectors:
        all_vectors.append(vectors[language])
        lang_labels += [language] * vectors[language].shape[0]
    all_vectors = np.concatenate(all_vectors, axis=0)
    # normalize all vectors
    all_vectors = all_vectors / np.linalg.norm(all_vectors, axis=1)[:, np.newaxis]
    print(f'silhouette: {round(metrics.silhouette_score(all_vectors, lang_labels), 4)}')
    v_2d = TSNE(n_components=2, learning_rate='auto', perplexity=30,
                init='random', random_state=123).fit_transform(all_vectors)
    df = pd.DataFrame(v_2d, columns=['tsne1', 'tsne2'])
    df['language'] = lang_labels
    return df


def compute_most_similar(vectors, labels, in_language, nonterminal):
    vector = vectors[in_language][labels[in_language].index(nonterminal)]
    # normalize vector
    vector = vector / np.linalg.norm(vector)
    for language in vectors:
        if language == in_language:
            continue
        similarities = np.dot(vectors[language] / np.linalg.norm(vectors[language], axis=1)[:, np.newaxis], vector)
        max_sim = np.max(similarities)
        max_idx = np.argmax(similarities)
        print(f'{language}: {labels[language][max_idx]} ({round(max_sim, 4)})')


def main(args):
    # load vectors
    vectors = load_vectors(args)
    # run tsne
    df = run_tsne(vectors)
    scatter_tsne = (
            ggplot(df, aes(x='tsne1', y='tsne2', color='language')) + geom_point()
            + labs(title="", x="", y="", color="Languages")
    )
    scatter_tsne.save(f"scatter_{args.model}_{args.layer}.pdf", dpi=600)
    compute_most_similar(vectors, load_labels(), args.in_language, args.nonterminal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='codebert')
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--runs_folder', type=str, default='runs')
    parser.add_argument('--in_language', type=str, default='java')
    parser.add_argument('--nonterminal', type=str, default='method_declaration')
    args = parser.parse_args()
    main(args)
