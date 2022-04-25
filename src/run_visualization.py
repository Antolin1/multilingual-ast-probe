import glob
import logging
import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from openTSNE import TSNE
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from tree_sitter import Parser
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px

from data import PY_LANGUAGE, JS_LANGUAGE, GO_LANGUAGE
from data import convert_sample_to_features, LANGUAGES, collator_with_mask
from data.binary_tree import ast2binary, tree_to_distance, distance_to_tree, \
    extend_complex_nodes, add_unary, remove_empty_nodes, get_precision_recall_f1, \
    get_recall_non_terminal, SEPARATOR
from data.code2ast import code2ast, get_tokens_ast
from data.data_loading import convert_to_ids_multilingual
from data.utils import match_tokenized_to_untokenized_roberta
from probe import ParserProbe
from probe.utils import get_embeddings, align_function
from run_probing import generate_baseline
from run_probing import get_lmodel, parsers

logger = logging.getLogger(__name__)


# todo: add the dictionaries for the classification
def run_visualization(args):
    code_samples = []
    if args.lang == 'python':
        for filename in glob.glob('code_samples/*.py'):
            with open(filename, 'r') as f:
                code_samples.append(f.read())
    elif args.lang == 'javascript':
        for filename in glob.glob('code_samples/*.js'):
            with open(filename, 'r') as f:
                code_samples.append(f.read())
    elif args.lang == 'go':
        for filename in glob.glob('code_samples/*.go'):
            with open(filename, 'r') as f:
                code_samples.append(f.read())

    # @todo: load lmodel and tokenizer from checkpoint
    # @todo: model_type in ProgramArguments
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    if args.model_type == 't5':
        lmodel = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
        lmodel = lmodel.to(args.device)
    else:
        lmodel = AutoModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
        if '-baseline' in args.run_name:
            lmodel = generate_baseline(lmodel)
        lmodel = lmodel.to(args.device)

    # select the parser
    parser = Parser()
    if args.lang == 'python':
        parser.set_language(PY_LANGUAGE)
    elif args.lang == 'javascript':
        parser.set_language(JS_LANGUAGE)
    elif args.lang == 'go':
        parser.set_language(GO_LANGUAGE)

    # load the labels
    labels_file_path = os.path.join(args.dataset_name_or_path, 'labels.pkl')
    with open(labels_file_path, 'rb') as f:
        data = pickle.load(f)
        labels_to_ids_c = data['labels_to_ids_c']
        ids_to_labels_c = data['ids_to_labels_c']
        labels_to_ids_u = data['labels_to_ids_u']
        ids_to_labels_u = data['ids_to_labels_u']

    final_probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)

    final_probe_model.load_state_dict(torch.load(os.path.join(args.model_checkpoint, f'pytorch_model.bin'),
                                                 map_location=torch.device(args.device)))

    __run_visualization_code_samples(lmodel, tokenizer, final_probe_model, code_samples, parser,
                                     ids_to_labels_c, ids_to_labels_u, args)
    vectors_c = final_probe_model.vectors_c.detach().cpu().numpy().T
    vectors_u = final_probe_model.vectors_u.detach().cpu().numpy().T
    __run_visualization_vectors(vectors_c, vectors_u, ids_to_labels_c, ids_to_labels_u, args)


def __run_visualization_code_samples(lmodel, tokenizer, probe_model, code_samples,
                                     parser, ids_to_labels_c, ids_to_labels_u, args):
    lmodel.eval()
    probe_model.eval()

    for c, code in enumerate(code_samples):
        G, pre_code = code2ast(code, parser, args.lang)
        binary_ast = ast2binary(G)
        ds_current, cs_labels, _, us_labels = tree_to_distance(binary_ast, 0)
        tokens = get_tokens_ast(G, pre_code)

        # align tokens with subtokens
        to_convert, mapping = match_tokenized_to_untokenized_roberta(tokens, tokenizer)
        # generate inputs and masks
        inputs = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.cls_token] +
                                                               to_convert +
                                                               [tokenizer.sep_token])]).to(args.device)
        mask = torch.tensor([[1] * inputs.shape[1]]).to(args.device)

        # get align tensor
        j = 0
        indices = []
        for t in range(len(mapping)):
            indices += [j] * len(mapping[t])
            j += 1
        indices += [j] * (inputs.shape[1] - 1 - len(indices))
        alig = torch.tensor([indices]).to(args.device)

        # get embeddings from the lmodel
        emb = get_embeddings(inputs, mask, lmodel, args.layer, args.model_type)
        emb = align_function(emb, alig)

        # generating distance matrix
        d_pred, scores_c, scores_u = probe_model(emb.to(args.device))
        scores_c = torch.argmax(scores_c, dim=2)
        scores_u = torch.argmax(scores_u, dim=2)
        len_tokens = len(tokens)

        d_pred_current = d_pred[0, 0:len_tokens - 1].tolist()
        score_c_current = scores_c[0, 0:len_tokens - 1].tolist()
        score_u_current = scores_u[0, 0:len_tokens].tolist()

        scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
        scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

        ground_truth_tree = distance_to_tree(ds_current, cs_labels, us_labels, tokens)
        ground_truth_tree = extend_complex_nodes(add_unary(remove_empty_nodes(ground_truth_tree)))

        pred_tree = distance_to_tree(d_pred_current, scores_c_labels, scores_u_labels, tokens)
        pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))

        prec_score, recall_score, f1_score = get_precision_recall_f1(ground_truth_tree, pred_tree)
        # _, recall_block, _ = get_precision_recall_f1(ground_truth_tree, pred_tree, filter_non_terminal='block')

        logger.info(f'For code {c}, prec = {prec_score}, recall = {recall_score}, f1 = {f1_score}.')
        # logger.info(f'For code {c}, recall block = {recall_block}.')

        recall_score = get_recall_non_terminal(ground_truth_tree, pred_tree)
        for k, s in recall_score.items():
            logger.info(f'Non-terminal {k} | recall {s}')

        figure, axis = plt.subplots(2, figsize=(15, 15))
        nx.draw(nx.Graph(ground_truth_tree), labels=nx.get_node_attributes(ground_truth_tree, 'type'), with_labels=True,
                ax=axis[0])
        axis[0].set_title("True ast")
        nx.draw(nx.Graph(pred_tree), labels=nx.get_node_attributes(pred_tree, 'type'), with_labels=True,
                ax=axis[1])
        axis[1].set_title("Pred ast")
        plt.show()
        plt.savefig(f'fig_{c}_{args.lang}.png')

        labels_axis = [tokens[i] + '-' + tokens[i + 1] for i in range(0, len(tokens) - 1)]
        figure, axis = plt.subplots(2, figsize=(40, 40))
        axis[0].bar(labels_axis, ds_current)
        axis[0].set_title("True dist")
        for ix, label in enumerate(scores_c_labels):
            axis[0].annotate(label, (labels_axis[ix], ds_current[ix]))

        axis[1].bar(labels_axis, d_pred_current)
        axis[1].set_title("Pred dist")
        for ix, label in enumerate(cs_labels):
            axis[1].annotate(label, (labels_axis[ix], d_pred_current[ix]))
        plt.show()
        plt.savefig(f'fig_{c}_{args.lang}_syn_dis.png')


COLORS = {'java': 'r',
          'javascript': 'b',
          'go': 'g',
          'python': 'c',
          'c': 'm',
          'ruby': 'y',
          'csharp': 'k'
          }


def __run_visualization_vectors(vectors, ids_to_labels, type_labels, args, method='TSNE'):
    if method == 'TSNE':
        perplexity = 30.0
        if type_labels == 'u':
            perplexity = 5.0
        vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        v_2d = TSNE(n_components=2, learning_rate='auto', perplexity=perplexity,
                    init='random', random_state=args.seed).fit_transform(vectors)
    else:
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        v_2d = PCA(n_components=2).fit_transform(vectors_norm)

    figure, axis = plt.subplots(1, figsize=(20, 20))
    axis.set_title(f"Vectors {type_labels}")
    for ix, label in ids_to_labels.items():
        if SEPARATOR in label:
            continue
        l = label.split('--')[1]
        axis.scatter(v_2d[ix, 0], v_2d[ix, 1], color=COLORS[l], label=l)

    for ix, label in ids_to_labels.items():
        if SEPARATOR in label:
            continue
        axis.annotate(label, (v_2d[ix, 0], v_2d[ix, 1]))
    plt.show()
    plt.savefig(f'vectors_{type_labels}.png')


def run_visualization_multilingual(args):
    model_bin = os.path.join(args.output_path, f'pytorch_model.bin')
    # load the labels
    labels_file_path_c = os.path.join(args.output_path, 'global_labels_c.pkl')
    labels_file_path_u = os.path.join(args.output_path, 'global_labels_u.pkl')
    with open(labels_file_path_c, 'rb') as f:
        goblal_labels_c = pickle.load(f)
    goblal_labels_c = {y: x for x, y in goblal_labels_c.items()}
    with open(labels_file_path_u, 'rb') as f:
        goblal_labels_u = pickle.load(f)
    goblal_labels_u = {y: x for x, y in goblal_labels_u.items()}

    check_point = torch.load(model_bin, map_location=torch.device(args.device))

    vectors_c = check_point['vectors_c'].detach().cpu().numpy().T
    vectors_u = check_point['vectors_u'].detach().cpu().numpy().T

    __visualize_dataset_after_projection(args)

    # __perform_analog(vectors_c, goblal_labels_c)
    # __run_visualization_vectors(vectors_c, goblal_labels_c, 'c', args, method='TSNE')
    # __run_visualization_vectors(vectors_u, goblal_labels_u, 'u', args, method='TSNE')
    # __visualize_after_displacement(vectors_c, goblal_labels_c, args, target='csharp')


def __apply_kmeans(vectors, ids_to_labels, min_clusters, max_clusters, plot_name, args):
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]

    plt.figure(10)
    # Instantiate the clustering model and visualizer
    model = KMeans(random_state=args.seed)
    visualizer = KElbowVisualizer(model, k=(min_clusters, max_clusters))

    visualizer.fit(vectors_norm)  # Fit the data to the visualizer
    visualizer.show(outpath=plot_name)

    optimal = visualizer.elbow_value_
    kmeans_final = KMeans(n_clusters=optimal, random_state=args.seed).fit(vectors_norm)

    labels = kmeans_final.labels_
    for i in range(optimal):
        logger.info(f'Cluster {i}:')
        cont = 0
        for ix, l in enumerate(labels):
            if l == i and SEPARATOR not in ids_to_labels[ix]:
                logger.info(ids_to_labels[ix])
                cont += 1
            if cont == 10:
                break


def __perform_knn(vectors, ids_to_labels):
    vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    kd_tree = KDTree(vectors)
    l2id = {y: x for x, y in ids_to_labels.items()}
    for cand in ['for_statement--java',
                 'unary_expression--go',
                 'array--javascript', '<empty>--ruby']:
        id_cand = l2id[cand]
        _, i = kd_tree.query([vectors[id_cand]], k=10)
        i = i[0]
        for j, idx in enumerate(i):
            print(f'Neig k={j} for {cand} is {ids_to_labels[idx]}')


def __perform_analog(vectors, ids_to_labels):
    l2id = {y: x for x, y in ids_to_labels.items()}

    vectors_unit = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    kd_tree = KDTree(vectors_unit)

    source_lang = 'csharp'
    target_lan = 'c'

    list_nonterminals_source = [l.split('--')[0] for l in l2id.keys()
                                if SEPARATOR not in l and
                                l.endswith(f'--{source_lang}')]

    for nonterminal in list_nonterminals_source:
        # y_diff_langs = vectors[l2id[f'if_statement--{target_lan}']] - vectors[l2id[f'if_statement--{source_lang}']]
        y_source_mean = np.mean(np.stack([vectors[l2id[x]] for x in l2id if x.endswith(f'--{source_lang}')]), axis=0)
        y_target_mean = np.mean(np.stack([vectors[l2id[x]] for x in l2id if x.endswith(f'--{target_lan}')]), axis=0)
        y_diff_langs = y_target_mean - y_source_mean
        y = y_diff_langs + vectors[l2id[f'{nonterminal}--{source_lang}']]
        y = y / np.linalg.norm(y)
        _, i = kd_tree.query([y], k=3)
        i = i[0]
        for j, idx in enumerate(i):
            if ids_to_labels[idx].endswith(f'--{target_lan}'):
                print(f'Neig k={j} for analogy {nonterminal}--{source_lang} is {ids_to_labels[idx]}')


def __visualize_after_displacement(vectors, ids_to_labels, args, target='java'):
    new_vectors = []
    l2id = {y: x for x, y in ids_to_labels.items()}
    y_target_mean = np.mean(np.stack([vectors[l2id[x]] for x in l2id if x.endswith(f'--{target}')]),
                            axis=0)
    for i, _ in enumerate(ids_to_labels):
        l = ids_to_labels[i]
        if l.endswith(f'--{target}'):
            new_vectors.append(vectors[i])
        else:
            lang = l.split('--')[1]
            y_source_mean = np.mean(np.stack([vectors[l2id[x]] for x in l2id if x.endswith(f'--{lang}')]),
                                    axis=0)
            y_diff_langs = y_target_mean - y_source_mean
            y = y_diff_langs + vectors[i]
            new_vectors.append(y)

    __apply_kmeans(np.stack(new_vectors), ids_to_labels,
                   4, 80, f'elbow_displacement_{target}.png', args)

    vectors = np.stack(new_vectors) / np.linalg.norm(np.stack(new_vectors), axis=1)[:, np.newaxis]
    v_2d = TSNE(n_components=2, learning_rate='auto',
                init='random', random_state=args.seed).fit_transform(vectors)

    figure, axis = plt.subplots(1, figsize=(20, 20))
    axis.set_title(f"Vectors displaced to {target}")
    for ix, label in ids_to_labels.items():
        if SEPARATOR in label:
            continue
        l = label.split('--')[1]
        axis.scatter(v_2d[ix, 0], v_2d[ix, 1], color=COLORS[l], label=l)

    for ix, label in ids_to_labels.items():
        if SEPARATOR in label:
            continue
        nt = label.split('--')[0]
        axis.annotate(nt, (v_2d[ix, 0], v_2d[ix, 1]))
    plt.show()
    plt.savefig(f'vectors_displaced_{target}.png')


def __visualize_dataset_after_projection(args):
    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # load dictionaries
    labels_file_path_c = os.path.join(args.output_path, 'global_labels_c.pkl')
    labels_file_path_u = os.path.join(args.output_path, 'global_labels_u.pkl')
    with open(labels_file_path_c, 'rb') as f:
        labels_to_ids_c_global = pickle.load(f)
    ids_to_labels_c_global = {y: x for x, y in labels_to_ids_c_global.items()}
    with open(labels_file_path_u, 'rb') as f:
        labels_to_ids_u_global = pickle.load(f)
    ids_to_labels_u_global = {y: x for x, y in labels_to_ids_u_global.items()}

    # load models
    lmodel = get_lmodel(args)
    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c_global),
        number_labels_u=len(labels_to_ids_u_global)).to(args.device)
    logger.info('Loading best model.')
    checkpoint = torch.load(os.path.join(args.output_path, 'pytorch_model.bin'))
    probe_model.load_state_dict(checkpoint)

    # load datasets
    data_files = {}
    for lang in LANGUAGES:
        data_files[f'train_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'train.jsonl')
        data_files[f'valid_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'valid.jsonl')
        data_files[f'test_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'test.jsonl')

    data_sets = {x: load_dataset('json', data_files=y) for x, y in data_files.items()}
    data_sets = {x: y.map(lambda e: convert_sample_to_features(e['original_string'], parsers[x.split('_')[1]],
                                                               x.split('_')[1]))
                 for x, y in data_sets.items()}

    data_sets = {x: y.map(lambda e: convert_to_ids_multilingual(e['c'], 'c', labels_to_ids_c_global, x.split('_')[1]))
                 for x, y in data_sets.items()}
    data_sets = {x: y.map(lambda e: convert_to_ids_multilingual(e['u'], 'u', labels_to_ids_u_global, x.split('_')[1]))
                 for x, y in data_sets.items()}

    train_datasets = [y['train'] for x, y in data_sets.items() if 'train_' in x]
    valid_datasets = [y['train'] for x, y in data_sets.items() if 'valid_' in x]
    test_datasets = [y['train'] for x, y in data_sets.items() if 'test_' in x]

    # train_set = concatenate_datasets(train_datasets)
    # valid_set = concatenate_datasets(valid_datasets)
    test_set = concatenate_datasets(test_datasets)

    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_with_mask(batch, tokenizer,
                                                                             ids_to_labels_c_global,
                                                                             ids_to_labels_u_global),
                                 num_workers=8)

    lmodel.eval()
    probe_model.eval()
    projections = []
    all_cs = []
    for step, batch in enumerate(tqdm(test_dataloader,
                                      desc='[test batch]',
                                      bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):

        all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment, masks_c, masks_u = batch

        embds = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer,
                               args.model_type)
        embds = align_function(embds.to(args.device), alignment.to(args.device))

        projection = probe_model.apply_projection(embds.to(args.device))
        for i, len_tokens in enumerate(batch_len_tokens):
            len_tokens = len_tokens.item()
            # lxd
            projection_i = projection[i, 0:len_tokens - 1].detach().cpu().numpy()
            projections.append(projection_i)
            cs_current = cs[i, 0:len_tokens - 1].detach().cpu().numpy()
            all_cs.append(cs_current)
    projections = np.concatenate(projections, axis=0)
    all_cs = np.concatenate(all_cs, axis=0)

    mask = [('<empty>' not in ids_to_labels_c_global[c]
             and SEPARATOR not in ids_to_labels_c_global[c])
            for c in all_cs]
    projections = projections[mask]
    all_cs = all_cs[mask]

    values, counts = np.unique(all_cs, return_counts=True)
    indices = counts.argsort()[-100:][::-1]
    for a, b in zip(values[indices], counts[indices]):
        print(ids_to_labels_c_global[a], b)


    # projections = projections[np.isin(all_cs, considered_labels)]
    # all_cs = all_cs[np.isin(all_cs, considered_labels)]

    print(projections.shape)
    print(all_cs.shape)

    # idx = np.random.choice(np.arange(len(projections)), 1000000, replace=False)
    # projections = projections[idx]
    # all_cs = all_cs[idx]
    # langs = langs[idx]

    vectors = projections / np.linalg.norm(np.stack(projections), axis=1)[:, np.newaxis]
    try:
        v_2d = np.load(os.path.join(args.output_path, 'tsne_embeddings.np.npy'))
        all_cs = np.load(os.path.join(args.output_path, 'all_cs.np.npy'))
    except:
        logger.info('Cannot load embeddings, recomputing')
        tsne_obj = TSNE(n_components=2, n_jobs=12, random_state=args.seed, verbose=True).fit(vectors)
        v_2d = tsne_obj.transform(vectors)
        np.save(os.path.join(args.output_path, 'tsne_embeddings.np'), v_2d)
        np.save(os.path.join(args.output_path, 'all_cs.np'), all_cs)

    langs = np.array([ids_to_labels_c_global[c].split('--')[1] for c in all_cs])
    all_cs = np.array([ids_to_labels_c_global[c].split('--')[0] for c in all_cs])

    figure, axis = plt.subplots(1, figsize=(20, 20))
    markers = {
        'java': '.',
        'javascript': ',',
        'go': 'o',
        'python': 'v',
        'c': '^',
        'ruby': '<',
        'csharp': '>'
    }
    for g in np.unique(all_cs):
        i = np.where(all_cs == g)
        ls = langs[i]
        ms = [markers[j] for j in ls]
        if 'argument' in g:
            axis.scatter(v_2d[i, 0], v_2d[i, 1], color='black', alpha=0.3)
        elif 'call' in g:
            axis.scatter(v_2d[i, 0], v_2d[i, 1], color='red', alpha=0.3)
        elif 'return' in g:
            axis.scatter(v_2d[i, 0], v_2d[i, 1], color='tab:gray', alpha=0.3)
        elif 'method_invocation' in g:
            axis.scatter(v_2d[i, 0], v_2d[i, 1], color='c', alpha=0.3)
        elif 'block' in g:
            axis.scatter(v_2d[i, 0], v_2d[i, 1], color='m', alpha=0.3)
        elif 'binary' in g:
            axis.scatter(v_2d[i, 0], v_2d[i, 1], color='tab:olive', alpha=0.3)
        elif 'member_expression' == g:
            axis.scatter(v_2d[i, 0], v_2d[i, 1], color='g', alpha=0.3)
    # axis.legend()
    # axis.scatter(v_2d[:, 0], v_2d[:, 1])
    plt.show()
    plt.savefig(f'test_projected.png')

    fig = px.scatter(x=v_2d[:, 0], y=v_2d[:, 1], color=all_cs)
    fig.update_layout(showlegend=False)
    fig.write_html('first_figure.html', auto_open=False)
    # print(projections.shape)
    # print(all_cs.shape)
