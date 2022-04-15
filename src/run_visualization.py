from data.utils import match_tokenized_to_untokenized_roberta
from data.code2ast import code2ast, get_tokens_ast
from data.binary_tree import ast2binary, tree_to_distance, distance_to_tree, \
    extend_complex_nodes, add_unary, remove_empty_nodes, get_precision_recall_f1, \
    get_recall_non_terminal, SEPARATOR
import torch
from probe.utils import get_embeddings, align_function
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from data import PY_LANGUAGE, JS_LANGUAGE, GO_LANGUAGE
from probe import ParserProbe
import os
from transformers import AutoModel, AutoTokenizer, RobertaModel, T5EncoderModel
from run_probing import generate_baseline
from tree_sitter import Parser
import glob
import logging
import pickle
import numpy as np
from yellowbrick.cluster import KElbowVisualizer
from scipy.spatial import KDTree

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

    __perform_knn(vectors_c, goblal_labels_c)
    __run_visualization_vectors(vectors_c, goblal_labels_c, 'c', args, method='TSNE')
    __run_visualization_vectors(vectors_u, goblal_labels_u, 'u', args, method='TSNE')
    # __apply_kmeans(vectors_c, goblal_labels_c, 4, 80, 'elbow_c.png', args)
    # __apply_kmeans(vectors_u, goblal_labels_u, 4, 30, 'elbow_u.png', args)


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
                 'array--javascript']:
        id_cand = l2id[cand]
        _, i = kd_tree.query([vectors[id_cand]], k=10)
        i = i[0]
        for j, idx in enumerate(i):
            print(f'Neig k={j} for {cand} is {ids_to_labels[idx]}')
