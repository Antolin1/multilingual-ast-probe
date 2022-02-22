import random
from data.utils import match_tokenized_to_untokenized_roberta
from data.code2ast import code2ast, get_depths_tokens_ast, \
    get_matrix_tokens_ast, enrich_ast_with_deps, get_dependency_tree, \
    get_matrix_and_tokens_dep, get_spear
import torch
import numpy as np
from probe.utils import get_embeddings, align_function
from data.code2ast import get_tree_from_distances, get_uas
import networkx as nx
import matplotlib.pyplot as plt
from data import convert_sample_to_features, PY_LANGUAGE, JS_LANGUAGE
from probe import ParserProbe
import os
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, RobertaModel, T5EncoderModel
from run_probing import generate_baseline
from tree_sitter import Parser
import seaborn as sns
import glob
from scipy.stats import spearmanr

#todo: add the dictionaries for the classification
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


    final_probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)


    final_probe_model.load_state_dict(torch.load(os.path.join(args.output_path, f'pytorch_model.bin'),
                                                 map_location=torch.device(args.device)))

    __run_visualization(lmodel, tokenizer, final_probe_model, code_samples, parser, args)


def __run_visualization(lmodel, tokenizer, probe_model, code_samples, parser, args):
    lmodel.eval()
    probe_model.eval()

    for c, code in enumerate(code_samples):
        G, pre_code = code2ast(code, parser)
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
        len_tokens = len(tokens)

        d_pred_current = d_pred[0, 0:len_tokens - 1].tolist()
        score_c_current = scores_c[0, 0:len_tokens - 1].tolist()
        score_u_current = scores_u[0, 0:len_tokens].tolist()

        scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
        scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

        ground_truth_tree = distance_to_tree(ds_current, cs_labels, us_labels, [str(i) for i in range(len_tokens)])
        ground_truth_tree = extend_complex_nodes(add_unary(remove_empty_nodes(ground_truth_tree)))

        pred_tree = distance_to_tree(d_pred_current, scores_c_labels, scores_u_labels,
                                     [str(i) for i in range(len_tokens)])
        pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))

        _, _, f1_score = get_precision_recall_f1(ground_truth_tree, pred_tree)


