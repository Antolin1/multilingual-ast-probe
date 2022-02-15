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
from probe import TwoWordPSDProbe, OneWordPSDProbe
import os
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, RobertaModel, T5EncoderModel
from run_probing import generate_baseline
from tree_sitter import Parser
import seaborn as sns
import glob
from scipy.stats import spearmanr

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

    if args.type_probe != 'depth_probe':
        final_probe_model = TwoWordPSDProbe(args.rank, args.hidden, args.device)
    else:
        final_probe_model = OneWordPSDProbe(args.rank, args.hidden, args.device)

    final_probe_model.load_state_dict(torch.load(os.path.join(args.output_path, f'pytorch_model.bin'),
                                                 map_location=torch.device(args.device)))

    __run_visualization(lmodel, tokenizer, final_probe_model, code_samples, parser, args)


def __run_visualization(lmodel, tokenizer, probe_model, code_samples, parser, args):
    lmodel.eval()
    probe_model.eval()

    for c, code in enumerate(code_samples):
        G, pre_code = code2ast(code, parser, args.lang)
        if args.type_probe == 'depth_probe':
            depths, code_tokens = get_depths_tokens_ast(G, pre_code)
            tokens = code_tokens
            real_dis = depths
        elif args.type_probe == 'ast_probe':
            matrix, code_tokens = get_matrix_tokens_ast(G, pre_code)
            tokens = code_tokens
            real_dis = matrix
        elif args.type_probe == 'dep_probe':
            enrich_ast_with_deps(G)
            T = get_dependency_tree(G)
            matrix, code_tokens = get_matrix_and_tokens_dep(T, pre_code)
            tokens = code_tokens
            real_dis = matrix

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
        outputs = probe_model(emb.to(args.device))
        if args.type_probe != 'depth_probe':
            pred_dis = outputs[0, 0:len(tokens), 0:len(tokens)].cpu().detach().numpy()
        else:
            pred_dis = outputs[0, 0:len(tokens)].cpu().detach().numpy()

        # generating trees
        #T_real = get_tree_from_distances(real_dis, tokens)
        #T_pred = get_tree_from_distances(pred_dis, tokens)

        # plotting trees
        #figure, axis = plt.subplots(2)
        #nx.draw(T_real, labels=nx.get_node_attributes(T_real, 'type'), with_labels=True, ax=axis[0])
        #axis[0].set_title("Real tree sample " + str(i))
        #nx.draw(T_pred, labels=nx.get_node_attributes(T_pred, 'type'), with_labels=True, ax=axis[1])
        #axis[1].set_title("Predicted tree sample " + str(i))
        #plt.show()

        #plotting heatmaps
        if args.type_probe != 'depth_probe':
            figure, axis = plt.subplots(2)
            sns.heatmap(real_dis, ax=axis[0], xticklabels=tokens, yticklabels=tokens)
            sns.heatmap(pred_dis, ax=axis[1], xticklabels=tokens, yticklabels=tokens)
            plt.show()
        else:
            x = np.arange(len(tokens))
            fig, ax = plt.subplots()
            ax.scatter(x, pred_dis, color='blue')
            ax.scatter(x, real_dis, color='0.8')
            for i, txt in enumerate(tokens):
                ax.text(x[i], max(pred_dis[i], real_dis[i]) + 0.2, txt, rotation=90)
                #ax.annotate(txt, (x[i], pred_dis[i]))
            plt.show()
            plt.savefig(f'code_samples/plot_{c}.png')

        if args.type_probe != 'depth_probe':
            spear = np.mean(get_spear(real_dis, pred_dis))
            print(f'Spear corr for {c}: {spear}')
        else:
            spear = spearmanr(real_dis, pred_dis).correlation
            print(f'Spear corr for {c}: {spear}')

        # UAS
        #print('UAS in sample', i, ':', get_uas(T_real, T_pred))
