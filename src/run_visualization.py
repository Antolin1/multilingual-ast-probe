import random
from data.utils import match_tokenized_to_untokenized_roberta
import torch
from probe.utils import get_embeddings, align_function
from data.code2ast import getTreeFromDistances, getUAS
import networkx as nx
import matplotlib.pyplot as plt
from data import convert_sample_to_features, PY_LANGUAGE, JS_LANGUAGE
from probe import TwoWordPSDProbe
import os
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, RobertaModel, T5EncoderModel
from run_probing import generate_baseline
from tree_sitter import Parser
import seaborn as sns


def run_visualization(args):
    data_files = {'train': os.path.join(args.dataset_name_or_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_name_or_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_name_or_path, 'test.jsonl')}

    test_set = load_dataset('json', data_files=data_files, split='test')
    test_set = test_set.filter(lambda e: len(e['code_tokens']) <= 100)
    test_set = test_set.shuffle(args.seed).select(range(4000))

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

    #taking small number of tokens
    test_set = test_set.filter(lambda e: len(e['code_tokens']) <= 30)
    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser))
    test_set = test_set.remove_columns(['original_string', 'code_tokens'])

    final_probe_model = TwoWordPSDProbe(args.rank, args.hidden, args.device)
    final_probe_model.load_state_dict(torch.load(os.path.join(args.output_path, f'pytorch_model.bin'),
                                                 map_location=torch.device(args.device)))

    __run_visualization(lmodel, tokenizer, final_probe_model, test_set, args.samples, args)


def __run_visualization(lmodel, tokenizer, probe_model, dataset, samples, args):
    lmodel.eval()
    probe_model.eval()

    indices = random.sample(list(range(len(dataset))), samples)
    for i in indices:
        sample = dataset[i]
        tokens = sample['tokens']
        real_dis = sample['matrix']

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
        pred_dis = outputs[0, 0:len(tokens), 0:len(tokens)].cpu().detach().numpy()

        # generating trees
        T_real = getTreeFromDistances(real_dis, tokens)
        T_pred = getTreeFromDistances(pred_dis, tokens)

        # plotting trees
        figure, axis = plt.subplots(2)
        nx.draw(T_real, labels=nx.get_node_attributes(T_real, 'type'), with_labels=True, ax=axis[0])
        axis[0].set_title("Real tree sample " + str(i))
        nx.draw(T_pred, labels=nx.get_node_attributes(T_pred, 'type'), with_labels=True, ax=axis[1])
        axis[1].set_title("Predicted tree sample " + str(i))
        plt.show()

        #plotting heatmaps
        figure, axis = plt.subplots(2)
        sns.heatmap(real_dis, ax=axis[0], xticklabels=tokens, yticklabels=tokens)
        sns.heatmap(pred_dis, ax=axis[1], xticklabels=tokens, yticklabels=tokens)
        plt.show()

        # UAS
        print('UAS in sample', i, ':', getUAS(T_real, T_pred))
