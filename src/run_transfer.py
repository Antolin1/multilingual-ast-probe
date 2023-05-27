import argparse
import os
import pickle

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from probe import ParserProbe
from data import PARSER_OBJECT_BY_NAME, convert_sample_to_features, collator_fn
from data.data_loading import convert_to_ids
from run_probing import get_lmodel, run_probing_eval_f1


def load_model(path):
    loaded_model = torch.load(path,
                              map_location=torch.device('cpu'))
    vectors_c = loaded_model['vectors_c'].cpu().detach().numpy().T
    vectors_u = loaded_model['vectors_u'].cpu().detach().numpy().T
    proj = loaded_model['proj'].cpu().detach().numpy()
    return vectors_c, vectors_u, proj


def build_model(args):
    _, _, proj_source = load_model(args.source_model)
    vectors_c_target, vectors_u_target, proj_target = load_model(args.target_model)

    real_vectors_c_target = np.matmul(vectors_c_target, proj_target.T)
    real_vectors_u_target = np.matmul(vectors_u_target, proj_target.T)
    new_vectors_c_target = np.matmul(real_vectors_c_target, proj_source)
    new_vectors_u_target = np.matmul(real_vectors_u_target, proj_source)

    probe_model = ParserProbe(
        probe_rank=proj_target.shape[1],
        hidden_dim=proj_target.shape[0],
        number_labels_c=new_vectors_c_target.shape[0],
        number_labels_u=new_vectors_u_target.shape[0])
    probe_model.vectors_c.data = torch.from_numpy(new_vectors_c_target.T)
    probe_model.vectors_u.data = torch.from_numpy(new_vectors_u_target.T)
    probe_model.proj.data = torch.from_numpy(proj_source)
    probe_model.to(args.device)
    probe_model.eval()
    return probe_model


DATASET = 'dataset'


def run_probe(args, probe_model):
    # select the parser
    parser = PARSER_OBJECT_BY_NAME[args.target_lang]

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    data_files = {'test': os.path.join(DATASET, args.target_lang, 'test.jsonl')}
    test_set = load_dataset('json', data_files=data_files, split='test')

    # get d and c for each sample
    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.target_lang))

    # get class labels-ids mapping for c and u
    labels_file_path = os.path.join(DATASET, args.target_lang, 'labels.pkl')

    with open(labels_file_path, 'rb') as f:
        data = pickle.load(f)
        labels_to_ids_c = data['labels_to_ids_c']
        ids_to_labels_c = data['ids_to_labels_c']
        labels_to_ids_u = data['labels_to_ids_u']
        ids_to_labels_u = data['ids_to_labels_u']

    test_set = test_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    test_set = test_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))

    lmodel = get_lmodel(args)
    lmodel.eval()
    metrics = {'test_precision': None, 'test_recall': None, 'test_f1': None}

    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                 num_workers=8)

    eval_precision, eval_recall, eval_f1_score, recall_nonterminals = run_probing_eval_f1(test_dataloader,
                                                                                          probe_model,
                                                                                          lmodel,
                                                                                          ids_to_labels_c,
                                                                                          ids_to_labels_u, args,
                                                                                          compute_recall_nonterminals=True)
    metrics['test_precision'] = round(eval_precision, 4)
    metrics['test_recall'] = round(eval_recall, 4)
    metrics['test_f1'] = round(eval_f1_score, 4)
    metrics['recall_nonterminals'] = recall_nonterminals
    print(f'test precision: {round(eval_precision, 4)} | test recall: {round(eval_recall, 4)} '
          f'| test F1 score: {round(eval_f1_score, 4)}')

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, '_'.join([args.source_lang, args.target_lang])) + '.log', 'wb') as f:
        pickle.dump(metrics, f)


def main(args):
    probe_model = build_model(args)
    run_probe(args, probe_model)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='microsoft/codebert-base')
    parser.add_argument('--layer', type=int, default=5)
    parser.add_argument('--source_model', type=str, default='runs/codebert_ruby_5_128/pytorch_model.bin')
    parser.add_argument('--target_model', type=str, default='runs/codebert_php_5_128/pytorch_model.bin')
    parser.add_argument('--source_lang', type=str, default='ruby')
    parser.add_argument('--target_lang', type=str, default='php')
    parser.add_argument('--model_type', type=str, default='roberta')
    parser.add_argument('--out_dir', help='output directory', default='./transfer')
    args = parser.parse_args()
    args.dispatch_model_weights = False
    args.run_name = ''
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.batch_size = 32
    args.do_train_all_languages, args.do_test_all_languages = False, False
    main(args)
