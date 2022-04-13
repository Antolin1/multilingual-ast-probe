import os
import argparse
import logging
import pickle
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, RobertaModel, T5EncoderModel
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm

from data import convert_sample_to_features, collator_fn, \
    PY_PARSER, GO_PARSER, JS_PARSER, PHP_PARSER, JAVA_PARSER, RUBY_PARSER, LANGUAGES, CSHARP_PARSER, C_PARSER
from probe import ParserProbe, ParserLoss, get_embeddings, align_function
from data.data_loading import get_non_terminals_labels, convert_to_ids, convert_to_ids_multilingual
from data.binary_tree import distance_to_tree, remove_empty_nodes, \
    extend_complex_nodes, get_precision_recall_f1, add_unary, get_recall_non_terminal

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

parsers = {
        'python': PY_PARSER,
        'javascript': JS_PARSER,
        'go': GO_PARSER,
        'php': PHP_PARSER,
        'ruby': RUBY_PARSER,
        'java': JAVA_PARSER,
        'csharp': CSHARP_PARSER,
        'c': C_PARSER
    }


def generate_baseline(model):
    config = model.config
    baseline = RobertaModel(config)
    baseline.embeddings = model.embeddings
    return baseline


def get_lmodel(args):
    logger.info('Loading models.')
    if args.model_type == 't5':
        lmodel = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
    else:
        lmodel = AutoModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
        if '-baseline' in args.run_name:
            lmodel = generate_baseline(lmodel)
    lmodel = lmodel.to(args.device)
    return lmodel


def run_train_general(probe_model, lmodel, train_dataloader, valid_dataloader, metrics, pretrained, args):
    if pretrained:
        optimizer = torch.optim.Adam([probe_model.vectors_c, probe_model.vectors_u], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(probe_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)
    criterion = ParserLoss(loss='rank', pretrained=pretrained)

    probe_model.train()
    lmodel.eval()
    best_eval_loss = float('inf')
    patience_count = 0
    for epoch in range(1, args.epochs + 1):
        training_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader,
                                          desc='[training batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment = batch

            embds = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer,
                                   args.model_type)
            embds = align_function(embds.to(args.device), alignment.to(args.device))

            d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
            loss = criterion(
                d_pred=d_pred.to(args.device),
                scores_c=scores_c.to(args.device),
                scores_u=scores_u.to(args.device),
                d_real=ds.to(args.device),
                c_real=cs.to(args.device),
                u_real=us.to(args.device),
                length_batch=batch_len_tokens.to(args.device))

            if not pretrained:
                reg = args.orthogonal_reg * (
                        torch.norm(torch.matmul(torch.transpose(probe_model.proj, 0, 1), probe_model.proj)
                                   - torch.eye(args.rank).to(args.device)) ** 2)
                loss += reg
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()

        training_loss = training_loss / len(train_dataloader)
        eval_loss, _, _, _ = run_probing_eval(valid_dataloader, probe_model, lmodel, criterion, args)
        scheduler.step(eval_loss)
        logger.info(f'[epoch {epoch}] train loss: {round(training_loss, 4)}, validation loss: {round(eval_loss, 4)}')
        metrics['training_loss'].append(round(training_loss, 4))
        metrics['validation_loss'].append(round(eval_loss, 4))

        if eval_loss < best_eval_loss:
            logger.info('-' * 100)
            logger.info('Saving model checkpoint')
            logger.info('-' * 100)
            output_path = os.path.join(args.output_path, f'pytorch_model.bin')
            torch.save(probe_model.state_dict(), output_path)
            logger.info(f'Probe model saved: {output_path}')
            patience_count = 0
            best_eval_loss = eval_loss
        else:
            patience_count += 1
        if patience_count == args.patience:
            logger.info('Stopping training loop (out of patience).')
            break


def run_probing_train(args: argparse.Namespace):
    logger.info('-' * 100)
    logger.info('Running probing training.')
    logger.info('-' * 100)

    # select the parser
    parser = parsers[args.lang]

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    logger.info('Loading dataset from local file.')
    data_files = {'train': os.path.join(args.dataset_name_or_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_name_or_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_name_or_path, 'test.jsonl')}

    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')

    # get d and c for each sample
    train_set = train_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang))
    valid_set = valid_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang))
    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang))

    # get class labels-ids mapping for c and u
    labels_file_path = os.path.join(args.dataset_name_or_path, 'labels.pkl')
    if not os.path.exists(labels_file_path):
        # convert each non-terminal labels to its id
        labels_to_ids_c = get_non_terminals_labels(train_set['c'], valid_set['c'], test_set['c'])
        ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
        labels_to_ids_u = get_non_terminals_labels(train_set['u'], valid_set['u'], test_set['u'])
        ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
        with open(labels_file_path, 'wb') as f:
            pickle.dump({
                'labels_to_ids_c': labels_to_ids_c, 'ids_to_labels_c': ids_to_labels_c,
                'labels_to_ids_u': labels_to_ids_u, 'ids_to_labels_u': ids_to_labels_u
            }, f)
    else:
        with open(labels_file_path, 'rb') as f:
            data = pickle.load(f)
            labels_to_ids_c = data['labels_to_ids_c']
            ids_to_labels_c = data['ids_to_labels_c']
            labels_to_ids_u = data['labels_to_ids_u']
            ids_to_labels_u = data['ids_to_labels_u']

    train_set = train_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    valid_set = valid_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    test_set = test_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))

    train_set = train_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))
    valid_set = valid_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))
    test_set = test_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))

    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=valid_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)

    lmodel = get_lmodel(args)
    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)
    metrics = {'training_loss': [], 'validation_loss': [], 'test_precision': None, 'test_recall': None, 'test_f1': None}

    run_train_general(probe_model, lmodel, train_dataloader, valid_dataloader, metrics, pretrained=False, args=args)

    logger.info('Loading test set.')
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                 num_workers=8)

    logger.info('Loading best model.')
    checkpoint = torch.load(os.path.join(args.output_path, 'pytorch_model.bin'))
    probe_model.load_state_dict(checkpoint)

    logger.info('Evaluating probing on test set.')
    eval_precision, eval_recall, eval_f1_score = run_probing_eval_f1(test_dataloader, probe_model, lmodel,
                                                                     ids_to_labels_c, ids_to_labels_u, args)
    metrics['test_precision'] = round(eval_precision, 4)
    metrics['test_recall'] = round(eval_recall, 4)
    metrics['test_f1'] = round(eval_f1_score, 4)
    logger.info(f'test precision: {round(eval_precision, 4)} | test recall: {round(eval_recall, 4)} '
                f'| test F1 score: {round(eval_f1_score, 4)}')

    logger.info('-' * 100)
    logger.info('Saving metrics.')
    with open(os.path.join(args.output_path, 'metrics.log'), 'wb') as f:
        pickle.dump(metrics, f)


def run_probing_eval(test_dataloader, probe_model, lmodel, criterion, args):
    probe_model.eval()
    eval_loss = 0.0
    total_hits_c = 0
    total_c = 0
    total_hits_u = 0
    total_u = 0
    total_hits_d = 0
    total_d = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          desc='[test batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment = batch
            ds = ds.to(args.device)
            cs = cs.to(args.device)
            us = us.to(args.device)

            embds = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer,
                                   args.model_type)
            embds = align_function(embds.to(args.device), alignment.to(args.device))

            d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
            loss = criterion(
                d_pred=d_pred.to(args.device),
                scores_c=scores_c.to(args.device),
                scores_u=scores_u.to(args.device),
                d_real=ds.to(args.device),
                c_real=cs.to(args.device),
                u_real=us.to(args.device),
                length_batch=batch_len_tokens.to(args.device))
            eval_loss += loss.item()

            # compute the classes c and u
            # scores_c /= probe_model.vectors_c.norm(p=2, dim=0)
            # scores_u /= probe_model.vectors_u.norm(p=2, dim=0)
            scores_c = torch.argmax(scores_c, dim=2)
            scores_u = torch.argmax(scores_u, dim=2)

            batch_len_tokens = batch_len_tokens.to(args.device)
            lens_d = (batch_len_tokens - 1).to(args.device)
            max_len_d = torch.max(lens_d)
            mask_c = torch.arange(max_len_d, device=args.device)[None, :] < lens_d[:, None]
            mask_u = torch.arange(max_len_d + 1, device=args.device)[None, :] < batch_len_tokens[:, None]

            scores_c = torch.masked_select(scores_c, mask_c)
            scores_u = torch.masked_select(scores_u, mask_u)
            cs = torch.masked_select(cs, mask_c)
            us = torch.masked_select(us, mask_u)

            hits_c = (scores_c == cs).sum().item()
            hits_u = (scores_u == us).sum().item()

            total_hits_u += hits_u
            total_hits_c += hits_c
            total_c += mask_c.sum().item()
            total_u += mask_u.sum().item()

            hits_d, total_d_current = compute_hits_d(d_pred, ds, mask_c)
            total_hits_d += hits_d
            total_d += total_d_current

            # compute the accuracy on d

    acc_u = float(total_hits_u) / float(total_u)
    acc_c = float(total_hits_c) / float(total_c)
    acc_d = float(total_hits_d) / float(total_d)

    return (eval_loss / len(test_dataloader)), acc_c, acc_u, acc_d


def compute_hits_d(input, target, mask):
    diff = input[:, :, None] - input[:, None, :]
    target_diff = ((target[:, :, None] - target[:, None, :]) > 0).float()
    mask = mask[:, :, None] * mask[:, None, :] * target_diff
    loss = torch.relu(target_diff - diff)
    hits = (((loss * mask) == 0) * mask).sum().item()
    return hits, mask.sum().item()


def run_probing_eval_f1(test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args):
    # todo: filter categories using the language
    probe_model.eval()
    precisions = [] if (not args.do_train_all_languages) and (not args.do_hold_one_out_training) else defaultdict(list)
    recalls = [] if (not args.do_train_all_languages) and (not args.do_hold_one_out_training) else defaultdict(list)
    f1_scores = [] if (not args.do_train_all_languages) and (not args.do_hold_one_out_training) else defaultdict(list)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          desc='[test batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment = batch

            embds = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer,
                                   args.model_type)
            embds = align_function(embds.to(args.device), alignment.to(args.device))

            d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
            scores_c = torch.argmax(scores_c, dim=2)
            scores_u = torch.argmax(scores_u, dim=2)

            for i, len_tokens in enumerate(batch_len_tokens):
                len_tokens = len_tokens.item()
                d_pred_current = d_pred[i, 0:len_tokens - 1].tolist()
                score_c_current = scores_c[i, 0:len_tokens - 1].tolist()
                score_u_current = scores_u[i, 0:len_tokens].tolist()
                ds_current = ds[i, 0:len_tokens - 1].tolist()
                cs_current = cs[i, 0:len_tokens - 1].tolist()
                us_current = us[i, 0:len_tokens].tolist()

                cs_labels = [ids_to_labels_c[c] for c in cs_current]
                us_labels = [ids_to_labels_u[c] for c in us_current]
                scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
                scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

                ground_truth_tree = distance_to_tree(ds_current, cs_labels, us_labels,
                                                     [str(i) for i in range(len_tokens)])
                ground_truth_tree = extend_complex_nodes(add_unary(remove_empty_nodes(ground_truth_tree)))

                pred_tree = distance_to_tree(d_pred_current, scores_c_labels, scores_u_labels,
                                             [str(i) for i in range(len_tokens)])
                pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))

                p, r, f1_score = get_precision_recall_f1(ground_truth_tree, pred_tree)
                if args.do_train_all_languages or args.do_hold_one_out_training:
                    lang = us_labels[0].split('--')[1]
                    f1_scores[lang].append(f1_score)
                    precisions[lang].append(p)
                    recalls[lang].append(r)
                else:
                    f1_scores.append(f1_score)
                    precisions.append(p)
                    recalls.append(r)
    if args.do_train_all_languages or args.do_hold_one_out_training:
        return ({x: np.mean(y) for x, y in precisions.items()},
                {x: np.mean(y) for x, y in recalls.items()},
                {x: np.mean(y) for x, y in f1_scores.items()})
    else:
        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)


def run_probing_eval_recall_non_terminal(test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args):
    probe_model.eval()
    recall_scores = defaultdict(list)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,
                                          desc='[test batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment = batch

            embds = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer,
                                   args.model_type)
            embds = align_function(embds.to(args.device), alignment.to(args.device))

            d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
            scores_c = torch.argmax(scores_c, dim=2)
            scores_u = torch.argmax(scores_u, dim=2)

            for i, len_tokens in enumerate(batch_len_tokens):
                len_tokens = len_tokens.item()
                d_pred_current = d_pred[i, 0:len_tokens - 1].tolist()
                score_c_current = scores_c[i, 0:len_tokens - 1].tolist()
                score_u_current = scores_u[i, 0:len_tokens].tolist()
                ds_current = ds[i, 0:len_tokens - 1].tolist()
                cs_current = cs[i, 0:len_tokens - 1].tolist()
                us_current = us[i, 0:len_tokens].tolist()

                cs_labels = [ids_to_labels_c[c] for c in cs_current]
                us_labels = [ids_to_labels_u[c] for c in us_current]
                scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
                scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

                ground_truth_tree = distance_to_tree(ds_current, cs_labels, us_labels,
                                                     [str(i) for i in range(len_tokens)])
                ground_truth_tree = extend_complex_nodes(add_unary(remove_empty_nodes(ground_truth_tree)))

                pred_tree = distance_to_tree(d_pred_current, scores_c_labels, scores_u_labels,
                                             [str(i) for i in range(len_tokens)])
                pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))

                recall_score = get_recall_non_terminal(ground_truth_tree, pred_tree)
                for k, s in recall_score.items():
                    recall_scores[k].append(s)

    return {k: np.mean(v) for k, v in recall_scores.items()}


def run_probing_test(args):
    logger.info('-' * 100)
    logger.info('Running probing test.')
    logger.info('-' * 100)

    # select the parser
    parser = parsers[args.lang]

    logger.info('Loading tokenizer')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    logger.info('Loading dataset from local file.')
    data_files = {'test': os.path.join(args.dataset_name_or_path, 'test.jsonl')}

    # get class labels-ids mapping for c and u
    labels_file_path = os.path.join(args.dataset_name_or_path, 'labels.pkl')
    with open(labels_file_path, 'rb') as f:
        data = pickle.load(f)
        labels_to_ids_c = data['labels_to_ids_c']
        ids_to_labels_c = data['ids_to_labels_c']
        labels_to_ids_u = data['labels_to_ids_u']
        ids_to_labels_u = data['ids_to_labels_u']

    test_set = load_dataset('json', data_files=data_files, split='test')

    # get d and c for each sample
    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang))
    test_set = test_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    test_set = test_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))

    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                 num_workers=8)

    lmodel = get_lmodel(args)
    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)
    criterion = ParserLoss(loss='rank')

    if args.model_checkpoint:
        logger.info('Restoring model checkpoint.')
        checkpoint = torch.load(os.path.join(args.model_checkpoint, 'pytorch_model.bin'))
        probe_model.load_state_dict(checkpoint)

    probe_model.eval()
    lmodel.eval()
    eval_loss, acc_c, acc_u, acc_d = run_probing_eval(test_dataloader, probe_model, lmodel, criterion, args)
    logger.info(f'test loss: {eval_loss} | acc_c: {acc_c} | acc_u: {acc_u} | acc_d: {acc_d}')

    recall_dict = run_probing_eval_recall_non_terminal(test_dataloader, probe_model, lmodel,
                                                       ids_to_labels_c, ids_to_labels_u, args)

    for v, k in sorted(((v, k) for k, v in recall_dict.items()), reverse=True):
        logger.info(f'Non-terminal {k} | recall {v}')


def run_probing_from_given_projection(args):
    logger.info('-' * 100)
    logger.info('Running probing training.')
    logger.info('-' * 100)

    # select the parser
    parser = parsers[args.lang]

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    logger.info('Loading dataset from local file.')
    data_files = {'train': os.path.join(args.dataset_name_or_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_name_or_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_name_or_path, 'test.jsonl')}

    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')

    # get d and c for each sample
    train_set = train_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang))
    valid_set = valid_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang))
    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.lang))

    # get class labels-ids mapping for c and u
    labels_file_path = os.path.join(args.dataset_name_or_path, 'labels.pkl')
    if not os.path.exists(labels_file_path):
        # convert each non-terminal labels to its id
        labels_to_ids_c = get_non_terminals_labels(train_set['c'], valid_set['c'], test_set['c'])
        ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
        labels_to_ids_u = get_non_terminals_labels(train_set['u'], valid_set['u'], test_set['u'])
        ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
        with open(labels_file_path, 'wb') as f:
            pickle.dump({
                'labels_to_ids_c': labels_to_ids_c, 'ids_to_labels_c': ids_to_labels_c,
                'labels_to_ids_u': labels_to_ids_u, 'ids_to_labels_u': ids_to_labels_u
            }, f)
    else:
        with open(labels_file_path, 'rb') as f:
            data = pickle.load(f)
            labels_to_ids_c = data['labels_to_ids_c']
            ids_to_labels_c = data['ids_to_labels_c']
            labels_to_ids_u = data['labels_to_ids_u']
            ids_to_labels_u = data['ids_to_labels_u']

    train_set = train_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    valid_set = valid_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))
    test_set = test_set.map(lambda e: convert_to_ids(e['c'], 'c', labels_to_ids_c))

    train_set = train_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))
    valid_set = valid_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))
    test_set = test_set.map(lambda e: convert_to_ids(e['u'], 'u', labels_to_ids_u))

    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=valid_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)

    lmodel = get_lmodel(args)

    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c),
        number_labels_u=len(labels_to_ids_u)).to(args.device)

    logger.info('Loading model checkpoint.')
    checkpoint = torch.load(os.path.join(args.model_source_checkpoint, 'pytorch_model.bin'))
    probe_model.proj = torch.nn.Parameter(data=checkpoint['proj'])

    metrics = {'training_loss': [], 'validation_loss': [], 'test_precision': None, 'test_recall': None, 'test_f1': None}
    run_train_general(probe_model, lmodel, train_dataloader, valid_dataloader, metrics, pretrained=True, args=args)

    logger.info('Loading test set.')
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                 num_workers=8)

    logger.info('Loading best model.')
    checkpoint = torch.load(os.path.join(args.output_path, 'pytorch_model.bin'))
    probe_model.load_state_dict(checkpoint)

    logger.info('Evaluating probing on test set.')
    eval_precision, eval_recall, eval_f1_score = run_probing_eval_f1(test_dataloader, probe_model, lmodel,
                                                                     ids_to_labels_c, ids_to_labels_u, args)
    metrics['test_precision'] = round(eval_precision, 4)
    metrics['test_recall'] = round(eval_recall, 4)
    metrics['test_f1'] = round(eval_f1_score, 4)
    logger.info(f'test precision: {round(eval_precision, 4)} | test recall: {round(eval_recall, 4)} '
                f'| test F1 score: {round(eval_f1_score, 4)}')

    logger.info('-' * 100)
    logger.info('Saving metrics.')
    with open(os.path.join(args.output_path, 'metrics.log'), 'wb') as f:
        pickle.dump(metrics, f)


def run_probing_all_languages(args):
    logger.info('-' * 100)
    logger.info('Running probing on all languages.')
    logger.info('-' * 100)

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    logger.info('Loading dataset from local file.')
    data_files = {}
    for lang in LANGUAGES:
        data_files[f'train_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'train.jsonl')
        data_files[f'valid_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'valid.jsonl')
        data_files[f'test_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'test.jsonl')

    data_sets = {x: load_dataset('json', data_files=data_files, split=x) for x, y in data_files.items()}
    data_sets = {x: y.map(lambda e: convert_sample_to_features(e['original_string'], parsers[x.split('_')[1]],
                                                               x.split('_')[1]))
                 for x, y in data_sets.items()}

    labels_c = []
    labels_u = []
    for lang in LANGUAGES:
        labels_file_path = os.path.join(args.dataset_name_or_path, lang, 'labels.pkl')
        with open(labels_file_path, 'rb') as f:
            data = pickle.load(f)
            labels_to_ids_c = data['labels_to_ids_c']
            labels_to_ids_u = data['labels_to_ids_u']
            labels_c += [x + '--' + lang for x in labels_to_ids_c.keys()]
            labels_u += [x + '--' + lang for x in labels_to_ids_u.keys()]

    # conversion to global labels
    labels_to_ids_c_global = {x: y for y, x in enumerate(labels_c)}
    ids_to_labels_c_global = {y: x for x, y in labels_to_ids_c_global.items()}
    labels_to_ids_u_global = {x: y for y, x in enumerate(labels_u)}
    ids_to_labels_u_global = {y: x for x, y in labels_to_ids_u_global.items()}

    #save labels
    with open(os.path.join(args.output_path, 'global_labels_c.pkl'), 'wb') as f:
        pickle.dump(labels_to_ids_c_global, f)
    with open(os.path.join(args.output_path, 'global_labels_u.pkl'), 'wb') as f:
        pickle.dump(labels_to_ids_u_global, f)

    data_sets = {x: y.map(lambda e: convert_to_ids_multilingual(e['c'], 'c', labels_to_ids_c_global, x.split('_')[1]))
                 for x, y in data_sets.items()}
    data_sets = {x: y.map(lambda e: convert_to_ids_multilingual(e['u'], 'u', labels_to_ids_u_global, x.split('_')[1]))
                 for x, y in data_sets.items()}

    train_datasets = [y for x, y in data_sets.items() if 'train_' in x]
    valid_datasets = [y for x, y in data_sets.items() if 'valid_' in x]
    test_datasets = [y for x, y in data_sets.items() if 'test_' in x]

    train_set = concatenate_datasets(train_datasets)
    valid_set = concatenate_datasets(valid_datasets)
    test_set = concatenate_datasets(test_datasets)

    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=valid_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)

    lmodel = get_lmodel(args)
    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c_global),
        number_labels_u=len(labels_to_ids_u_global)).to(args.device)

    metrics = {'training_loss': [], 'validation_loss': [], 'test_precision': None, 'test_recall': None, 'test_f1': None}

    run_train_general(probe_model, lmodel, train_dataloader, valid_dataloader, metrics, pretrained=False, args=args)

    logger.info('Loading test set.')
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                 num_workers=8)

    logger.info('Loading best model.')
    checkpoint = torch.load(os.path.join(args.output_path, 'pytorch_model.bin'))
    probe_model.load_state_dict(checkpoint)

    logger.info('Evaluating probing on test set.')
    eval_precision, eval_recall, eval_f1_score = run_probing_eval_f1(test_dataloader, probe_model, lmodel,
                                                                     ids_to_labels_c_global, ids_to_labels_u_global,
                                                                     args)
    for lang in eval_precision.keys():
        metrics[f'test_precision_{lang}'] = round(eval_precision[lang], 4)
        metrics[f'test_recall_{lang}'] = round(eval_recall[lang], 4)
        metrics[f'test_f1_{lang}'] = round(eval_f1_score[lang], 4)
        logger.info(
            f'test precision_{lang}: {round(eval_precision[lang], 4)} | test recall_{lang}: {round(eval_recall[lang], 4)} '
            f'| test F1 score_{lang}: {round(eval_f1_score[lang], 4)}')

    logger.info('-' * 100)
    logger.info('Saving metrics.')
    with open(os.path.join(args.output_path, 'metrics.log'), 'wb') as f:
        pickle.dump(metrics, f)


def run_hold_one_out_training(args):
    logger.info('-' * 100)
    logger.info('Running hold one out training.')
    logger.info('-' * 100)

    logger.info('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    logger.info('Loading dataset from local file.')
    data_files = {}
    for lang in LANGUAGES:
        if args.lang != lang:
            data_files[f'train_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'train.jsonl')
            data_files[f'valid_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'valid.jsonl')
            data_files[f'test_{lang}'] = os.path.join(args.dataset_name_or_path, lang, 'test.jsonl')

    data_sets = {x: load_dataset('json', data_files=data_files, split=x) for x, y in data_files.items()}
    data_sets = {x: y.map(lambda e: convert_sample_to_features(e['original_string'], parsers[x.split('_')[1]],
                                                               x.split('_')[1]))
                 for x, y in data_sets.items()}

    labels_c = []
    labels_u = []
    for lang in LANGUAGES:
        if args.lang != lang:
            labels_file_path = os.path.join(args.dataset_name_or_path, lang, 'labels.pkl')
            with open(labels_file_path, 'rb') as f:
                data = pickle.load(f)
                labels_to_ids_c = data['labels_to_ids_c']
                labels_to_ids_u = data['labels_to_ids_u']
                labels_c += [x + '--' + lang for x in labels_to_ids_c.keys()]
                labels_u += [x + '--' + lang for x in labels_to_ids_u.keys()]

    # conversion to global labels
    labels_to_ids_c_global = {x: y for y, x in enumerate(labels_c)}
    ids_to_labels_c_global = {y: x for x, y in labels_to_ids_c_global.items()}
    labels_to_ids_u_global = {x: y for y, x in enumerate(labels_u)}
    ids_to_labels_u_global = {y: x for x, y in labels_to_ids_u_global.items()}

    # save labels
    with open(os.path.join(args.output_path, 'global_labels_c.pkl'), 'wb') as f:
        pickle.dump(labels_to_ids_c_global, f)
    with open(os.path.join(args.output_path, 'global_labels_u.pkl'), 'wb') as f:
        pickle.dump(labels_to_ids_u_global, f)

    data_sets = {x: y.map(lambda e: convert_to_ids_multilingual(e['c'], 'c', labels_to_ids_c_global, x.split('_')[1]))
                 for x, y in data_sets.items()}
    data_sets = {x: y.map(lambda e: convert_to_ids_multilingual(e['u'], 'u', labels_to_ids_u_global, x.split('_')[1]))
                 for x, y in data_sets.items()}

    train_datasets = [y for x, y in data_sets.items() if 'train_' in x]
    valid_datasets = [y for x, y in data_sets.items() if 'valid_' in x]
    test_datasets = [y for x, y in data_sets.items() if 'test_' in x]

    train_set = concatenate_datasets(train_datasets)
    valid_set = concatenate_datasets(valid_datasets)
    test_set = concatenate_datasets(test_datasets)

    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)
    valid_dataloader = DataLoader(dataset=valid_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                  num_workers=8)

    lmodel = get_lmodel(args)
    probe_model = ParserProbe(
        probe_rank=args.rank,
        hidden_dim=args.hidden,
        number_labels_c=len(labels_to_ids_c_global),
        number_labels_u=len(labels_to_ids_u_global)).to(args.device)

    metrics = {'training_loss': [], 'validation_loss': [], 'test_precision': None, 'test_recall': None, 'test_f1': None}

    run_train_general(probe_model, lmodel, train_dataloader, valid_dataloader, metrics, pretrained=False, args=args)

    logger.info('Loading test set.')
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer),
                                 num_workers=8)

    logger.info('Loading best model.')
    checkpoint = torch.load(os.path.join(args.output_path, 'pytorch_model.bin'))
    probe_model.load_state_dict(checkpoint)

    logger.info('Evaluating probing on test set.')
    eval_precision, eval_recall, eval_f1_score = run_probing_eval_f1(test_dataloader, probe_model, lmodel,
                                                                     ids_to_labels_c_global, ids_to_labels_u_global,
                                                                     args)
    for lang in eval_precision.keys():
        metrics[f'test_precision_{lang}'] = round(eval_precision[lang], 4)
        metrics[f'test_recall_{lang}'] = round(eval_recall[lang], 4)
        metrics[f'test_f1_{lang}'] = round(eval_f1_score[lang], 4)
        logger.info(
            f'test precision_{lang}: {round(eval_precision[lang], 4)} | test recall_{lang}: {round(eval_recall[lang], 4)} '
            f'| test F1 score_{lang}: {round(eval_f1_score[lang], 4)}')

    logger.info('-' * 100)
    logger.info('Saving metrics.')
    with open(os.path.join(args.output_path, 'metrics.log'), 'wb') as f:
        pickle.dump(metrics, f)