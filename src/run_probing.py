import os
import pickle
import argparse
import logging
from typing import Union

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, RobertaModel, T5EncoderModel
from datasets import load_dataset
from tree_sitter import Parser
from tqdm import tqdm

from data import convert_sample_to_features, PY_LANGUAGE, collator_fn, JS_LANGUAGE
from probe import OneWordPSDProbe, TwoWordPSDProbe, L1DistanceLoss, get_embeddings,\
    align_function, report_uas, report_spear, L1DepthLoss
from data.utils import match_tokenized_to_untokenized_roberta

logger = logging.getLogger(__name__)


def generate_baseline(model):
    config = model.config
    baseline = RobertaModel(config)
    baseline.embeddings = model.embeddings
    return baseline

def filter_strategy(code_tokens, tokenizer, max_tokens):
    to_convert, _ = match_tokenized_to_untokenized_roberta(code_tokens, tokenizer)
    return len(code_tokens) <= max_tokens and (len(to_convert) + 2 <= 512)

def run_probing_train(args: argparse.Namespace):
    logger.info('-' * 100)
    logger.info('Running probing training.')
    logger.info('-' * 100)

    # select the parser
    parser = Parser()
    if args.lang == 'python':
        parser.set_language(PY_LANGUAGE)
    elif args.lang == 'javascript':
        parser.set_language(JS_LANGUAGE)

    logger.info('Loading dataset from local file.')
    data_files = {'train': os.path.join(args.dataset_name_or_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_name_or_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_name_or_path, 'test.jsonl')}

    logger.info('Loading tokenizer')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')


    train_set = train_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.type_probe, args.lang))
    train_set = train_set.filter(lambda e: filter_strategy(e['tokens'], tokenizer, args.max_tokens))
    train_set = train_set.shuffle(args.seed).select(range(min(20000, len(train_set))))
    train_set = train_set.remove_columns(['original_string', 'code_tokens'])

    valid_set = valid_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.type_probe, args.lang))
    valid_set = valid_set.filter(lambda e: filter_strategy(e['tokens'], tokenizer, args.max_tokens))
    valid_set = valid_set.shuffle(args.seed).select(range(min(2000, len(valid_set))))
    valid_set = valid_set.remove_columns(['original_string', 'code_tokens'])


    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser, args.type_probe, args.lang))
    test_set = test_set.filter(lambda e: filter_strategy(e['tokens'], tokenizer, args.max_tokens))
    test_set = test_set.shuffle(args.seed).select(range(min(4000, len(test_set))))
    test_set = test_set.remove_columns(['original_string', 'code_tokens'])

    # @todo: load lmodel and tokenizer from checkpoint
    # @todo: model_type in ProgramArguments
    logger.info('Loading model')

    if args.model_type == 't5':
        lmodel = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
        lmodel = lmodel.to(args.device)
    else:
        lmodel = AutoModel.from_pretrained(args.pretrained_model_name_or_path, output_hidden_states=True)
        if '-baseline' in args.run_name:
            lmodel = generate_baseline(lmodel)
        lmodel = lmodel.to(args.device)




    train_dataloader = DataLoader(dataset=train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer, args.type_probe),
                                  num_workers=10)
    valid_dataloader = DataLoader(dataset=valid_set,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  collate_fn=lambda batch: collator_fn(batch, tokenizer, args.type_probe),
                                  num_workers=10)
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer, args.type_probe),
                                 num_workers=10)

    if args.type_probe == 'depth_probe':
        probe_model = OneWordPSDProbe(args.rank, args.hidden, args.device)
        criterion = L1DistanceLoss(args.device)
    else:
        probe_model = TwoWordPSDProbe(args.rank, args.hidden, args.device)
        criterion = L1DepthLoss(args.device)

    optimizer = torch.optim.Adam(probe_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)

    probe_model.train()
    lmodel.eval()
    best_eval_loss = float('inf')
    metrics = {'training_loss': [], 'validation_loss': [], 'validation_spear': [], 'test_spear': []}
    patience_count = 0
    for epoch in tqdm(range(args.epochs), desc='[training epoch loop]'):
        training_loss = 0.0
        for step, batch in enumerate(tqdm(train_dataloader,
                                          desc='[training batch]',
                                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}')):
            probe_model.train()
            all_inputs, all_attentions, dis, lens, alig = batch

            emb = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer, args.model_type)
            emb = align_function(emb.to(args.device), alig.to(args.device))

            outputs = probe_model(emb.to(args.device))
            loss, count = criterion(outputs, dis.to(args.device), lens.to(args.device))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()

        training_loss = training_loss / len(train_dataloader)
        eval_loss = run_probing_eval(valid_dataloader, probe_model, criterion, lmodel, args.layer, args)

        _, eval_spear = report_spear(valid_dataloader, probe_model, lmodel, args)
        scheduler.step(eval_loss)
        logger.info(f'[epoch {epoch}] train loss: {round(training_loss, 4)}, '
                    f'validation loss: {round(eval_loss, 4)}, validation SPEAR: {round(eval_spear, 4)}')
        metrics['training_loss'].append(round(training_loss, 4))
        metrics['validation_loss'].append(round(eval_loss, 4))
        metrics['validation_spear'].append(round(eval_spear, 4))

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

    logger.info('-' * 100)
    logger.info('Loading best probe model.')
    final_probe_model = TwoWordPSDProbe(args.rank, args.hidden, args.device)
    final_probe_model.load_state_dict(torch.load(os.path.join(args.output_path, f'pytorch_model.bin')))
    _, test_spear = report_spear(test_dataloader, probe_model, lmodel, args)
    logger.info(f'Test SPEAR: {test_spear}')
    metrics['test_spear'].append(round(test_spear, 4))

    logger.info('-' * 100)
    logger.info('Saving metrics.')
    with open(os.path.join(args.output_path, 'metrics.log'), 'wb') as f:
        pickle.dump(metrics, f)


def run_probing_eval(
        test_dataloader: Union[DataLoader, None] = None,
        probe_model: Union[TwoWordPSDProbe, None] = None,
        criterion: Union[L1DistanceLoss, None] = None,
        lmodel: Union[AutoModel, None] = None,
        layer: Union[int, None] = None,
        args: Union[argparse.Namespace, None] = None):
    probe_model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='[valid batch]'):
            all_inputs, all_attentions, dis, lens, alig = batch
            emb = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, layer, args.model_type)
            emb = align_function(emb.to(args.device), alig.to(args.device))
            outputs = probe_model(emb.to(args.device))
            loss, count = criterion(outputs, dis.to(args.device), lens.to(args.device))
            eval_loss += loss.item()
    return eval_loss / len(test_dataloader)
