import os
import argparse
import logging
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tree_sitter import Parser
from tqdm import tqdm

from data import convert_sample_to_features, PY_LANGUAGE, collator_fn
from probe import TwoWordPSDProbe, L1DistanceLoss


logger = logging.getLogger(__name__)


def get_embeddings(all_inputs, all_attentions, model, layer):
    model.eval()
    with torch.no_grad():
        embs = model(input_ids=all_inputs, attention_mask=all_attentions)[2][layer][:, 1:, :]
    return embs


def align_function(embs, align):
    seq = []
    for j, emb in enumerate(embs):
        seq.append(scatter_mean(emb, align[j], dim=0))
    # remove the last token since it corresponds to <\s> or padding to much the lens
    return pad_sequence(seq, batch_first=True)[:, :-1, :]


def run_probing_train(args: argparse.Namespace):
    logger.info('-' * 100)
    logger.info('Running probing training.')
    logger.info('-' * 100)

    logger.info('Loading dataset from local file.')
    data_files = {'train': os.path.join(args.dataset_name_or_path, 'train.jsonl'),
                  'valid': os.path.join(args.dataset_name_or_path, 'valid.jsonl'),
                  'test': os.path.join(args.dataset_name_or_path, 'test.jsonl')}
    train_set = load_dataset('json', data_files=data_files, split='train')
    valid_set = load_dataset('json', data_files=data_files, split='valid')
    test_set = load_dataset('json', data_files=data_files, split='test')

    # @todo: load from checkpoint
    logger.info('Loading model and tokenizer.')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    lmodel = AutoModel.from_pretrained(args.tokenizer_name_or_path, output_hidden_states=True)

    # @todo: create a function to encapsulate everything
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    test_set = test_set.map(lambda e: convert_sample_to_features(e['original_string'], parser))
    test_set = test_set.remove_columns('original_string')
    test_dataloader = DataLoader(dataset=test_set,
                                 batch_size=64,
                                 shuffle=False,
                                 collate_fn=lambda batch: collator_fn(batch, tokenizer))

    probe_model = TwoWordPSDProbe(128, 768, args.device)
    criterion = L1DistanceLoss(args.device)

    optimizer = torch.optim.Adam(probe_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0)

    probe_model.train()
    best_eval_loss = 0.0
    patience_count = 0
    for epoch in tqdm(range(args.epochs), desc='[training epoch loop]'):
        training_loss = 0.0
        for batch in tqdm(test_dataloader,
                          desc='[training batch]',
                          bar_format='{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}'):
            all_inputs, all_attentions, dis, lens, alig = batch
            emb = get_embeddings(all_inputs, all_attentions, lmodel, args.layer)
            emb = align_function(emb, alig)

            outputs = probe_model(emb)

            loss, count = criterion(outputs, dis, lens)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            training_loss += loss.item()

        training_loss = training_loss / len(test_dataloader)
        eval_loss = run_probing_eval(test_dataloader, probe_model, criterion, lmodel, args.layer)
        scheduler.step(eval_loss)
        tqdm.write('[epoch {}] train loss: {}, validation loss: {}'.format(epoch,
                                                                    training_loss,
                                                                    eval_loss))

        logger.info('-' * 100)
        logger.info('Saving model checkpoint')
        logger.info('-' * 100)
        if eval_loss > best_eval_loss:
            output_path = os.path.join(args.model_chkpt_path, f'pytorch_model.bin')
            torch.save(probe_model.state_dict(), output_path)
            logger.info(f'Probe model saved: {output_path}')
            patience_count = 0
            best_eval_loss = eval_loss
        else:
            patience_count += 1
        if patience_count == args.patience:
            break


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
        for batch in tqdm(test_dataloader, desc='[dev batch]'):
            all_inputs, all_attentions, dis, lens, alig = batch
            emb = get_embeddings(all_inputs, all_attentions, lmodel, layer)
            emb = align_function(emb, alig)
            outputs = probe_model(emb)
            loss, count = criterion(outputs, dis, lens)
            eval_loss += loss.item()
    return eval_loss / len(test_dataloader)
