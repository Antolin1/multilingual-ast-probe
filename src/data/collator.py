#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:16:08 2022

@author: Jose Antonio
"""
import numpy as np
import torch


from .utils import match_tokenized_to_untokenized_roberta


def collator_fn(batch, tokenizer):
    tokens = [b['tokens'] for b in batch]
    matrices = [np.array(b['matrix']) for b in batch]
    

    
    #token lens
    len_tokens = [len(t) for t in tokens]
    max_len_tokens = np.max(len_tokens)
    len_tokens = torch.tensor(len_tokens)

    #pad matrices
    padded_matices = []
    for m in matrices:
        if m.shape[0] < max_len_tokens:
            ones = -np.ones((m.shape[0], max_len_tokens - m.shape[0]))
            padded_dis = np.concatenate((m, ones), axis=1)
            ones = -np.ones((max_len_tokens - m.shape[0], max_len_tokens))
            padded_dis = np.concatenate((padded_dis, ones), 0)
            padded_matices.append(padded_dis)
        else:
            padded_matices.append(m)
    padded_matices = torch.tensor(np.array(padded_matices))

    
    #generate inputs and attention masks
    all_inputs = []
    all_attentions = []
    all_mappings = []
    for untokenized_sent in tokens:
        to_convert, mapping = match_tokenized_to_untokenized_roberta(untokenized_sent, tokenizer)
        inputs = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + 
                                                               to_convert + 
                                                               [tokenizer.sep_token])
       
        mask = [1]*len(inputs)
        all_inputs.append(inputs)
        all_attentions.append(mask)
        all_mappings.append({x:[l + 1 for l in y] for x,y in mapping.items()})
    #padding
    max_len_subtokens = np.max([len(m) for m in all_attentions])
    all_inputs = torch.tensor([inputs +tokenizer.convert_tokens_to_ids([tokenizer.pad_token]*(max_len_subtokens - len(inputs)))
                  for inputs in all_inputs])
    all_attentions = torch.tensor([mask +([0]*(max_len_subtokens - len(mask)))
                  for mask in all_attentions])

    #generate alig
    #after generate embeddings, remove first token
    alig = []
    for mapping in all_mappings:
        j = 0
        indices = []
        for i in range(len(mapping)):
            indices += [j]*len(mapping[i])
            j += 1
        indices += [j]*(max_len_subtokens - 1 - len(indices))
        alig.append(indices)
    alig = torch.tensor(alig)

    
    return (all_inputs, all_attentions, padded_matices, len_tokens, alig)
