from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from .utils import get_embeddings, align_function
from src.data.code2ast import getTreeFromDistances, getUAS, getSpear


def report_uas(test_loader, probe_model, lmodel, args):
    lmodel.eval()
    probe_model.eval()
    uas_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='[valid UAS]'):
            all_inputs, all_attentions, dis, lens, alig = batch
            emb = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer, args.model_type)
            emb = align_function(emb.to(args.device), alig.to(args.device))
            outputs = probe_model(emb.to(args.device))
            for j, dis_pred in enumerate(outputs):
                l = lens[j].item()
                dis_real = dis[j].cpu().detach().numpy()[0:l, 0:l]
                dis_pred = dis_pred[0:l, 0:l].cpu().detach().numpy()
                # the distances are already aligned
                # getTreeFromDistances needs the tokens to label the nodes
                code_tokens_invented = ['a']*l
                # get trees
                T_real = getTreeFromDistances(dis_real, code_tokens_invented)
                T_pred = getTreeFromDistances(dis_pred, code_tokens_invented)
                uas_scores.append(getUAS(T_real, T_pred))
    return np.mean(uas_scores)


def report_spear(test_loader, probe_model, lmodel, args):
    lmodel.eval()
    probe_model.eval()
    lengths_to_spearmanrs = defaultdict(list)
    for batch in tqdm(test_loader, desc='[valid SPEAR]'):
        all_inputs, all_attentions, dis, lens, alig = batch
        emb = get_embeddings(all_inputs.to(args.device), all_attentions.to(args.device), lmodel, args.layer, args.model_type)
        emb = align_function(emb.to(args.device), alig.to(args.device))
        outputs = probe_model(emb.to(args.device))
        for j, dis_pred in enumerate(outputs):
            l = lens[j].item()
            dis_real = dis[j].cpu().detach().numpy()[0:l, 0:l]
            dis_pred = dis_pred[0:l, 0:l].cpu().detach().numpy()
            spear = getSpear(dis_real, dis_pred)
            lengths_to_spearmanrs[l].extend(spear)
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length])
                                     for length in lengths_to_spearmanrs}
    return mean_spearman_for_each_length, np.mean([mean_spearman_for_each_length[x]
                                                   for x in mean_spearman_for_each_length])
