
from ..run_probing import get_embeddings, align_function
from ..data.code2ast import getTreeFromDistances, getUAS, getSpear
import numpy as np
from collections import defaultdict

def report_UAS(test_loader, probe_model, lmodel, args):
    lmodel.eval()
    probe_model.eval()
    uas_scores = []
    for batch in test_loader:
        all_inputs, all_attentions, dis, lens, alig = batch
        emb = get_embeddings(all_inputs, all_attentions, lmodel, args.layer)
        emb = align_function(emb, alig)
        outputs = probe_model(emb)
        for j, dis_pred in enumerate(outputs):
            l = lens[j].item()
            dis_real = dis[j].detach().numpy()[0:l, 0:l]
            dis_pred = dis_pred[0:l, 0:l].detach().numpy()
            #the distances are already aligned
            #getTreeFromDistances needs the tokens to label the nodes
            code_tokens_invented = ['a']*l
            #get trees
            T_real = getTreeFromDistances(dis_real, code_tokens_invented)
            T_pred = getTreeFromDistances(dis_pred, code_tokens_invented)
            uas_scores.append(getUAS(T_real, T_pred))
    return np.mean(uas_scores)

def report_spear(test_loader, probe_model, lmodel, args):
    lmodel.eval()
    probe_model.eval()
    uas_scores = []
    lengths_to_spearmanrs = defaultdict(list)
    for batch in test_loader:
        all_inputs, all_attentions, dis, lens, alig = batch
        emb = get_embeddings(all_inputs, all_attentions, lmodel, args.layer)
        emb = align_function(emb, alig)
        outputs = probe_model(emb)
        for j, dis_pred in enumerate(outputs):
            l = lens[j].item()
            dis_real = dis[j].detach().numpy()[0:l, 0:l]
            dis_pred = dis_pred[0:l, 0:l].detach().numpy()
            spear = getSpear(dis_real, dis_pred)
            lengths_to_spearmanrs[l].extend(spear)
    mean_spearman_for_each_length = {length: np.mean(lengths_to_spearmanrs[length])
                                     for length in lengths_to_spearmanrs}
    return mean_spearman_for_each_length, np.mean([mean_spearman_for_each_length[x]
                                                   for x in mean_spearman_for_each_length])