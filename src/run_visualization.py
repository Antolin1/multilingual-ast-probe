
import random
from data.utils import match_tokenized_to_untokenized_roberta
import torch
from probe.utils import get_embeddings, align_function
from data.code2ast import getTreeFromDistances, getUAS
import networkx as nx
import matplotlib.pyplot as plt


def run_visualization(lmodel, tokenizer, probe_model, dataset, samples, args):
    lmodel.eval()
    probe_model.eval()

    indices = random.sample(list(range(len(dataset))), samples)
    for i in indices:
        sample = dataset[i]
        tokens = sample['tokens']
        real_dis = sample['matrix']

        #align tokens with subtokens
        to_convert, mapping = match_tokenized_to_untokenized_roberta(tokens, tokenizer)
        #generate inputs and masks
        inputs = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.cls_token] +
                                                 to_convert +
                                                 [tokenizer.sep_token])]).to(args.device)
        mask = torch.tensor([[1] * inputs.shape[1]]).to(args.device)

        #get align tensor
        j = 0
        indices = []
        for t in range(len(mapping)):
            indices += [j] * len(mapping[t])
            j += 1
        indices += [j] * (inputs.shape[1] - 1 - len(indices))
        alig = torch.tensor([indices]).to(args.device)

        #get embeddings from the lmodel
        emb = get_embeddings(inputs, mask, lmodel, args.layer)
        emb = align_function(emb, alig)

        #generating distance matrix
        outputs = probe_model(emb.to(args.device))
        pred_dis = outputs[0,0:len(tokens),0:len(tokens)].cpu().detach().numpy()

        #generating trees
        T_real = getTreeFromDistances(real_dis, tokens)
        T_pred = getTreeFromDistances(pred_dis, tokens)

        #plotting trees
        figure, axis = plt.subplots(1, 2)
        nx.draw(T_real, labels=nx.get_node_attributes(T_real, 'type'), with_labels=True, ax=axis[0][0])
        axis[0][0].set_title("Real tree sample " + str(i))
        nx.draw(T_pred, labels=nx.get_node_attributes(T_pred, 'type'), with_labels=True, ax=axis[0][1])
        axis[0][1].set_title("Predicted tree sample" + str(i))
        plt.show()

        #UAS
        print('UAS in sample', i,':',getUAS(T_real, T_pred))
