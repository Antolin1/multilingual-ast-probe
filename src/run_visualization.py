
import random
from data.utils import match_tokenized_to_untokenized_roberta
import torch
from probe.utils import get_embeddings, align_function
from data.code2ast import getTreeFromDistances
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
        inputs = torch.tensor([tokenizer.convert_tokens_to_ids([tokenizer.cls_token] +
                                                 to_convert +
                                                 [tokenizer.sep_token])])

        mask = torch.tensor([[1] * len(inputs)])
        j = 0
        indices = []
        for t in range(len(mapping)):
            indices += [j] * len(mapping[t])
            j += 1
        indices += [j] * (len(inputs) - 1 - len(indices))
        alig = torch.tensor([indices])

        #get embeddings from the lmodel
        emb = get_embeddings(inputs, mask, lmodel, args.layer)
        emb = align_function(emb, alig)

        outputs = probe_model(emb.to(args.device))
        pred_dis = outputs[0,0:len(tokens),0:len(tokens)].detach().numpy()

        #visualize predicted tree vs real one
        T_real = getTreeFromDistances(real_dis, tokens)
        T_pred = getTreeFromDistances(pred_dis, tokens)

        figure, axis = plt.subplots(1, 2)

        nx.draw(T_real, labels=nx.get_node_attributes(T_real, 'type'), with_labels=True, ax=axis[0][0])
        axis[0][0].set_title("Real Tree")

        nx.draw(T_real, labels=nx.get_node_attributes(T_real, 'type'), with_labels=True, ax=axis[0][1])
        axis[0][1].set_title("Predicted Tree")

        plt.show()
