import glob
import argparse
import os

import torch
from scipy.linalg import subspace_angles
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Script for generating the plots of the paper')
    parser.add_argument('--run_dir', default='./runs_monolingual', help='Path of the run logs')
    args = parser.parse_args()
    dic_models = {}
    for file in glob.glob(args.run_dir + "/*/metrics.log"):
        lang = os.path.dirname(file).split('/')[-1]
        checkpoint = torch.load(os.path.join(os.path.dirname(file), 'pytorch_model.bin'),
                                map_location=torch.device('cpu'))
        proj = checkpoint['proj'].cpu().detach().numpy()
        dic_models[lang] = proj

    langs = ['python', 'javascript', 'go', 'ruby', 'php', 'java']
    for l1 in langs:
        for l2 in langs:
            if l1 == l2:
                continue
            p1 = dic_models[l1]
            p2 = dic_models[l2]
            print(f'{l1} vs {l2}, {np.mean(subspace_angles(p1, p2))}, {np.rad2deg(np.mean(subspace_angles(p1, p2)))}')
        print('--' * 10)


if __name__ == '__main__':
    main()
