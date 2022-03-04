import glob
import argparse
import os
import pickle

import pandas as pd
from plotnine import ggplot, aes, geom_line


def main():
    parser = argparse.ArgumentParser(description='Script for generating the plots of the paper')
    parser.add_argument('--run_dir', default='./runs', help='Path of the run logs')
    args = parser.parse_args()

    data = {'model': [], 'lang': [], 'layer': [], 'rank': [], 'precision': [], 'recall': [], 'f1': []}
    for file in glob.glob(args.run_dir + "/*/metrics.log"):
        parent = os.path.dirname(file).split('/')[-1]
        model, lang, layer, rank = parent.split('_')
        with open(file, 'rb') as f:
            results = pickle.load(f)
        data['model'].append(model)
        data['lang'].append(lang)
        data['layer'].append(int(layer))
        data['rank'].append(int(rank))
        data['precision'].append(results['test_precision'])
        data['recall'].append(results['test_recall'])
        data['f1'].append(results['test_f1'])

    df = pd.DataFrame(data)
    print(df[df['lang'] == 'python'])
    myPlot = (
            ggplot(df[df['lang'] == 'python'])  # What data to use
            + aes(x="layer", y="f1", color='model')  # What variable to use
            + geom_line()  # Geometric object to use for drawing
    )
    myPlot.save("myplot.png", dpi=600)


if __name__ == '__main__':
    main()
