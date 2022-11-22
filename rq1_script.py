import argparse
import os

from src.data.data_loading import LANGUAGES

models = ['microsoft/codebert-base',
          'microsoft/graphcodebert-base',
          'Salesforce/codet5-base',
          'huggingface/CodeBERTa-small-v1',
          'roberta-base',
          'microsoft/codebert-base',
          'bert-base-uncased',
          'distilbert-base-uncased',
          'distilroberta-base',
          'microsoft/unixcoder-base',
          'microsoft/unixcoder-base-nine',
          'microsoft/unixcoder-base-unimodal']
folders = ['codebert',
           'graphcodebert',
           'codet5',
           'codeberta',
           'roberta',
           'codebert-baseline',
           'bert',
           'distilbert',
           'distilroberta',
           'unixcoder-base',
           'unixcoder-base-nine',
           'unixcoder-base-unimodal']
model_types = ['roberta',
               'roberta',
               't5',
               'roberta',
               'roberta',
               'roberta',
               'bert',
               'distilbert',
               'roberta',
               'roberta',
               'roberta',
               'roberta']

assert len(model_types) == len(folders)
assert len(model_types) == len(models)


def get_model_folder_type(split):
    if split == 'all':
        return models, folders, model_types
    elif split == '1':
        return models[0:4], folders[0:4], model_types[0:4]
    elif split == '2':
        return models[4:8], folders[4:8], model_types[4:8]
    elif split == '3':
        return models[8:10], folders[8:10], model_types[8:10]
    elif split == '4':
        return models[10:], folders[10:], model_types[10:]


def main(args):
    models_split, folders_split, model_types_split = get_model_folder_type(args.split)
    assert len(models_split) == len(folders_split)
    assert len(folders_split) == len(model_types_split)
    for model, folder, model_type in zip(models_split, folders_split, model_types_split):
        for lang in LANGUAGES:
            if (model == 'huggingface/CodeBERTa-small-v1' or model == 'distilroberta-base'
                    or model == 'distilbert-base-uncased'):
                layers = list(range(1, 7))
            else:
                layers = list(range(1, 13))
            for layer in layers:
                run_name = '_'.join([folder, lang, str(layer), '128'])
                if not os.path.exists(os.path.join('./runs', run_name, 'metrics.log')):
                    os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_device} python src/main.py "
                              f"--do_train "
                              f"--run_name {run_name} "
                              f"--pretrained_model_name_or_path {model} "
                              f"--model_type {model_type} "
                              f"--lang {lang} "
                              f"--layer {layer} "
                              f"--rank 128")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script the first research question')
    parser.add_argument('--split', help='split of models.', choices=['1', '2', '3', '4', 'all'],
                        default='all')
    parser.add_argument('--cuda_device', help='Cuda device', default=0)
    args = parser.parse_args()
    main(args)
