import os


def main():
    models = ['microsoft/codebert-base', 'microsoft/graphcodebert-base'
                                         'Salesforce/codet5-base', 'huggingface/CodeBERTa-small-v1',
              'roberta', 'microsoft/codebert-base']
    folders = ['codebert', 'graphcodebert', 'codet5', 'codeberta', 'roberta',
               'codebert-baseline']
    model_types = ['roberta', 'roberta', 't5', 'roberta', 'roberta', 'roberta']

    for lang in ['python', 'javascript']:
        for model, folder, model_type in zip(models, folders, model_types):
            for probe_type in ['ast_probe', 'depth_probe']:
                if model == 'huggingface/CodeBERTa-small-v1':
                    layers = list(range(1, 7))
                else:
                    layers = list(range(1, 13))
                for layer in layers:
                    folder = '_'.join([folder, lang, probe_type, str(layer), '128'])
                    os.system(f"python3 src/main.py --do_train --run_name {folder} "
                              f"--pretrained_model_name_or_path {model} "
                              f"--model_type {model_type} --lang {lang} "
                              f"--layer {layer} --rank 128  --type_probe {probe_type}")


if __name__ == '__main__':
    main()
