
import argparse
import json
import os

huggingface_names = {
    'CodeBERT': 'microsoft/codebert-base',
    'GraphCodeBERT': 'microsoft/graphcodebert-base',
    'CodeT5': 'Salesforce/codet5-base',
    'CodeBERTa': 'huggingface/CodeBERTa-small-v1',
    'RoBERTa': 'roberta-base',
    'BERT': 'bert-base-uncased',
    'DistilBERT': 'distilbert-base-uncased',
    'DistilRoBERTa': 'distilroberta-base',
    'CodeBERTrand': 'microsoft/codebert-base',
    'UniXcoder': 'microsoft/unixcoder-base',
    'UniXcoder-9': 'microsoft/unixcoder-base-nine'
}

model_types = {
    'CodeBERT': 'roberta',
    'GraphCodeBERT': 'roberta',
    'CodeT5': 't5',
    'CodeBERTa': 'roberta',
    'RoBERTa': 'roberta',
    'BERT': 'bert',
    'DistilBERT': 'distilbert',
    'DistilRoBERTa': 'roberta',
    'CodeBERTrand': 'roberta',
    'UniXcoder': 'roberta',
    'UniXcoder-9': 'roberta'
}


def main(args):
    with open(args.out_best_layer_per_model_rq1) as json_file:
        model_layer = json.load(json_file)
    for m_l in model_layer:
        model = m_l['model']
        layer = m_l['layer']
        hfn = huggingface_names[model]
        model_type = model_types[model]

        run_name = '_'.join(['multilingual', model + ('-baseline' if model == 'CodeBERTrand' else '')])
        if not os.path.exists(os.path.join('./runs', run_name, 'metrics.log')):
            os.system(f"CUDA_VISIBLE_DEVICES={args.cuda_device} python src/main.py "
                      f"--do_train_all_languages "
                      f"--run_name {run_name} "
                      f"--pretrained_model_name_or_path {hfn} "
                      f"--model_type {model_type} "
                      f"--layer {layer} "
                      f"--rank 128 --epochs 20")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script the second research question')
    parser.add_argument('--cuda_device', help='Cuda device', default=0)
    parser.add_argument('--out_best_layer_per_model_rq1', default='best_layer_per_model.json',
                        help='Csv for the best layer per model')
    args = parser.parse_args()
    main(args)
