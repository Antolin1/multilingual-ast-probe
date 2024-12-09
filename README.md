# Syntactic multilingual probing of pre-trained language models of code - Codebase and data

## Installation

1. Clone the repository.
```sh
git clone https://github.com/PELAB-LiU/multilingual-ast-probe
cd multilingual-ast-probe
```

2. Create a python3 virtual environment and install `requirements.txt`.
```sh
python3 -m venv <env_ast_probe>
source env_ast_probe/bin/activate
pip install -r requirements.txt
# and install torch and torch scatter
```
The requirements do not include PyTorch and Torch Scatter. You should install the version that better fits your 
computer.


3. Install tree-sitter grammars and build them:

```sh
./script_grammars.sh
```

4. Add project directory to Python path:

```sh
export PYTHONPATH="${PYTHONPATH}:~/multilingual-ast-probe/"
```

5. [Optional] Execute tests:
 
```sh
python -m unittest discover
```

6. Dataset generation:

```sh
./dataset_generation_all_lang.sh
```
This script will download all datasets, filter code snippets, sample 20,000 samples,
and splitting them into training/validation/testing. The filtering criteria are the following:
* We filter out code snippets that have a length `> 512` after tokenization.
* We remove code snippets that cannot be parsed by tree-sitter.
* We remove code snippets containing syntax errors

## Running monolingual AST-Probe 🚀

To run the monolingual AST-Probe just execute the following:
```sh
python src/main.py \
  --do_train \
  --run_name <folder_run_name> \
  --pretrained_model_name_or_path <hugging_face_model> \
  --model_type <model_type> \
  --lang <lang> \
  --layer <layer> \
  --rank <rank>
```

The main arguments are the following:
*  `--do_train`: if you want to train a probe classifier.
*  `--run_name`: indicates the name of the folder where the log, model and results will be stored.
*  `--pretrained_model_name_or_path`: the pre-trained model's id in the HuggingFace Hub.
*e.g.*, `microsoft/codebert-base`, `roberta-base`, `Salesforce/codet5-base`, etc.
*  `--model_type`: the model architecture. Currently, we only support `roberta`, `t5`, `bert`, and `distilbert`.
*  `--lang`: programming language. Currently, we only support `python`, `javascript`, `go`, `php`, `java`, `ruby`, `c`, 
and `csharp`.
*  `--layer`: the layer of the transformer model to probe. Normally, it goes from 0 to 12. 
If the pre-trained models is `huggingface/CodeBERTa-small-v1` or a distilled model, then this argument should range between 0 and 6.
*  `--rank`: dimension of the syntactic subspace.

As a result of this script, a folder `runs/folder_run_name` will be generated. This folder contains three files:
*  `ìnfo.log`: log file.
*  `pytorch_model.bin`: the probing model serialized *i.e.*, the basis of the syntactic subspace, the vectors C and U.
*  `metrics.log`: a serialized dictionary that contains the training losses, the validation losses, the precision, recall, and F1 score on the test set. 
You can use `python -m pickle runs/folder_run_name/metrics.log` to check the metrics for the run.


Here is an example of the usage of this script:
```sh
python src/main.py \
  --do_train \
  --run_name codebert_python_5_128 \
  --pretrained_model_name_or_path microsoft/codebert-base \
  --model_type roberta \
  --lang python \
  --layer 5 \
  --rank 128
```
This command trains a 128-dimensional probe over the output embeddings of the 5th layer of CodeBERT using the Python dataset. 
After running this command, the folder `runs/codebert_python_5_128` is created.

## Running multilingual AST-Probe 🚀

### Direct transfer

To run the direct transfer probe, just execute the following:
```sh
python src/run_transfer.py \
    --source_model <source_model> \
    --target_model <target_model> \
    --source_lang <source_lang> \
    --target_lang <target_lang>
```

Example:

```sh
python src/run_transfer.py \
    --source_model runs/codebert_javascript_5_128/pytorch_model.bin \
    --target_model runs/codebert_java_5_128/pytorch_model.bin \
    --source_lang javascript \
    --target_lang java
```

This command will use the javascript syntactic subspace to predict the Java ASTs.

### The shared syntactic subspace

To compute the shared syntactic subspace, just execute the following:
```sh
python src/main.py \
  --do_train_all_languages \
  --run_name <folder_run_name> \
  --pretrained_model_name_or_path <hugging_face_model> \
  --model_type <model_type> \
  --layer <layer> \
  --rank <rank>
```

The arguments are the same as the monolingual AST-Probe but instead of `--do_train` we have `--do_train_all_languages`.
As a result of this script, a folder `runs/folder_run_name` will be generated. This folder contains three files:
*  `ìnfo.log`: log file.
*  `pytorch_model.bin`: the probing model serialized *i.e.*, the basis of the syntactic subspace, the vectors C and U.
*  `metrics.log`: a serialized dictionary that contains the training losses, the validation losses, the precision, recall, and F1 score on all test sets.
*  `global_labels_c.pkl`: dictionary with the constituency labels for all languages.
*  `global_labels_u.pkl`: dictionary with the unary labels for all languages.

Here is an example of the usage of this script:
```sh
python src/main.py \
  --do_train_all_languages \
  --run_name multilingual_CodeBERT \
  --pretrained_model_name_or_path microsoft/codebert-base \
  --model_type roberta \
  --layer 5 \
  --rank 128
```
This command trains a 128-dimensional multilingual probe over the output embeddings of the 5th layer of CodeBERT 
using the all the training datasets. After running this command, the folder `runs/multilingual_CodeBERT` is created.
