# AstProbing

Generate venv and install dependencies:

```sh
python3 -m venv <env_ast_probe>
source env_ast_probe/bin/activate
pip install -r requirements.txt
```


Install tree-sitter grammars and build them:

```sh
./script_grammars.sh
```

Add project directory to Python path:

```sh
export PYTHONPATH="${PYTHONPATH}:~/AstProbing/"
```

Execute tests:
 
```sh
python -m unittest discover
```

Dataset generation:

```sh
./dataset_generation_all_lang.sh
```

Train probing:

```sh
python src/main.py --do_train --run_name folder_run_name 
                   --pretrained_model_name_or_path hugging_face_model
                   --model_type roberta|t5
                   --layer layer_num
                   --rank rank_of_the_probing
```

RQs:
```sh
python rq1_script.py
python analyze_results_rq1.py
python rq2_script.py
python analyze_results_rq2.py
python analyze_results_rq3.py --results_finetuning finetuning_results/code_search_avg.json
```

Visualization:
```sh
python visualization_multilingual.py --run_folder runs/multilingual_CodeBERTrand-baseline/ --model baseline
python visualization_multilingual.py --run_folder runs/multilingual_CodeBERT/ --model codebert
python angle_subspaces.py --model codebert
```