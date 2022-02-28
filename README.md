# AstProbing

Install tree-sitter grammars:

```sh
mkdir grammars
cd grammars
git clone https://github.com/tree-sitter/tree-sitter-python.git
git clone https://github.com/tree-sitter/tree-sitter-javascript.git
git clone https://github.com/tree-sitter/tree-sitter-go.git
```

Build grammars:

```sh
python src/data/build_grammars.py
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
python src/dataset_generator.py --download --lang python
python src/dataset_generator.py --lang javascript
python src/dataset_generator.py --lang go
```

Train probing:

```sh
python src/main.py --do_train --run_name folder_run_name 
                   --pretrained_model_name_or_path hugging_face_model
                   --model_type roberta|t5
                   --layer layer_num
                   --rank rank_of_the_probing
```