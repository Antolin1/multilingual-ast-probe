# AstProbing

Install tree-sitter grammars:

```sh
mkdir grammars
cd grammars
git clone https://github.com/tree-sitter/tree-sitter-python.git
git clone https://github.com/tree-sitter/tree-sitter-javascript.git
git clone https://github.com/tree-sitter/tree-sitter-go.git
git clone https://github.com/tree-sitter/tree-sitter-php.git
git clone https://github.com/tree-sitter/tree-sitter-ruby.git
git clone https://github.com/tree-sitter/tree-sitter-java.git
git clone https://github.com/tree-sitter/tree-sitter-c-sharp.git
git clone https://github.com/tree-sitter/tree-sitter-c.git
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
python src/dataset_generator.py --download_csn --lang python
python src/dataset_generator.py --lang javascript
python src/dataset_generator.py --lang go
python src/dataset_generator.py --lang php
python src/dataset_generator.py --lang java
python src/dataset_generator.py --lang ruby
python src/dataset_generator.py --download_cxg --lang csharp
python src/dataset_generator.py --lang c
```

Train probing:

```sh
python src/main.py --do_train --run_name folder_run_name 
                   --pretrained_model_name_or_path hugging_face_model
                   --model_type roberta|t5
                   --layer layer_num
                   --rank rank_of_the_probing
```