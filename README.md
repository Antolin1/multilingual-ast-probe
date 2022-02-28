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
