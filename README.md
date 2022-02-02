# AstProbing

Install tree-sitter grammars:

```sh
mkdir grammars
cd grammars
git clone https://github.com/tree-sitter/tree-sitter-python.git
```

Build grammars:

```sh
python src/data/buildGrammars.py
```

Execute tests:
 
```sh
python -m unittest discover
```
