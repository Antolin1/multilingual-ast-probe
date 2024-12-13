#!/bin/bash

mkdir grammars
cd grammars


git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-python.git
cd tree-sitter-python
git fetch origin 9e53981ec31b789ee26162ea335de71f02186003
git checkout 9e53981ec31b789ee26162ea335de71f02186003
cd ..

git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-javascript.git
cd tree-sitter-javascript
git fetch origin 7a29d06274b7cf87d643212a433d970b73969016
git checkout 7a29d06274b7cf87d643212a433d970b73969016
cd ..

git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-go.git
cd tree-sitter-go
git fetch origin 64457ea6b73ef5422ed1687178d4545c3e91334a
git checkout 64457ea6b73ef5422ed1687178d4545c3e91334a
cd ..

git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-php.git
cd tree-sitter-php
git fetch origin 47dd3532df8204a444dd6eb042135f1e7964f9cb
git checkout 47dd3532df8204a444dd6eb042135f1e7964f9cb
cd ..

git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-ruby.git
cd tree-sitter-ruby
git fetch origin c91960320d0f337bdd48308a8ad5500bd2616979
git checkout c91960320d0f337bdd48308a8ad5500bd2616979
cd ..

git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-java.git
cd tree-sitter-java
git fetch origin 09d650def6cdf7f479f4b78f595e9ef5b58ce31e
git checkout 09d650def6cdf7f479f4b78f595e9ef5b58ce31e
cd ..


git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-c-sharp.git
cd tree-sitter-c-sharp
git fetch origin d83b3c661db34fde4dcd80e79ce1653d4524998d
git checkout d83b3c661db34fde4dcd80e79ce1653d4524998d
cd ..


git clone --depth 1 --no-checkout https://github.com/tree-sitter/tree-sitter-c.git
cd tree-sitter-c
git fetch origin 7175a6dd5fc1cee660dce6fe23f6043d75af424a
git checkout 7175a6dd5fc1cee660dce6fe23f6043d75af424a
cd ..


cd ..
python src/data/build_grammars.py