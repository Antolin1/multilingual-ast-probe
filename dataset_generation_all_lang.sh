#!/bin/bash

python src/dataset_generator.py --download_csn --lang python
python src/dataset_generator.py --lang javascript
python src/dataset_generator.py --lang go
python src/dataset_generator.py --lang php
python src/dataset_generator.py --lang java
python src/dataset_generator.py --lang ruby
python src/dataset_generator.py --download_cxg --lang csharp
python src/dataset_generator.py --lang c
