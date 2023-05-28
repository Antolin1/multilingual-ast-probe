import os

from src.data import LANGUAGES

for lang in LANGUAGES:
    os.system(f"CUDA_VISIBLE_DEVICES=0 python src/main.py --do_holdout_training "
              f"--run_name holdout_{lang} --lang {lang} --epochs 10")
