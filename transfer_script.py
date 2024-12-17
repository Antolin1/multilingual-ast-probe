import os

from src.data import LANGUAGES


def main():
    for source in LANGUAGES:
        for target in LANGUAGES:
            os.system(f"CUDA_VISIBLE_DEVICES=1 python src/run_transfer.py "
                      f"--source_model runs/codebert_{source}_5_128/pytorch_model.bin "
                      f"--target_model runs/codebert_{target}_5_128/pytorch_model.bin "
                      f"--source_lang {source} "
                      f"--target_lang {target}")


if __name__ == '__main__':
    main()
