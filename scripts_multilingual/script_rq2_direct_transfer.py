import os


def main():
    run_dir = 'runs_direct_transfer'
    for i in ['python', 'javascript', 'go', 'php', 'ruby', 'java', 'csharp']:
        for j in ['c']:
            if i == j:
                continue
            print(f'{i} -> {j} direct transfer')
            os.system(
                f"CUDA_VISIBLE_DEVICES=3 python src/main.py --do_train_from_given_projection --run_base_path {run_dir} "
                f"--run_name {i}_transfer_{j} "
                f"--lang {j} "
                f"--model_source_checkpoint runs_monolingual/{i}")


if __name__ == '__main__':
    main()
