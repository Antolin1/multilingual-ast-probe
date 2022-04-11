import os


def main():
    run_dir = 'runs_hold_one_out'
    for i in ['python', 'javascript', 'go', 'php', 'ruby', 'java']:
        os.system(
                f"CUDA_VISIBLE_DEVICES=3 python src/main.py --do_hold_one_out_training --run_base_path {run_dir} "
                f"--run_name all_less_{i} --lang {i}")
    for j in ['python', 'javascript', 'go', 'php', 'ruby', 'java']:
        os.system(
            f"CUDA_VISIBLE_DEVICES=3 python src/main.py --do_train_from_given_projection --run_base_path {run_dir} "
            f"--run_name hold_one_out_{j} "
            f"--lang {j} "
            f"--model_source_checkpoint runs_hold_one_out/all_less_{j}")


if __name__ == '__main__':
    main()
