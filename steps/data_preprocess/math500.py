"""
Preprocess the Math 500 dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/math500")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "HuggingFaceH4/MATH-500"

    dataset = datasets.load_dataset(data_source)['test']

    # train validate test splits by portion of 0.6, 0.2, 0.2
    train_test_dataset = dataset.train_test_split(test_size=0.4, seed=42, shuffle=True)
    test_val_dateset = train_test_dataset['test'].train_test_split(test_size=0.5, seed=42, shuffle=True)

    train_dataset = train_test_dataset['train']
    test_dataset = test_val_dateset['train']
    val_dataset = test_val_dateset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            solution = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    val_dataset = val_dataset.map(function=make_map_fn("val"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
