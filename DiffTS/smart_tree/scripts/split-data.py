""" Script to split dataset into train and test sets 

  usage: 

  python smart_tree/scripts/split-data.py --read_directory=/speed-tree/speed-tree-outputs/processed_vines/ --json_save_path=/smart-tree/smart_tree/conf/vine-split.json --sample_type=random


"""

import json
import os
import random
from pathlib import Path

import click


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def random_sample(read_dir, json_save_path):
    items = [str(path.name) for path in Path(read_dir).rglob("*.npz")]
    random.shuffle(items)

    data = {}

    data["train"] = sorted(items[: int(0.8 * len(items))])
    data["test"] = sorted(items[int(0.8 * len(items)) : int(0.9 * len(items))])
    data["validation"] = sorted(items[int(0.9 * len(items)) :])

    with open(json_save_path, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)


def strattified_sample(read_dir, json_save_path):
    dirs = os.listdir(read_dir)

    train_paths = []
    test_paths = []
    val_paths = []

    for directory in dirs:
        items = [
            str(path.resolve())
            for path in Path(f"{read_dir}/{directory}").rglob("*.npz")
        ]
        random.shuffle(items)

        train_paths.append(items[: int(0.8 * len(items))])
        test_paths.append(
            items[int(0.8 * len(items)) : int(0.8 * len(items) + int(0.1 * len(items)))]
        )
        val_paths.append(items[int(0.8 * len(items)) + int(0.1 * len(items)) :])

    data = {}

    data["train"] = sorted(flatten_list(train_paths))
    data["test"] = sorted(flatten_list(test_paths))
    data["validation"] = sorted(flatten_list(val_paths))

    with open(json_save_path, "w") as outfile:
        json.dump(data, outfile, indent=4, sort_keys=False)


@click.command()
@click.option(
    "--read_directory", type=click.Path(exists=True), prompt="read directory?"
)
@click.option("--json_save_path", prompt="json path?")
@click.option("--sample_type", type=str, default=False, required=False)
def main(read_directory, json_save_path, sample_type):
    if sample_type == "random":
        random_sample(read_directory, json_save_path)

    if sample_type == "strattified":
        strattified_sample(read_directory, json_save_path)


if __name__ == "__main__":
    main()
