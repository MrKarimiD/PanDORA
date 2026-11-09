import argparse
import os
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np
import math


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str)
    parser.add_argument('--output_json', type=str)
    parser.add_argument('--train_split_fraction', type=float, default= 0.8)
    parser.add_argument('--eval_split_fraction', type=float, default= 0.1)
    args = parser.parse_args()

    assert os.path.isfile(args.input_json) and args.input_json.endswith('.json'), 'The input JSON file is missing!'

    json_data = load_from_json( Path(args.input_json) )
    
    
    new_json_data = {}
    all_keys = json_data.keys()
    for key in all_keys:
        new_json_data[key] = json_data[key]


    num_images = len(json_data['frames'])
    num_train_images = math.ceil(num_images * args.train_split_fraction)
    num_eval_test_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval_test = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval_test) == num_eval_test_images

    num_eval_images = math.ceil(num_images * args.eval_split_fraction)
    num_test_images = num_eval_test_images - num_train_images
    i_eval = np.linspace(
        0, num_eval_test_images - 1, num_eval_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = i_eval_test[i_eval]
    i_test = np.setdiff1d(i_eval_test, i_eval)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images


    # has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))

    train_list = []
    for i in i_train:
        train_list.append(json_data['frames'][i]["file_path"])
    new_json_data["train_filenames"] = train_list

    test_list = []
    for i in i_test:
        test_list.append(json_data['frames'][i]["file_path"])
    new_json_data["test_filenames"] = test_list

    eval_list = []
    for i in i_eval:
        eval_list.append(json_data['frames'][i]["file_path"])
    new_json_data["val_filenames"] = eval_list
    
    with open(args.output_json, "w") as outfile: 
        json.dump(new_json_data, outfile, indent=4)
    
    print("All done!")