import argparse
import json
import math
import os
import re
import shutil
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def is_mask_all_valid(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return np.all(mask > 200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--train_split_fraction', type=float, default= 0.60)
    parser.add_argument('--eval_split_fraction', type=float, default= 0.01)
    parser.add_argument('--min_frame_number', type=int, default=100)
    parser.add_argument('--frames_per_direction', type=int, default=100)
    args = parser.parse_args()

    assert os.path.isfile(args.input_json) and args.input_json.endswith('.json'), 'The input JSON file is missing!'

    os.makedirs(args.output_dir, exist_ok=True)

    transforms_data = load_from_json( Path(args.input_json) )
    exposures_data = load_from_json( Path(args.input_json).parent / 'exposures.json' )
    
    
    selected_left_frames = []
    selected_right_frames = []
    new_exposures = dict()
    
    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
    for i, frame in enumerate(transforms_data['frames']):
        file_path = frame['file_path'] # e.g. images/left_e1_output_0010_0_linear.exr (frame num is 0010)
        mask_path = str(Path(args.input_json).parent / frame['mask_path']) # e.g. masks/left_e1_output_0010_0.png
        # get frame number
        frame_num = int(re.search(r'(\d{4})', file_path).group(0))
        direction = re.search(r'(left|right)', file_path, re.IGNORECASE).group(0).lower()

        if int(frame_num) >= args.min_frame_number:
            if direction == 'left':
                if len(selected_left_frames) < args.frames_per_direction and is_mask_all_valid(mask_path):
                    selected_left_frames.append(frame)
                    
                    shutil.copy(str(Path(args.input_json).parent / file_path), os.path.join(args.output_dir, file_path))
                    shutil.copy(str(Path(args.input_json).parent / file_path.replace('_linear.exr', '.png')), os.path.join(args.output_dir, file_path.replace('_linear.exr', '.png')))
                    shutil.copy(mask_path, os.path.join(args.output_dir, frame['mask_path']))
                    shutil.copy(mask_path.replace('.png', '_saturation_mask.png'), os.path.join(args.output_dir, frame['mask_path'].replace('.png', '_saturation_mask.png')))

                    new_exposures[file_path] = exposures_data[file_path]
            else:
                if len(selected_right_frames) < args.frames_per_direction and is_mask_all_valid(mask_path):
                    selected_right_frames.append(frame)
                            
                    shutil.copy(str(Path(args.input_json).parent / file_path), os.path.join(args.output_dir, file_path))
                    shutil.copy(str(Path(args.input_json).parent / file_path.replace('_linear.exr', '.png')), os.path.join(args.output_dir, file_path.replace('_linear.exr', '.png')))
                    shutil.copy(mask_path, os.path.join(args.output_dir, frame['mask_path']))
                    shutil.copy(mask_path.replace('.png', '_saturation_mask.png'), os.path.join(args.output_dir, frame['mask_path'].replace('.png', '_saturation_mask.png')))

                    new_exposures[file_path] = exposures_data[file_path]

        
    new_transforms_data = {}
    all_keys = transforms_data.keys()
    for key in all_keys:
        new_transforms_data[key] = transforms_data[key]
    
    new_transforms_data['frames'] = selected_left_frames + selected_right_frames

    num_images_per_direction = len(selected_left_frames)

    num_train_images = math.ceil(num_images_per_direction * args.train_split_fraction)
    num_eval_test_images = num_images_per_direction - num_train_images
    i_all = np.arange(num_images_per_direction)
    i_train = np.linspace(
        0, num_images_per_direction - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval_test = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval_test) == num_eval_test_images

    num_eval_images = math.ceil(num_images_per_direction * args.eval_split_fraction)
    num_test_images = num_eval_test_images - num_train_images
    i_eval = np.linspace(
        0, num_eval_test_images - 1, num_eval_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = i_eval_test[i_eval]
    i_test = np.setdiff1d(i_eval_test, i_eval)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    hdr_nerf_transforms_data = dict()
    hdr_nerf_exposures_data = dict()
    for split, indices in zip(("train", "val", "test"), (i_train, i_eval, i_test)):
        hdr_nerf_transforms_data[split] = deepcopy(transforms_data)
        hdr_nerf_transforms_data[split]["frames"] = [selected_left_frames[i] for i in indices] + [selected_right_frames[i] for i in indices]
        hdr_nerf_exposures_data[split] = {k: v for k, v in new_exposures.items() if k in [frame["file_path"] for frame in hdr_nerf_transforms_data[split]["frames"]]}

        with open(os.path.join(args.output_dir, f'transforms_{split}.json'), "w") as outfile:
            json.dump(hdr_nerf_transforms_data[split], outfile, indent=4)

        with open(os.path.join(args.output_dir, f'exposure_{split}.json'), "w") as outfile:
            json.dump(hdr_nerf_exposures_data[split], outfile, indent=4)

    # has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))

    train_list = []
    for i in i_train:
        train_list.append(selected_left_frames[i]["file_path"])
        train_list.append(selected_right_frames[i]["file_path"])
    new_transforms_data["train_filenames"] = train_list

    test_list = []
    for i in i_test:
        test_list.append(selected_left_frames[i]["file_path"])
        test_list.append(selected_right_frames[i]["file_path"])
    new_transforms_data["test_filenames"] = test_list

    eval_list = []
    for i in i_eval:
        eval_list.append(selected_left_frames[i]["file_path"])
        eval_list.append(selected_right_frames[i]["file_path"])
    new_transforms_data["val_filenames"] = eval_list
    
    with open(os.path.join(args.output_dir, 'transforms.json'), "w") as outfile: 
        json.dump(new_transforms_data, outfile, indent=4)
    
    with open(os.path.join(args.output_dir, 'exposures.json'), "w") as outfile:
        json.dump(new_exposures, outfile, indent=4)
    
    print("All done!")
