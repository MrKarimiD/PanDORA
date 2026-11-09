import os
import shutil
import json
import yaml
import argparse

def copy_images(src_folder, dest_folder, folder):
    os.makedirs(dest_folder, exist_ok=True)

    files = os.listdir(src_folder)
    for filename in files:
        if filename.endswith(".png"):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_folder, folder + "_" + filename)
            shutil.copy(src_file, dest_file)

def create_camera_models_overrides_json(dest_folder):
    camera_models_data = {
        "all": {
            "projection_type": "equirectangular",
            "width": 3840,
            "height": 1920
        }
    }

    json_file_path = os.path.join(dest_folder, 'camera_models_overrides.json')
    
    with open(json_file_path, 'w') as json_file:
        json.dump(camera_models_data, json_file, indent=2)

def create_config_yml(dest_folder):
    config_data = {
        "processes": 8,
        "read_processes": 8,
    }
    
    yaml_file_path = os.path.join(dest_folder, 'config.yml')
    
    with open(yaml_file_path, 'w') as yaml_file:
        yaml_file.write("# Other params\n")
        yaml.dump(config_data, yaml_file, default_flow_style=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default="/mnt/data/scene")
    args = argparser.parse_args()

    src_folder = args.input_dir + "data/"
    dest_folder = args.input_dir + "sfm/"
    dest_image_folder = dest_folder + "images"
    dest_mask_folder = dest_folder + "masks"

    for folder in ("left_e1", "left_sfm", "right_sfm"):
        copy_images(src_folder + folder + "/", dest_image_folder, folder)
        copy_images(src_folder + folder + "/mask/", dest_mask_folder, folder)

    create_camera_models_overrides_json(dest_folder)
    create_config_yml(dest_folder)
    