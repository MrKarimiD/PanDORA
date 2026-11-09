import argparse
import os
from tqdm import tqdm
from pathlib import Path
import json


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
    parser.add_argument('--img_addr', type=str)
    args = parser.parse_args()
    
    assert os.path.isfile(args.input_json), 'The input JSON file is missing!'

    old_json_data = load_from_json( Path(args.input_json) )
    new_json_data = {}
    all_keys = old_json_data.keys() - {'frames'}
    for key in all_keys:
        new_json_data[key] = old_json_data[key]
    frames = []
    for frame in tqdm(old_json_data['frames']):
        base = str(Path(frame["file_path"]).stem)
        if Path(args.img_addr + '/' + base + '.exr').is_file():
            new_frame = {
                    "file_path": frame["file_path"],
                    "transform_matrix": frame["transform_matrix"],
                }
            frames.append(new_frame)
    new_json_data['frames'] = frames
    
    with open(args.output_json, "w") as outfile: 
        json.dump(new_json_data, outfile)
    
    print("All done!")