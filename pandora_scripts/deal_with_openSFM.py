import numpy as np
import argparse
import json


def read_json_file(path):

    with open(path) as f:
        data = json.load(f)
    camera_dict = {}
    camera_dict_file = './transform.json'

    camera_dict["frames"] = []

    for img_name in data[0]['shots']:

        rvec = np.array(tuple(map(float, data[0]['shots'][img_name]['rotation'])))
        tvec = np.array(tuple(map(float, data[0]['shots'][img_name]['translation'])))
        
        from scipy.spatial.transform import Rotation as R
        rot = R.from_rotvec(rvec).as_matrix()
        W2C = np.eye(4)
        W2C[:3, :3] = rot
        W2C[:3, 3] = np.array(tvec)

        camera_dict["frames"].append(
            {
                "file_path": img_name,
                "transform_matrix": W2C.tolist()
            }
        )
        
    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and write COLMAP binary and text models')
    parser.add_argument('--input_model', help='path to input model folder')
    args = parser.parse_args()
    read_json_file(args.input_model)
