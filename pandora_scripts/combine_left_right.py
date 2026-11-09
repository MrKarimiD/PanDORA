import argparse
import copy
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def weighting_function(pixels, zmin, zmax):
    threshold = 0.5 * (zmin + zmax)
    weights = np.zeros(pixels.shape) + np.finfo(np.float64).eps
    
    # mask_lowerbound = zmin <= pixels <= threshold
    mask_lowerbound = np.greater_equal(pixels, zmin) & np.less_equal(pixels, threshold)
    weights[mask_lowerbound] = pixels[mask_lowerbound] - zmin

    # mask_upperbound = threshold <= pixels <= zmax
    mask_upperbound = np.greater_equal(pixels, threshold) & np.less_equal(pixels, zmax)
    weights[mask_upperbound] = zmax - pixels[mask_upperbound]

    # sums_R = np.sum(weights[:,:,0])
    # sums_G = np.sum(weights[:,:,1])
    # sums_B = np.sum(weights[:,:,2])

    return weights #/ [sums_R, sums_G, sums_B]


def apply_the_exposure_get_weights(hdr_data, exposure):
    e1_data = np.clip(hdr_data * exposure, 0, 1)
    e1_data = (e1_data * 255.0).astype("uint8")
    e1_data = e1_data.astype('float32')
    e1_data = e1_data / 255.0
    weights = weighting_function(e1_data, 0.05, 0.95)
    e1_data_HDR = e1_data / exposure

    return e1_data, e1_data_HDR, weights


def make_weights_binary(pixels, weights, thresh_min, thresh_max, threshold_weights = False):
    outputs = np.zeros(pixels.shape)
    if threshold_weights:
        weights_e1_if_case = copy.deepcopy(weights)
    else:
        weights_e1_if_case = copy.deepcopy(pixels)
    # masked = thresh_min <= weights_e1_if_case <= thresh_max
    masked = np.greater_equal(weights_e1_if_case, thresh_min) & np.less_equal(weights_e1_if_case, thresh_max)
    outputs[masked] = 1.0
    outputs_unit = (255. * outputs).astype('uint8')
    
    return outputs_unit


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.

    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    # with open(filename, "w", encoding="UTF-8") as file:
    #     json.dump(content, file)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=4)



def process_one_image(pixels, out_dir, name, has_exposure, exposure_value, zmin, zmax):
    if has_exposure:
        e1_ldr, e1_hdr, weights_e1 = apply_the_exposure_get_weights(pixels, exposure_value)
        weights_e1_unit = make_weights_binary(e1_ldr, weights_e1, zmin, zmax)
        mask_e1 = weights_e1_unit[:, :, 0] & weights_e1_unit[:, :, 1] & weights_e1_unit[:, :, 2]
        e1_mask_addr = os.path.join(out_dir, name + '.png')
        # e1_mask_addr = os.path.join(args.out_dir, pano_name + '_e1.png')
        Image.fromarray(mask_e1).save(e1_mask_addr)
    else:
        e1_hdr = pixels
    # out_addr = os.path.join(args.out_dir, pano_name + '_e1.exr')
    out_addr = os.path.join(out_dir, name + '.exr')
    os.makedirs(os.path.dirname(out_addr), exist_ok=True)
    cv2.imwrite(out_addr, e1_hdr) 


def process_all_images(address, out_dir, prefix, has_exposure, exposure_value, zmin, zmax):
    images = [os.path.join(address, f) for f in os.listdir(address) if os.path.isfile(os.path.join(address, f)) and f.endswith('.exr')]
    
    images = sorted(images)
    for pano_addr in tqdm(images):
        # original_pano = EnvironmentMap(pano_addr, 'latlong')
        # pixels = original_pano.data[:, :, 0:3]
        original_pano = np.array(cv2.imread(str(pano_addr), cv2.IMREAD_UNCHANGED)).astype("float32")
        pixels = original_pano[:, :, 0:3]

        pano_file = os.path.basename(pano_addr)
        pano_name =  pano_file[:-4]
        name = prefix + pano_name

        process_one_image(pixels, out_dir, name, has_exposure, exposure_value, zmin, zmax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--left_dir', type=str, default='/mnt/workspace/share_w_Junming/Dual_Cameras/left/')
    parser.add_argument('--right_dir', type=str, default='/mnt/workspace/share_w_Junming/Dual_Cameras/right/')
    parser.add_argument('--out_dir', type=str, default='/mnt/workspace/share_w_Junming/Dual_Cameras/pandora-HDR/')
    parser.add_argument('--has_exposure', action='store_true')
    parser.add_argument('--exposure1', type=float, default=1.0)
    parser.add_argument('--exposure2', type=float, default=0.009)
    parser.add_argument('--skip_the_json', action='store_true')
    args = parser.parse_args()
    
    if not args.skip_the_json:
        assert os.path.isfile(args.left_dir + '/transforms.json'), 'The JSON files for left set is missing!'
        assert os.path.isfile(args.right_dir + '/transforms.json'), 'The JSON files for right set is missing!'
    assert os.path.isdir(args.left_dir + '/HDR_Normal/'), 'The data folder for left set is missing!'
    assert os.path.isdir(args.right_dir + '/HDR_Normal/'), 'The data folder for right set is missing!'

    assert not os.path.isdir(args.out_dir), 'The output folder exists!! Take care of it first!'
    os.mkdir(args.out_dir)
    os.mkdir(args.out_dir + 'HDR_Normal/')
    
    if not args.skip_the_json:
        cam_params_lhs = load_from_json( Path(args.left_dir + '/transforms.json') )
        cam_params_combine = {}
        cam_params_combine["camera_angle_x"] = cam_params_lhs["camera_angle_x"]
        cam_params_combine["frames"] = []
        for frame in tqdm(cam_params_lhs["frames"]):
            file_path = 'lhs_' + frame["file_path"].split('/')[-1]
            transform_matrix = frame["transform_matrix"]
            cam_params_combine["frames"].append(
                {
                    "file_path": file_path,
                    "transform_matrix": transform_matrix
                }
            )
            # TODO adding exposures
        cam_params_rhs = load_from_json( Path(args.right_dir + '/transforms.json') )
        for frame in tqdm(cam_params_rhs["frames"]):
            file_path = 'rhs_' + frame["file_path"].split('/')[-1]
            transform_matrix = frame["transform_matrix"]
            cam_params_combine["frames"].append(
                {
                    "file_path": file_path,
                    "transform_matrix": transform_matrix
                }
            )
            # TODO adding exposures
        write_to_json(Path(args.out_dir + '/transforms_all.json'), cam_params_combine)
    
    process_all_images(args.left_dir + '/HDR_Normal/', args.out_dir + '/HDR_Normal/', 'lhs_', args.has_exposure, args.exposure1, 0.0, 0.95)
    process_all_images(args.right_dir + '/HDR_Normal/', args.out_dir + '/HDR_Normal/', 'rhs_', args.has_exposure, args.exposure2, 0.2, 1.0)

print("Done!")