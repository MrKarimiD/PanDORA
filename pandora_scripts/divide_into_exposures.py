import argparse
from envmap import EnvironmentMap
import numpy as np
import os
from tqdm import tqdm
from pathlib import Path
from hdrio import imwrite
from PIL import Image
import copy


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/nerfstudio_ds/synthetic_classroom/cameras_w_baseline/left/')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/nerfstudio_ds/synthetic_classroom/cameras_w_baseline/all/')
    args = parser.parse_args()

    e1 =  1.0 # None # Left
    e2 =  None # 0.009 # Right
    
    test_images = []
    for path in Path(args.data_dir).rglob('*.exr'):
        test_images.append(path)

    for pano_addr in tqdm(test_images):
        original_pano = EnvironmentMap(pano_addr, 'latlong')
        pixels = original_pano.data[:, :, 0:3]

        pano_file = os.path.basename(pano_addr)
        pano_name =  pano_file[:-4]

        if e1 is not None:
            e1_ldr, e1_hdr, weights_e1 = apply_the_exposure_get_weights(pixels, e1)
            weights_e1_unit = make_weights_binary(e1_ldr, weights_e1, 0.0, 0.95)
            mask_e1 = weights_e1_unit[:, :, 0] & weights_e1_unit[:, :, 1] & weights_e1_unit[:, :, 2]
            # out_addr = os.path.join(args.out_dir, pano_name + '_e1.exr')
            out_addr = os.path.join(args.out_dir, 'lhs_' + pano_name + '.exr')
            imwrite(e1_hdr, out_addr)
            e1_mask_addr = os.path.join(args.out_dir, 'lhs_' + pano_name + '.png')
            # e1_mask_addr = os.path.join(args.out_dir, pano_name + '_e1.png')
            Image.fromarray(mask_e1).save(e1_mask_addr)

        if e2 is not None:
            e2_ldr, e2_hdr, weights_e2 = apply_the_exposure_get_weights(pixels, e2)
            weights_e2_unit = make_weights_binary(e2_ldr, weights_e2, 0.2, 1.0)
            mask_e2 = weights_e2_unit[:, :, 0] & weights_e2_unit[:, :, 1] & weights_e2_unit[:, :, 2]
            # out_addr = os.path.join(args.out_dir, pano_name + '_e2.exr')
            out_addr = os.path.join(args.out_dir, 'rhs_' + pano_name + '.exr')
            imwrite(e2_hdr, out_addr)
            # e2_mask_addr = os.path.join(args.out_dir, pano_name + '_e2.png')
            e2_mask_addr = os.path.join(args.out_dir, 'rhs_' + pano_name + '.png')
            Image.fromarray(mask_e2).save(e2_mask_addr)

        # e1_e2_combined = (weights_e2 * e2_data +  weights_e1 * e1_data) / (weights_e1 + weights_e2)
        # out_addr = os.path.join(args.out_dir, pano_name + '_combine.exr')
        # imwrite(e1_e2_combined, out_addr)

print("Done!")