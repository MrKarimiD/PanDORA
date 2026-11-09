import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json


FAST_EXPOSURE_CUTOFF = 0.1 # good for real data (0.1 for synthetic)
WELL_EXPOSURE_CUTOFF = 0.98 # good for real data (0.9 for synthetic)


def apply_correction(coefs, img, CORRECTION_CURVE_TYPE) -> np.ndarray:
    # check that img is in range 0-255
    assert img.dtype == np.uint8
    assert CORRECTION_CURVE_TYPE in ['gamma']
    img = ((img.astype(np.float32) / 255.0) ** np.array(coefs).T) # * 255.0
    # img = ((img.astype(np.float32) / 255.0) ** coefs[None, None, :]) * 255.0
    return img

def cut_weighting_function(pixels, exposures):
    weights = np.zeros(pixels.shape, dtype=np.float32)
    
    if exposures == WELL_EXPOSURE:
        weights = np.clip(-20.0 * (pixels - WELL_EXPOSURE_CUTOFF) + 1, 0.0, 1.0)
    else:
        weights = np.clip(100.0 * (pixels - FAST_EXPOSURE_CUTOFF), 0.0, 1.0)

    return weights


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--well_dir', type=str)
    parser.add_argument('--fast_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--do_linearization', action='store_true')
    parser.add_argument('--experiment_location', type=str, required=True)
    args = parser.parse_args()

    experiment_setup = args.experiment_location + '/capture_settings.json'
    assert os.path.exists(experiment_setup), "!!!! Cannot find the files for exposure setup.  !!!"
    with open(experiment_setup) as f:
        data = json.load(f)
        WELL_EXPOSURE = 1.0
        FAST_EXPOSURE = data["right"]["shutter_speed"] / data["left"]["shutter_speed"]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    well_addr = args.well_dir

    with open(f'linearize-ricoh-z10-wb-3500.json') as f:
        camera_data = json.load(f)
        CORRECTION_CURVE_TYPE = 'gamma'

    well_images = []
    if args.do_linearization:
        for path in Path(well_addr).rglob('*.jpg'):
            well_images.append(path)
    else:
        for path in Path(well_addr).rglob('*.exr'):
            well_images.append(path)

    for well_addr in tqdm(well_images):

        well_addr = str(well_addr)

        hdr_well_image = cv2.imread(well_addr,  cv2.IMREAD_UNCHANGED)
        if args.fast_dir is not None:
            hdr_fast_image = cv2.imread(well_addr.replace(args.well_dir, args.fast_dir), cv2.IMREAD_UNCHANGED)


        # hdr_well_image = np.uint8(hdr_well_image * 255.0)
        # hdr_fast_image = np.uint8(hdr_fast_image * 255.0)
        
        if args.do_linearization:
            hdr_well_image = apply_correction((camera_data["b"], camera_data["g"], camera_data["r"]), np.array(hdr_well_image), CORRECTION_CURVE_TYPE)
                
            if args.fast_dir is not None:
                hdr_fast_image = apply_correction((camera_data["b"], camera_data["g"], camera_data["r"]), np.array(hdr_fast_image), CORRECTION_CURVE_TYPE)


        # well_mask_addr = well_addr.replace('/well_expo/', '/well_mask/')
        # well_mask_addr = well_mask_addr.replace('.exr', '.png')
        # well_mask = cv2.imread(well_mask_addr)
        # well_mask = well_mask.astype(np.float32) / 255.0
        # well_mask[well_mask > 0.1] = 1.0

        
        if args.fast_dir is not None:
            weights_well = cut_weighting_function(hdr_well_image, WELL_EXPOSURE)
            weights_fast = cut_weighting_function(hdr_fast_image, FAST_EXPOSURE)

        # fast_mask = cv2.imread(well_mask_addr.replace('/well_mask/', '/fast_mask/'))
        # fast_mask = fast_mask.astype(np.float32) / 255.0

        # final_output = weights_well * (hdr_well_image / WELL_EXPOSURE) + weights_fast * (hdr_fast_image / FAST_EXPOSURE)
        # final_output = final_output / (weights_well+weights_well)
        
        if args.fast_dir is not None:
            final_output = weights_well * (hdr_well_image / WELL_EXPOSURE) + (hdr_fast_image / FAST_EXPOSURE)
        else:
            final_output = hdr_well_image / WELL_EXPOSURE
        # final_output = weights_well * (hdr_well_image / WELL_EXPOSURE)

        output_addr = well_addr.replace(args.well_dir, args.out_dir)
        if args.do_linearization:
            output_addr = output_addr.replace('jpg', '.exr')
        cv2.imwrite(output_addr, np.float32(final_output) )
        # import pdb; pdb.set_trace()