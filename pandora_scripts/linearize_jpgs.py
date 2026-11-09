import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from pathlib import Path
import copy


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def apply_correction(coefs, img, CORRECTION_CURVE_TYPE) -> np.ndarray:
    # check that img is in range 0-255
    assert img.dtype == np.uint8
    assert CORRECTION_CURVE_TYPE in ['gamma']
    img = ((img.astype(np.float32) / 255.0) ** np.array(coefs).T) # * 255.0
    # img = ((img.astype(np.float32) / 255.0) ** coefs[None, None, :]) * 255.0
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/LDR_Nerfacto/pano_jpeg/')
    parser.add_argument('--out_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/LDR_Nerfacto/pano/')
    parser.add_argument('--do_resize', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    well_addr = args.data_dir

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(f'./linearize-ricoh-z10-wb-3500.json') as f:
        camera_data = json.load(f)
        CORRECTION_CURVE_TYPE = 'gamma'

    well_images = []
    for path in Path(well_addr).glob('*.png'):
        well_images.append(path)

    for well_addr in tqdm(well_images):

        well_addr = str(well_addr)
        hdr_well_image = cv2.imread(well_addr,  cv2.IMREAD_UNCHANGED)
        if args.do_resize:
            resized_image = cv2.resize(hdr_well_image, (2 * args.resolution, args.resolution), interpolation=cv2.INTER_LANCZOS4)
            hdr_well_image = copy.deepcopy(resized_image)
        final_output = apply_correction((camera_data["b"], camera_data["g"], camera_data["r"]), np.array(hdr_well_image), CORRECTION_CURVE_TYPE)

        output_addr = well_addr.replace(args.data_dir, args.out_dir)
        # output_addr = output_addr.replace('.png', '.exr')
        # cv2.imwrite(output_addr, np.float32(final_output) )
        cv2.imwrite(output_addr, np.float32(final_output*255) )