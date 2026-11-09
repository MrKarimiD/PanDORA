import argparse
import numpy as np
import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from imageio import imread, imsave
from envmap import EnvironmentMap, rotation_matrix
import math
from tqdm import tqdm
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/Henry_Gagnon_red_and_blue/PanoHDR-NeRF/left_e1_ldr2hdr/')
    parser.add_argument('--out_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/Henry_Gagnon_red_and_blue/PanoHDR-NeRF/left_e1_ldr2hdr_unrectify/')
    parser.add_argument('--unrectify', action='store_true')
    args = parser.parse_args()

    test_images = []
    if args.unrectify:
        for path in Path(args.data_dir).rglob('*.exr'):
            test_images.append(path)
    else:
        for path in Path(args.data_dir).rglob('*.png'):
            test_images.append(path)

    RicohPitch = math.radians(-85.4)
    RicohAzimuth = math.radians(132.03)
    RicohRoll = math.radians(0)

    for pano_addr in tqdm(test_images):
        pano_addr = str(pano_addr)
        e = EnvironmentMap(pano_addr, 'latlong')

        dcm = rotation_matrix(azimuth=0, elevation=RicohAzimuth, roll=RicohPitch)
        from numpy.linalg import inv
        if args.unrectify:
            dcm = inv(dcm)
        e.rotate(dcm)

        out_dir = pano_addr.replace(args.data_dir, args.out_dir)

        if args.unrectify:
            rectified = np.float32(e.data)
            rectified = cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_dir, rectified)
        else:
            rectified = np.clip(255.*e.data, 0, 255).astype('uint8')
            imsave(out_dir, rectified, quality=100)
    