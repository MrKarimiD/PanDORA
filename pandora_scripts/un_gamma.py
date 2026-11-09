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
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/Henry_Gagnon_red_and_blue/PanoHDR-NeRF/pano_gamma_tone_mapped/')
    parser.add_argument('--out_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/Henry_Gagnon_red_and_blue/PanoHDR-NeRF/pano/')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    test_images = []
    for path in Path(args.data_dir).rglob('*.exr'):
        test_images.append(path)
    
    for pano_addr in tqdm(test_images):
        pano_addr = str(pano_addr)
        e = EnvironmentMap(pano_addr, 'latlong')

        data = e.data**2.2

        out_dir = pano_addr.replace(args.data_dir, args.out_dir)

        data = np.float32(data)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_dir, data)