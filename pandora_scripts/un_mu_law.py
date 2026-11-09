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
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/linearization_then_nerf/fast_compressed_mu_law/')
    parser.add_argument('--out_dir', type=str, default='//Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/linearization_then_nerf/fast_expo/')
    parser.add_argument('--is_jpg', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    test_images = []
    if args.is_jpg:
        for path in Path(args.data_dir).rglob('*.jpg'):
            test_images.append(path)
    else:
        for path in Path(args.data_dir).rglob('*.exr'):
            test_images.append(path)
    
    for pano_addr in tqdm(test_images):
        pano_addr = str(pano_addr)
        e = EnvironmentMap(pano_addr, 'latlong')

        data = e.data**2.2

        u = 5000.
        img_uncompress = np.exp(e.data * np.log(u+1.)) - 1.
        img_uncompress /= u
        data = img_uncompress

        out_dir = pano_addr.replace(args.data_dir, args.out_dir)

        data = np.float32(data)
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        if args.is_jpg:
            cv2.imwrite(out_dir, data*255)
        else:
            cv2.imwrite(out_dir, data)