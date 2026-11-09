import argparse
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import PIL.ExifTags
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import warnings 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/GT/pano/')
    parser.add_argument('--mask_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/GT/mask/')
    parser.add_argument('--out_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/GT/masked_pano/')
    args = parser.parse_args()
    
    test_images = []
    for path in Path(args.data_dir).rglob('*.exr'):
        test_images.append(path)

    for pano_addr in tqdm(test_images):
        img = cv2.imread(str(pano_addr), cv2.IMREAD_UNCHANGED)

        mask_addr = str(pano_addr).replace(args.data_dir, args.mask_dir)
        mask_addr = mask_addr.replace('.exr', '.png')
        mask = cv2.imread(mask_addr, cv2.IMREAD_UNCHANGED)
        mask = np.float32(mask) / 255.

        # import pdb; pdb.set_trace()
        masked_img = np.expand_dims(mask, axis=-1) * img
        # import pdb; pdb.set_trace()
        out_addr = str(pano_addr).replace(args.data_dir, args.out_dir)
        # import pdb; pdb.set_trace()
        cv2.imwrite(out_addr, np.float32(masked_img))