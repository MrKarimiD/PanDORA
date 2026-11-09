import argparse
from envmap import EnvironmentMap, rotation_matrix
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import imageio

def tonemap(img, gamma=2.4):
    """Apply gamma, then clip between 0 and 1, finally convert to uint8 [0,255]"""
    return (np.clip(np.power(img,1/gamma), 0.0, 1.0)*255).astype('uint8')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Volumes/Momo/Adobe_2023/Evaluation/test_sets/weber_test_set_original_hdr_panos/')
    parser.add_argument('--out_dir', type=str, default='/Users/momo/Desktop/camera_ready/FID_test_set_256x512/')
    parser.add_argument('--do_resize', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument("--resolution", type=int, default=1024)
    args = parser.parse_args()
    
    test_images = []
    for path in Path(args.data_dir).rglob('*.exr'):
        test_images.append(path)
    
    imageio.plugins.freeimage.download()

    for pano_addr in tqdm(test_images):
        # original_pano = imageio.imread(pano_addr, format='HDR-FI') # EnvironmentMap(pano_addr, 'latlong')
        # # gt_ldr = tonemap(reexpose_hdr(original_pano.data)[0])
        # gt_ldr = tonemap(reexpose_hdr(original_pano)[0])
        
        original_pano = EnvironmentMap(pano_addr, 'latlong')
        # gt_ldr = tonemap(reexpose_hdr(original_pano.data)[0], gamma=2.2) # for computing the FID in table 1, sue tune mapping of 2.2
        # gt_ldr = tonemap(reexpose_hdr(original_pano)[0])
        gt_ldr = tonemap(original_pano.data, gamma=2.2) # No Need to re-expose the StyleLight and EverLight
        
        pano_file = os.path.basename(pano_addr)
        pano_file = pano_file.replace('.exr', '.png')
        gt_ldr_addr = os.path.join(args.out_dir, pano_file)
        if args.do_resize:
            Image.fromarray(gt_ldr).resize((2 * args.resolution, args.resolution), resample=Image.LANCZOS).save(gt_ldr_addr)
        else:
            Image.fromarray(gt_ldr).save(gt_ldr_addr)
print("Done!")