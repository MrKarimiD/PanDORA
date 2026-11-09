import os
import numpy as np
from tqdm import tqdm
from envmap import EnvironmentMap, rotation_matrix
from hdrio import imwrite
import argparse


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ldr_dir", type=str, required=True, default="/mnt/data/scene/", help="LDR folder, high resolution")
    argparser.add_argument("--hdr_dir", type=str, required=True, default="/mnt/data/scene/", help="LDR2HDR results folder, low resolution")
    argparser.add_argument("--output_dir", type=str, required=True, default="/mnt/data/scene/", help="Output")
    args = argparser.parse_args()

    imgFiles = [os.path.join(args.ldr_dir, f) for f in os.listdir(args.ldr_dir) if os.path.isfile(os.path.join(args.ldr_dir, f)) and f.endswith('.png')]
    imgFiles.sort()

    for ldr_addr in tqdm(imgFiles):
        ldr_pano = EnvironmentMap(ldr_addr, 'latlong')
        
        hdr_addr = ldr_addr.replace('.png', '.exr')
        hdr_addr = hdr_addr.replace(args.ldr_dir, args.hdr_dir)
        assert os.path.exists(hdr_addr), "HDR file does not exists!!!"
        hdr_pano = EnvironmentMap(hdr_addr, 'latlong')
        hdr_pano.resize(1920)

        out_pano = ldr_pano.data # ** 2.2
        list_hdr = hdr_pano.data > 1.0
        print("list_hdr.shape: ", list_hdr.shape, "list_hdr.shape: ",  ldr_pano.data.shape)
        out_pano[list_hdr] = hdr_pano.data[list_hdr]
        
        out_addr = hdr_addr.replace(args.hdr_dir, args.output_dir)
        imwrite(out_pano, out_addr)    

    print("Done!")