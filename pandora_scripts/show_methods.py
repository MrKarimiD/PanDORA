import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def tonemap(img, gamma=2.2):
    """Apply gamma, then clip between 0 and 1, finally convert to uint8 [0,255]"""
    return (np.clip(np.power(img,1/gamma), 0.0, 1.0)*255).astype('uint8')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/Henry_Gagnon_red_and_blue/')
    args = parser.parse_args()

    # methods = ['LDR-Nerfacto', 'linearization_then_nerf', 'nerf_then_linearization', 'PanoHDR-NeRF', 'HDR-Nerfacto']
    methods = ['LDR-Nerfacto', 'PanoHDR-NeRF', 'HDR-Nerfacto', 'linearization_then_nerf', 'nerf_then_linearization']
    # methods = ['LDR-Nerfacto', 'linearization_then_nerf', 'nerf_then_linearization']

    
    gt_folder = args.data_dir + 'GT' + '/render/'
    gt_images = []
    for path in Path(gt_folder).rglob('*.exr'):
        gt_images.append(str(path))

    fig = plt.figure(figsize=(8, 8))
    columns = len(methods) + 1
    rows = len(gt_images)

    i = 1
    for gt_addr in gt_images:
        gt_img = cv2.imread(gt_addr, cv2.IMREAD_UNCHANGED)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = tonemap(gt_img)
        for method in tqdm(methods):
            method_folder = args.data_dir + method + '/render/'
            img_addr = gt_addr.replace(gt_folder, method_folder)
            inp_img = cv2.imread(img_addr, cv2.IMREAD_UNCHANGED)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
            inp_img = tonemap(inp_img)
            fig.add_subplot(rows, columns, i)
            i += 1
            plt.axis('off')
            plt.imshow(inp_img)
            plt.title(method)
        fig.add_subplot(rows, columns, i)
        i += 1
        plt.imshow(gt_img)
        plt.axis('off')
        plt.title('GT')

    plt.show()