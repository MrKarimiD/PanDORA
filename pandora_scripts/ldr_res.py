import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import torch
import piq


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
    parser.add_argument('--gt_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/lab_downstairs/GT/pano/')
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/lab_downstairs/HDR-Nerfacto/pano/')
    parser.add_argument('--method_name', type=str)
    parser.add_argument('--is_exr', action='store_true')
    args = parser.parse_args()
    well_addr = args.data_dir

    gt_images = []
    for path in Path(args.gt_dir).rglob('*.exr'):
        gt_images.append(path)

    pred_images = []
    # if args.is_exr:
    for path in Path(args.data_dir).rglob('*.exr'):
        pred_images.append(path)
    # else:
    #     for path in Path(args.data_dir).rglob('*.png'):
    #         pred_images.append(path)
    assert len(pred_images) == len(gt_images), "The size of images don't match GT size"

    gts = []
    preds = []

    # import pdb; pdb.set_trace()
    for gt_addr in tqdm(gt_images):
        gt_addr = str(gt_addr)
        gt_image = cv2.imread(gt_addr, cv2.IMREAD_UNCHANGED)
        gt_image[gt_image < 0] = 0.0
        gt_image = cv2.resize(gt_image, (512, 256), interpolation = cv2.INTER_LINEAR)
        gt_image = np.clip(np.power(gt_image, 1/2.2), 0.0, 1.0)
        pred_addr = gt_addr.replace(args.gt_dir, args.data_dir)
        # if args.is_exr:
            # pred_addr = pred_addr.replace('.png', '.exr')
        pred_image = cv2.imread(pred_addr,  cv2.IMREAD_UNCHANGED)
        pred_image = np.clip(np.power(pred_image, 1/2.2), 0.0, 1.0)
        pred_image = cv2.resize(pred_image, (512, 256), interpolation = cv2.INTER_LINEAR)
        
        cv2.imwrite('./test.png', (pred_image*255).astype('uint8'))
        cv2.imwrite('./gt.png', (gt_image*255).astype('uint8'))
        # else:
        #     # pred_addr = pred_addr.replace('.png', '.jpg')
        #     pred_image = cv2.imread(pred_addr,  cv2.IMREAD_UNCHANGED) / 255.0

        if not np.isnan(gt_image).any():
            gt_image_torch = torch.from_numpy(gt_image) 
            gt_image_torch = gt_image_torch.permute(2, 0, 1)
            gts.append(gt_image_torch)
            pred_image_torch = torch.from_numpy(pred_image)
            pred_image_torch = pred_image_torch.permute(2, 0, 1)
            preds.append(pred_image_torch)

    ssim_index = piq.ssim(torch.stack(gts), torch.stack(preds), data_range=1.)
    psnr_index = piq.psnr(torch.stack(gts), torch.stack(preds), data_range=1.)
    
    print("SSIM: ", ssim_index)
    print("PSNR: ", psnr_index)
    lpips_index = piq.LPIPS(reduction='none')(torch.stack(gts), torch.stack(preds))
    print("LPIPS: ", torch.mean(lpips_index))

    results = {
    "SSIM": str(ssim_index),
    "PSNR": str(psnr_index),
    "LPIPS": str(torch.mean(lpips_index))
    }

    output_path = os.path.join(args.data_dir, "ldr_results.json")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved metrics to: {output_path}")