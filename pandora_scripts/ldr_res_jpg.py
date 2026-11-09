import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import torch
import piq


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/lab_downstairs/GT/pano/')
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/lab_downstairs/HDR-Nerfacto/pano/')
    parser.add_argument('--method_name', type=str)
    args = parser.parse_args()

    # Collect GT and prediction jpg images (case-insensitive extension).
    def list_jpgs(directory):
        images = {}
        for path in Path(directory).rglob('*'):
            if path.suffix.lower() in ('.jpg', '.jpeg'):
                images[path.stem] = path
        return images

    gt_images = list_jpgs(args.gt_dir)
    pred_images = list_jpgs(args.data_dir)

    assert len(pred_images) == len(gt_images), "The size of images don't match GT size"

    gts = []
    preds = []

    for stem, gt_addr in tqdm(sorted(gt_images.items())):
        assert stem in pred_images, f"No matching prediction for GT image: {stem}"
        gt_addr = str(gt_addr)
        pred_addr = str(pred_images[stem])

        # LDR jpg images are already gamma-encoded 8-bit; just normalize to [0, 1].
        gt_image = cv2.imread(gt_addr, cv2.IMREAD_UNCHANGED)
        gt_image = cv2.resize(gt_image, (512, 256), interpolation=cv2.INTER_LINEAR)
        gt_image = np.clip(gt_image.astype(np.float32) / 255.0, 0.0, 1.0)

        pred_image = cv2.imread(pred_addr, cv2.IMREAD_UNCHANGED)
        pred_image = cv2.resize(pred_image, (512, 256), interpolation=cv2.INTER_LINEAR)
        pred_image = np.clip(pred_image.astype(np.float32) / 255.0, 0.0, 1.0)

        cv2.imwrite('./test.png', (pred_image * 255).astype('uint8'))
        cv2.imwrite('./gt.png', (gt_image * 255).astype('uint8'))

        if not np.isnan(gt_image).any():
            gt_image_torch = torch.from_numpy(gt_image).permute(2, 0, 1)
            gts.append(gt_image_torch)
            pred_image_torch = torch.from_numpy(pred_image).permute(2, 0, 1)
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

    # output_path = os.path.join(args.data_dir, "ldr_fast_results.json")
    output_path = os.path.join(args.data_dir, "hdrnerf_fast_results.json")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved metrics to: {output_path}")
