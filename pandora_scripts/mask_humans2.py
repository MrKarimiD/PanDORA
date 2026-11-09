from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import argparse
import os

import cv2
import numpy as np
import torch
from lang_sam import LangSAM
from PIL import Image

# import some common detectron2 utilities
from tqdm import tqdm
from envmap import EnvironmentMap, rotation_matrix


FRAME_SUBFOLDERS_CONFIG = [
    {
        'name': 'left_sfm',
        'flip_k': -1,
        'mask_person': True,
        'custom_mask': './stick_masks/left_mask.png'
    },
    {
        'name': 'right_sfm',
        'flip_k': 1,
        'mask_person': True,
        'custom_mask': './stick_masks/right_mask.png'
    },
    {
        'name': 'left_e1',
        'flip_k': -1,
        'mask_person': True,
        'custom_mask': './stick_masks/left_mask.png'
    },
    {
        'name': 'right_e2',
        'flip_k': 1,
        'mask_person': False,
        'custom_mask': './stick_masks/right_mask.png'
    }
]

PERSON_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Directory containing all the 4 directories containing frames')
    parser.add_argument('--output_dir', type=str, help='Directory to save masked frames')
    
    args = parser.parse_args()
    out = args.output_dir
    # out = f'{args.output_dir}/overlay'
    if(not os.path.isdir(out)):
        os.makedirs(out)
    # stick = cv2.imread(args.stick)
    for folder in FRAME_SUBFOLDERS_CONFIG:
        assert os.path.isdir(os.path.join(args.input_dir, folder['name'])), f"Folder {folder['name']} does not exist"
    
    # imgs = os.listdir(args.input_dir)
    
    model = LangSAM()

    for folder in FRAME_SUBFOLDERS_CONFIG:
        print(f"Processing folder {folder['name']}")
        # imgs = os.listdir(os.path.join(args.input_dir, folder['name']))
        # filter png only
        imgs = [f for f in os.listdir(os.path.join(args.input_dir, folder['name'])) if f.endswith(".png")]
        custom_mask = cv2.imread(folder['custom_mask'])
        custom_mask = cv2.cvtColor(custom_mask, cv2.COLOR_BGR2GRAY)
        custom_mask = custom_mask.astype(np.bool_)
        for i, file in enumerate(tqdm(imgs)):
            original_image_pil = Image.open(os.path.join(args.input_dir, folder['name'], file)).convert("RGB")
            original_image_np = np.array(original_image_pil)

            original_image = EnvironmentMap(os.path.join(args.input_dir, folder['name'], file), 'latlong')
            dcm = rotation_matrix(azimuth=0, elevation=0, roll=np.pi/2)
            original_image.rotate(dcm)
            image_np_rot = (original_image.data * 255).astype(np.uint8)

            if folder['mask_person']:
                text_prompt = "person head torso arms legs feet shoes"
                # 90 degrees rotat
                # image_np_rot = original_image_np # np.rot90(original_image_np, k=folder['flip_k'])
                masks, boxes, phrases, logits = model.predict(Image.fromarray(image_np_rot), text_prompt)
                masks = masks.detach().cpu().numpy()
                boxes = boxes.detach().cpu().numpy()
                if masks.size != 0:
                    # masks = np.rot90(masks, k=-folder['flip_k'], axes=(1, 2))

                    photographer_mask = np.zeros((masks.shape[1], masks.shape[2], 1), dtype=np.uint8)
                    for i in range(boxes.shape[0]):
                        color = (255, 255, 255)   # White color in BGR
                        cv2.rectangle(photographer_mask, (int(boxes[i, 0]), int(boxes[i, 1])), (int(boxes[i, 2]), int(boxes[i, 3])), (255, 255, 255), cv2.FILLED)

                    
                    # mask_skylibs = EnvironmentMap(np.transpose(masks, (1, 2, 0)).astype(np.uint8), 'latlong')  
                    mask_skylibs = EnvironmentMap(photographer_mask, 'latlong')  
                    inv_dcm = rotation_matrix(azimuth=0, elevation=0, roll=-np.pi/2)
                    mask_skylibs.rotate(inv_dcm)
                    masks = mask_skylibs.data
                    masks = np.transpose(masks, (2, 0, 1))
                    # aggregate masks
                    masks = np.max(masks, axis=0)
                else:
                    masks = np.zeros((original_image_pil.height, original_image_pil.width))

            else:
                masks = np.zeros((original_image_pil.height, original_image_pil.width))
                
            # custom mask (logical or)
            masks = masks.astype(np.bool_)
            masks = masks | custom_mask

            # dilate mask
            masks = masks.astype(np.uint8)
            masks = cv2.dilate(masks, PERSON_DILATE_KERNEL)

            # save masked image
            out_image_np = original_image_np * (1 - masks)[:, :, None] + masks[:, :, None] * 255
            out_image_np = out_image_np.astype(np.uint8)
            out_masked_file_path = os.path.join(args.output_dir, folder['name'], file) + ".masked.png"
            os.makedirs(os.path.dirname(out_masked_file_path), exist_ok=True)
            out_masked_img = Image.fromarray(out_image_np)
            out_masked_img.save(out_masked_file_path)

            # masks to image
            out_masked_file_path = os.path.join(args.output_dir, folder['name'], file)
            os.makedirs(os.path.dirname(out_masked_file_path), exist_ok=True)
            # invert mask
            masks = 1 - masks
            out_mask = Image.fromarray(masks * 255)
            out_mask.save(out_masked_file_path)

