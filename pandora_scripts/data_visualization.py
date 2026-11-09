from os import makedirs
from html4vision import Col, imagetable
import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/GT/GT_exr')
    args = parser.parse_args()

    left_e1_dir = args.input_dir + 'left_e1/'
    left_e1_mask_dir = args.input_dir + 'masks/left_e1/'
    right_e2_dir = args.input_dir + 'right_e2/'
    right_e2_mask_dir = args.input_dir + 'masks/right_e2/'
    
    imgs = [f for f in os.listdir(left_e1_dir) if f.endswith(".png")]

    left_e1 = []
    left_mask = []
    right_e2 = []
    right_mask = []
    for img in imgs:
        left_e1.append( os.path.join(left_e1_dir, img) )
        left_mask.append( os.path.join(left_e1_mask_dir, img + '.masked.png') )

        right_e2.append( os.path.join(right_e2_dir, img) )
        right_mask.append( os.path.join(right_e2_mask_dir, img + '.masked.png') )

    cols = [
        Col('id1', 'ID'),
        Col('text', 'Name', imgs),
        Col('img', 'Left Pano', left_e1),
        Col('img', 'Left Mask', left_mask),
        Col('img', 'Right Pano', right_e2),
        Col('img', 'Right Mask', right_mask),
    ]

    imagetable(
        cols, 
        args.input_dir + 'data_index.html', 
        'NeRF sequence', 
        pathrep=(args.input_dir, './'), 
        sort_style='materialize',
        imscale=0.1, 
        zebra=True, 
        sticky_header=True, 
        overlay_toggle=True
        )

    left_sfm_dir = args.input_dir + 'left_sfm/'
    left_sfm_mask_dir = args.input_dir + 'masks/left_sfm/'
    right_sfm_dir = args.input_dir + 'right_sfm/'
    right_sfm_mask_dir = args.input_dir + 'masks/right_sfm/'
    
    imgs = [f for f in os.listdir(left_sfm_dir) if f.endswith(".png")]

    left_sfm = []
    left_mask = []
    right_sfm = []
    right_mask = []
    for img in imgs:
        left_sfm.append( os.path.join(left_sfm_dir, img) )
        left_mask.append( os.path.join(left_sfm_mask_dir, img + '.masked.png') )

        right_sfm.append( os.path.join(right_sfm_dir, img) )
        right_mask.append( os.path.join(right_sfm_mask_dir, img + '.masked.png') )

    cols = [
        Col('id1', 'ID'),
        Col('text', 'Name', imgs),
        Col('img', 'Left Pano', left_sfm),
        Col('img', 'Left Mask', left_mask),
        Col('img', 'Right Pano', right_sfm),
        Col('img', 'Right Mask', right_mask),
    ]

    imagetable(
        cols, 
        args.input_dir + 'sfm_index.html', 
        'SFM sequence', 
        pathrep=(args.input_dir, './'), 
        sort_style='materialize',
        imscale=0.1, 
        zebra=True, 
        sticky_header=True, 
        overlay_toggle=True
        )