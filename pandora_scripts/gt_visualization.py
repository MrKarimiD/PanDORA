import os
from os import makedirs
from html4vision import Col, imagetable
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/GT/GT_exr')
    parser.add_argument('--gt_render_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/GT/GT_exr_renders')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H')
    args = parser.parse_args()

    gt = []
    gt_render = []

    imgs = [f for f in os.listdir(args.gt_dir) if f.endswith(".png")]

    for key in imgs:
        img_name = key#.replace('exr', 'png')

        gt.append( os.path.join(args.gt_dir, img_name) )
        gt_render.append( os.path.join(args.gt_render_dir, img_name) )
    
    cols = [
        Col('text', 'Name', list(imgs)), # Col('id1', 'ID'), # 1-based indexing
        Col('img', 'GT Pano', gt),
        Col('img', 'GT Render', gt_render),
        ]
    
    imagetable(cols, os.path.join(args.out_dir, 'gt_index.html'), 'Pandora outputs', pathrep=(args.out_dir, './'), 
        sort_style='materialize',
        imsize=(300, 300),
        hori_center_img=True,
        sortable=True, 
        zebra=True, 
        sticky_header=True, 
        overlay_toggle=True
        )