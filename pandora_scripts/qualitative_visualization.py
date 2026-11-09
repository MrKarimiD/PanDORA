import os
from os import makedirs
from html4vision import Col, imagetable
import argparse
import json


def validate_folder(path_str):
    if not os.path.isdir(path_str):
        raise FileNotFoundError(f"The following path do not exists : {path_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/GT/GT_exr')
    args = parser.parse_args()

    gt_dir = os.path.join(args.data_dir, 'GT/GT_exr')
    gt_render_dir = os.path.join(args.data_dir, 'GT/GT_exr_renders')

    metrics_dir = os.path.join(args.data_dir, 'metrics')
    images_dir = os.path.join(metrics_dir, 'images')

    # Pandora directories
    pandora_dir = os.path.join(images_dir, 'pandora_cam_opt_on_aligned_lin')
    validate_folder(pandora_dir)
    pandora_render_dir = os.path.join(images_dir, 'pandora_cam_opt_on_aligned_lin_renders')
    validate_folder(pandora_render_dir)

    # HDR-Nerfacto directories
    hdrnerf_dir = os.path.join(images_dir, 'hdr_nerfacto_lin')
    validate_folder(hdrnerf_dir)
    hdrnerf_render_dir = os.path.join(images_dir, 'hdr_nerfacto_lin_renders')
    validate_folder(hdrnerf_render_dir)

    # PanoHDR-Nerfacto directories
    pano_hdr_nerf_dir = os.path.join(images_dir, 'panoHDRNerf_lin')
    validate_folder(pano_hdr_nerf_dir)
    pano_hdr_nerf_render_dir = os.path.join(images_dir, 'panoHDRNerf_lin_renders')
    validate_folder(pano_hdr_nerf_render_dir)

    # LDR-Nerfacto directories
    ldrnerf_dir = os.path.join(images_dir, 'ldr_cam_opt_on_lin')
    validate_folder(ldrnerf_dir)
    ldrnerf_render_dir = os.path.join(images_dir, 'ldr_cam_opt_on_lin_renders')
    validate_folder(ldrnerf_render_dir)
    
    cols = [
        Col('id1', 'ID'),
        Col('img', 'GT Pano', os.path.join(gt_dir, '*.png')),
        Col('img', 'GT Render',  os.path.join(gt_render_dir, '*.png')),
        # Col('img', 'Pandora Output', os.path.join(pandora_dir, '*.png')),
        Col('img', 'Pandora Render', os.path.join(pandora_render_dir, '*.png')),
        # Col('img', 'LDR-NeRFacto Output', os.path.join(ldrnerf_dir, '*.png')),
        Col('img', 'LDR-NeRFacto Render', os.path.join(ldrnerf_render_dir, '*.png')),
        # Col('img', 'PanoHDR-NeRFacto Output', os.path.join(pano_hdr_nerf_dir, '*.png')),
        Col('img', 'PanoHDR-NeRFacto Render', os.path.join(pano_hdr_nerf_render_dir, '*.png')),
        # Col('img', 'HDR-NeRFacto Output', os.path.join(hdrnerf_dir, '*.png')),
        Col('img', 'HDR-NeRFacto Render', os.path.join(hdrnerf_render_dir, '*.png')),
    ]
    

    imagetable(cols, os.path.join(args.data_dir, 'results_qualitative_index.html'), 'Pandora outputs', pathrep=(args.data_dir, './'), 
        sort_style='materialize',
        imsize=(300, 300),
        hori_center_img=True,
        sortable=True, 
        zebra=True, 
        sticky_header=True, 
        overlay_toggle=True
        )