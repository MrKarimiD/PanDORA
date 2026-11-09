import os
from os import makedirs
from html4vision import Col, imagetable
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/GT/GT_exr')
    parser.add_argument('--gt_render_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/GT/GT_exr_renders')
    parser.add_argument('--pandora_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/metrics/images/pandora_cam_opt_on_lin')
    parser.add_argument('--pandora_render_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H/metrics/images/pandora_cam_opt_on_lin_renders')
    parser.add_argument('--out_dir', type=str, default='/mnt/data/nerfstudio_ds/real_data/coffeeroom_1102H')
    parser.add_argument("--name", type=str, required=True, default="unaligned", help="Name of the method, used for naming output directories")
    args = parser.parse_args()

    results_json = os.path.join(args.out_dir, 'metrics/results_table')
    results_json = os.path.join(results_json, args.name + '_results.json')
    with open(results_json, 'r') as file:
        metrics = json.load(file)
    img_keys = metrics[args.name].keys() - ['mean']
    
    gt = []
    gt_render = []
    out = []
    out_render = []

    mse = []
    si_rmse = []
    psnr = []
    angular = []

    for key in img_keys:
        img_name = key.replace('exr', 'png')

        mse.append(metrics[args.name][key]['mse'])
        si_rmse.append(metrics[args.name][key]['si_rmse'])
        psnr.append(metrics[args.name][key]['psnr'])
        angular.append(metrics[args.name][key]['angular'])
        
        gt.append( os.path.join(args.gt_dir, img_name) )
        gt_render.append( os.path.join(args.gt_render_dir, img_name) )
        out.append( os.path.join(args.pandora_dir, img_name) )
        out_render.append( os.path.join(args.pandora_render_dir, img_name) )
    
    cols = [
        Col('text', 'Name', list(img_keys)), # Col('id1', 'ID'), # 1-based indexing
        Col('img', 'GT Pano', gt),
        Col('img', 'GT Render', gt_render),
        Col('img', 'Pandora Output', out),
        Col('img', 'Pandora Render', out_render),
        Col('text', 'MSE', mse),
        Col('text', 'si-RMSE', si_rmse),
        Col('text', 'PSNR', psnr),
        Col('text', 'Angular', angular),
    ]
    

    # imagetable(cols, os.path.join(args.out_dir, 'results_' + args.name + '_index.html'), 'Pandora outputs', pathrep=(args.out_dir, './'), 
    #     imscale=0.1, 
    #     sort_style='materialize',
    #     sortable=True, 
    #     zebra=True, 
    #     sticky_header=True, 
    #     overlay_toggle=True
    #     )

    imagetable(cols, os.path.join(args.out_dir, 'results_' + args.name + '_index.html'), 'Pandora outputs', pathrep=(args.out_dir, './'), 
        sort_style='materialize',
        imsize=(300, 300),
        hori_center_img=True,
        sortable=True, 
        zebra=True, 
        sticky_header=True, 
        overlay_toggle=True
        )