#!/bin/bash
echo "Evaluation for linearization then nerf pipeline"
dataset_dir=/mnt/data/nerfstudio_ds/real_data/Henry_Gagnon_blue_et_rouge/Henry_Gagnon_blue_et_rouge_ns/
opensfm_dir=/mnt/data/nerfstudio_ds/real_data/Henry_Gagnon_blue_et_rouge/Henry_Gagnon_blue_et_rouge_openSFM

checkpoint_dir=outputs/Henry_Gagnon_blue_et_rouge_ns/pandora-nerfacto/2024-05-16_015020/
output_dir=renders/Henry_Gagnon_blue_et_rouge_ns-nerfacto/nerf_then_linearization/test_w_camera_optimization_2024-05-16_015020

opensfm_file=$opensfm_dir/reconstruction.json

echo "Creating the test set"
ns-process-data pandora-GT-HDR --data $dataset_dir --output-dir $dataset_dir --metadata $opensfm_file --checkpoint $checkpoint_dir

echo "Rendering for the well exposed"
ns-render camera-path --load-config $checkpoint_dir/config.yml --camera-path-filename $dataset_dir/GT_transforms.json --output-path $output_dir/well-expo/  --output-format images

echo "Rendering for the fast exposure"
ns-render camera-path --load-config $checkpoint_dir/config.yml --camera-path-filename $dataset_dir/GT_transforms.json --output-path $output_dir/fast-expo/  --output-format images --rendered_output_names rgb_fast

#conda init bash
conda activate everlight_new

echo "Combine two exposures"
python pandora_scripts/combine.py --well_dir $output_dir/well-expo/ --fast_dir $output_dir/fast-expo/ --out_dir $output_dir/pano/ --do_linearization --experiment_location $root_dir

conda activate nerfstudio-HDR

blender --background pandora_scripts/scene_new.blend -P pandora_scripts/hdr_blender.py -- $output_dir/pano/ $output_dir/render/