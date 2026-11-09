#!/bin/bash
echo "Evaluation for linearization then nerf pipeline"

opensfm_dir=/mnt/data/nerfstudio_ds/real_data/my_apartment_corrected/my_apartment_openSFM
opensfm_file=$opensfm_dir/reconstruction.json

dataset_dir=/mnt/data/nerfstudio_ds/real_data/my_apartment_corrected/pano_hdr_nerf/pano_hdr_nerf_my_apt/

checkpoint_dir=outputs/pano_hdr_nerf_my_apt/panoHDR-nerfacto/2024-05-18_160257/
output_dir=/mnt/workspace/pandora/nerfstudio-HDR/renders/pano_hdr_nerf_my_apt


echo "Creating the test set"
ns-process-data pandora-GT-HDR --data $dataset_dir --output-dir $dataset_dir --metadata $opensfm_file --checkpoint $checkpoint_dir

echo "Rendering for the well exposed"
ns-render camera-path --load-config $checkpoint_dir/config.yml --camera-path-filename $dataset_dir/GT_transforms.json --output-path $output_dir/pano_gamma_compressed/  --output-format images --image-format exr

conda activate everlight_new

python pandora_scripts/un_gamma.py --data_dir $output_dir/pano_gamma_compressed/ --out_dir $output_dir/pano/

conda activate nerfstudio-HDR

blender --background pandora_scripts/scene_new_my_apt.blend -P pandora_scripts/hdr_blender.py -- $output_dir/pano/ $output_dir/render/