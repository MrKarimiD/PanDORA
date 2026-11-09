#!/bin/bash
echo "Evaluation for linearization then nerf pipeline"
root_dir=/mnt/data/nerfstudio_ds/real_data/Henry_Gagnon_blue_et_rouge
dataset_dir=$root_dir/Henry_Gagnon_blue_et_rouge_ns/

opensfm_dir=$root_dir/Henry_Gagnon_blue_et_rouge_openSFM
opensfm_file=$opensfm_dir/reconstruction.json

checkpoint_dir=outputs/Henry_Gagnon_blue_et_rouge_ns/pandora-nerfacto/2024-05-16_030223/
output_dir=renders/Henry_Gagnon_blue_et_rouge_ns-nerfacto/nerf_then_linearization/2024-05-16_030223_output

#echo "Creating the test set"
#ns-process-data pandora-GT-HDR --data $dataset_dir --output-dir $dataset_dir --metadata $opensfm_file --checkpoint $checkpoint_dir

#echo "Rendering for the well exposed"
#ns-render camera-path --load-config $checkpoint_dir/config.yml --camera-path-filename $dataset_dir/GT_transforms.json --output-path $output_dir/well-expo/  --output-format images --image-format exr

#echo "Rendering for the fast exposure"
#ns-render camera-path --load-config $checkpoint_dir/config.yml --camera-path-filename $dataset_dir/GT_transforms.json --output-path $output_dir/fast-mu-law_compressed/  --output-format images --image-format exr --rendered_output_names rgb_fast

echo "un-mu-law the fast exposure"
#conda init bash
conda activate everlight_new
python pandora_scripts/un_mu_law.py --data_dir $output_dir/fast-mu-law_compressed/ --out_dir $output_dir/fast-expo/ --is_jpg

echo "Combine two exposures"
python pandora_scripts/combine.py --well_dir $output_dir/well-expo/ --fast_dir $output_dir/fast-expo/ --out_dir $output_dir/pano/ --experiment_location $root_dir

conda activate nerfstudio-HDR

blender --background pandora_scripts/scene_new.blend -P pandora_scripts/hdr_blender.py -- $output_dir/pano/ $output_dir/render/