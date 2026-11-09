
export scene_name=my_apartment_ns
export output_dir=renders/${scene_name}/pandora-nerfacto/video_images
export root_dir=data/${scene_name}
export model_path=outputs/${scene_name}/pandora-nerfacto/2024-05-16_212421/config.yml

# camera paths
# video ground: data/Henry_Gagnon_blue_et_rouge_ns/camera_paths/2024-05-16_015020.json
# still bench: 2024-05-16_015020_render
# pespective cam: 2024-05-16_015020_render3

# meeting room
# export camera_path_name=2024-05-16_201826_persp
# export camera_path_name=2024-05-16_201826_envmap

# my_apartment
# export camera_path_name=2024-05-16_212421_persp
# export camera_path_name=2024-05-16_212421_env1
# export camera_path_name=2024-05-16_212421_env2
# export camera_path_name=2024-05-16_212421_env3
# export camera_path_name=2024-05-16_212421_env4
export camera_path_name=2024-05-16_212421_env5
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/well-expo/  --output-format images
ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/fast-expo/  --output-format images  --rendered_output_names rgb_fast

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir

# blender --background video_rendering_code/scene_new.blend -P video_rendering_code/hdr_blender.py -- $output_dir/${camera_path_name}/combined/ $output_dir/${camera_path_name}/render/




#########

export camera_path_name=2024-05-16_212421_env1
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir


export camera_path_name=2024-05-16_212421_env2
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir




export camera_path_name=2024-05-16_212421_env3
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir




export camera_path_name=2024-05-16_212421_env4
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir
export camera_path_name=2024-05-16_212421_env5
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

echo "Combine two exposures"
python video_rendering_code/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir


####

# meeting room teaser


export scene_name=meeting_room_ns
export output_dir=renders/${scene_name}/pandora-nerfacto/video_images
export root_dir=data/${scene_name}
export model_path=outputs/${scene_name}/pandora-nerfacto/2024-05-16_201826/config.yml

# meeting room
# export camera_path_name=2024-05-16_201826_persp
# export camera_path_name=2024-05-16_201826_persp2

#export camera_path_name=2024-05-16_201826_env1
# export camera_path_name=2024-05-16_201826_env2
# export camera_path_name=2024-05-16_201826_env3
# export camera_path_name=2024-05-16_201826_env4
# export camera_path_name=2024-05-16_201826_env5
# export camera_path_name=2024-05-16_201826_env6
# export camera_path_name=2024-05-16_201826_env7
export camera_path_name=2024-05-16_201826_env8

export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/well-expo/  --output-format images
ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/fast-expo/  --output-format images  --rendered_output_names rgb_fast

echo "Combine two exposures"
python pandora_scripts/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir

# Fig. 5 - ABO objects

export scene_name=LVSN_LAB_OpenSFM
export output_dir=renders/${scene_name}/pandora-nerfacto/video_images
export root_dir=data/${scene_name}
export model_path=outputs/${scene_name}/pandora-nerfacto/step_2/config.yml

# export camera_path_name=persp1
export camera_path_name=persp2
# export camera_path_name=env1
# export camera_path_name=env2
# export camera_path_name=env3

export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json

ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/well-expo/  --output-format images
ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/fast-expo/  --output-format images  --rendered_output_names rgb_fast

echo "Combine two exposures"
python pandora_scripts/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir

######### Render videos ############

# LVSN lab upstairs


export scene_name=lab_downstairs_ns

export model_type=hdr-nerfacto
# export model_type=ldr-nerfacto
# export model_type=panoHDR-nerfacto
# export model_type=pandora-nerfacto

# export model_path=outputs/${scene_name}/${model_type}/2024-05-18_143132/config.yml
# export model_path=outputs/${scene_name}/pandora-nerfacto/step_1/config.yml
# export model_path=outputs/${scene_name}/${model_type}/2024-05-18_145217/config.yml
export model_path=outputs/${scene_name}/${model_type}/2024-05-18_012547/config.yml

export output_dir=renders/${scene_name}/${model_type}/video_images


# export camera_path_name=video3
export camera_path_name=video4
export camera_path=data/${scene_name}/camera_paths/${camera_path_name}.json
# export camera_path=data/LVSN_LAB_OpenSFM_pano_hdr_nerf/camera_paths/${camera_path_name}.json

ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/well-expo/  --output-format images
ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/fast-expo/  --output-format images  --rendered_output_names rgb_fast

# panoHDR-nerfacto
# ns-render camera-path --load-config $model_path --camera-path-filename $camera_path --output-path $output_dir/${camera_path_name}/pano_gamma_compressed/  --output-format images --image-format exr
# python pandora_scripts/un_gamma.py --data_dir $output_dir/${camera_path_name}/pano_gamma_compressed/ --out_dir $output_dir/${camera_path_name}/combined/

echo "Combine two exposures"
python pandora_scripts/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --fast_dir $output_dir/${camera_path_name}/fast-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir
# python pandora_scripts/combine.py --well_dir $output_dir/${camera_path_name}/well-expo/ --out_dir $output_dir/${camera_path_name}/combined/ --do_linearization --experiment_location $root_dir


# for Yannick
cd $output_dir/${camera_path_name}
zip -rq combined_${scene_name}_${model_type}.zip combined/*
cd -

python video_rendering_code/upload_online.py $output_dir/${camera_path_name}/combined_${scene_name}_${model_type}.zip tmpForYannick hdrdb-public




blender --background video_rendering_code/scene_new.blend -P video_rendering_code/hdr_blender.py -- $output_dir/${camera_path_name}/combined/ $output_dir/${camera_path_name}/render/

mkdir $output_dir/${camera_path_name}/render_tonemapped
python pandora_scripts/tonemap.py --data_dir $output_dir/${camera_path_name}/render/ --out_dir $output_dir/${camera_path_name}/render_tonemapped
python pandora_scripts/mp4_maker.py --input_Addr $output_dir/${camera_path_name}/render_tonemapped --out_file $output_dir/${camera_path_name}/video_render.mp4

##

export scene_name=meeting_room_ns

export model_path=outputs/${scene_name}/pandora-nerfacto/2024-05-16_201826/config.yml

