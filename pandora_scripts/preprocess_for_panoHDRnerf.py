import subprocess
import argparse
import os
import shutil


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True, default="/mnt/data/scene/", help="Directory containing everything")
    args = argparser.parse_args()

    data_dir = os.path.join(args.input_dir, "data")
    left_e1_dir = os.path.join(data_dir, "left_e1")
    
    left_e1_linearized_dir = os.path.join(data_dir, "left_e1_linearized/")
    if os.path.exists(left_e1_linearized_dir):
        shutil.rmtree(left_e1_linearized_dir)
    os.makedirs(left_e1_linearized_dir)

    left_e1_linearized_resized_dir = os.path.join(data_dir, "left_e1_linearized_resized/")
    if os.path.exists(left_e1_linearized_resized_dir):
        shutil.rmtree(left_e1_linearized_resized_dir)
    
    left_e1_ldr2hdr_dir = os.path.join(data_dir, "left_e1_ldr2hdr/")
    if os.path.exists(left_e1_ldr2hdr_dir):
        shutil.rmtree(left_e1_ldr2hdr_dir)
    os.makedirs(left_e1_ldr2hdr_dir)

    left_e1_ldr2hdr_high_res_dir = os.path.join(data_dir, "left_e1_ldr2hdr_high_res/")
    if os.path.exists(left_e1_ldr2hdr_high_res_dir):
        shutil.rmtree(left_e1_ldr2hdr_high_res_dir)
    os.makedirs(left_e1_ldr2hdr_high_res_dir)
    left_e1_ldr2hdr_high_res_left_e1_dir = os.path.join(left_e1_ldr2hdr_high_res_dir, "left_e1/")
    os.makedirs(left_e1_ldr2hdr_high_res_left_e1_dir)

    print("Linearizing the JPEGs...")
    subprocess.run(["conda", "run", "-n", "nerfstudio", "python", "pandora_scripts/linearize_jpgs.py", 
    "--data_dir", left_e1_dir, 
    "--out_dir", left_e1_linearized_dir, 
    ], check=True)

    shutil.copytree(left_e1_linearized_dir, left_e1_linearized_resized_dir)

    print("Resizing target_jpg_resize...")
    result = subprocess.run(["find", left_e1_linearized_resized_dir, "-iname", "*.png", "-exec", "convert", "{}", "-verbose", "-resize", "512x256>", "{}", ";"], capture_output=True, text=True)
    print(result)

    print("LDR2HDR Network on the data")
    subprocess.run(["conda", "run", "-n", "lanet_env_37", "OPENCV_IO_ENABLE_OPENEXR=true", "python", "/mnt/workspace/pandora/PanoHDR-NeRF/LANet/test.py", 
    "--ckpt", "/mnt/workspace/pandora/PanoHDR-NeRF/LANet/epoch_128.pt",
    "--test_dir", left_e1_linearized_resized_dir,
    "--output_dir", left_e1_ldr2hdr_dir, 
    ], check=True)
    
    print("Combining the LDRs and HDRs...")
    subprocess.run(["conda", "run", "-n", "nerfstudio", "python", "pandora_scripts/combine_LDR_HDR.py", 
    "--ldr_dir", left_e1_linearized_dir, 
    "--hdr_dir", left_e1_ldr2hdr_dir, 
    "--output_dir", left_e1_ldr2hdr_high_res_left_e1_dir, 
    ],  check=True)

    sfm_dir = os.path.join(args.input_dir, "sfm")
    reconstruction_file = os.path.join(sfm_dir, "reconstruction.json")

    scene_name = str(args.input_dir).split('/')[-2]
    output_pano_hdr_nerf_ns =  os.path.join(args.input_dir, scene_name + "_pano_hdr_nerf_ns/")
    if os.path.exists(output_pano_hdr_nerf_ns):
        shutil.rmtree(output_pano_hdr_nerf_ns)

    left_e1_mask_dir = os.path.join(left_e1_dir, "mask/")
    target_mask_dir = os.path.join(left_e1_ldr2hdr_high_res_left_e1_dir, "mask/")
    if os.path.exists(target_mask_dir):
        shutil.rmtree(target_mask_dir)
    shutil.copytree(left_e1_mask_dir, target_mask_dir)
    
    print("Making it ready for nerfstudio...")
    subprocess.run(["conda", "run", "-n", "nerfstudio", "ns-process-data", "aligned_pano2plane", 
    "--data", left_e1_ldr2hdr_high_res_dir, 
    "--output-dir", output_pano_hdr_nerf_ns,
    "--metadata", reconstruction_file, 
    "--is_metadata_from_openSFM",
    ],  check=True)