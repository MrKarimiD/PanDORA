import subprocess
import argparse
import os
import shutil


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True, default="/mnt/data/scene/", help="Directory containing all the 4 videos")
    argparser.add_argument("--shutter_speed", type=float, required=True, default=0.003125)
    argparser.add_argument("--gt", action='store_true', default=False, help="Process GT images for metrics if flag is present")
    args = argparser.parse_args()

    subprocess.run(["python", "pandora_scripts/sync_and_trim.py", "--input_dir", args.input_dir], check=True)
    subprocess.run(["python", "pandora_scripts/mp4_videos_to_png_frames.py", "--input_dir", args.input_dir + "data/"], check=True)
    subprocess.run(["python", "pandora_scripts/mask_humans2.py", "--input_dir", args.input_dir + "data/", "--output_dir", args.input_dir + "data/masks"], check=True)
    subprocess.run(["python", "pandora_scripts/create_data_folder.py", "--input_dir", args.input_dir + "data/", "--shutter_speed", str(args.shutter_speed)], check=True)
    subprocess.run(["python", "pandora_scripts/create_sfm_folder.py", "--input_dir", args.input_dir], check=True)

    # For GT
    if args.gt:
        subprocess.run(["python", "pandora_scripts/debevec_bracket_merge.py", "--data_dir", args.input_dir + "GT/", "--experiment_location", args.input_dir + "data/"], check=True)
        subprocess.run(["python", "pandora_scripts/create_GT_folder.py", "--input_dir", args.input_dir + "GT/"], check=True)

        gt_dir = os.path.join(args.input_dir, "GT")
        gt_exr = os.path.join(gt_dir, "GT_exr")
        gt_exr_renders = os.path.join(gt_dir, "GT_exr_renders")

        print("Rendering GT images with Blender...")
        subprocess.run([
            "blender", "--background", "pandora_scripts/scene_new.blend",
            "-P", "pandora_scripts/hdr_blender.py", "--",
            os.path.join(gt_exr, ""), os.path.join(gt_exr_renders, "")
        ], check=True)

        print("Converting EXR panos to LDR visualization...")
        subprocess.run([
        "python", "pandora_scripts/tonemap.py",
        "--data_dir", gt_exr,
        "--out_dir", gt_exr
        ], check=True)
        subprocess.run([
            "python", "pandora_scripts/tonemap.py",
            "--data_dir", gt_exr_renders,
            "--out_dir", gt_exr_renders
        ], check=True)
        
        subprocess.run(["python", "pandora_scripts/gt_visualization.py", 
        "--gt_dir", gt_exr,
        "--gt_render_dir", gt_exr_renders,
        "--out_dir", args.input_dir
        ], check=True)
    
    subprocess.run(["python", "pandora_scripts/data_visualization.py", "--input_dir", args.input_dir + "data/"], check=True)
