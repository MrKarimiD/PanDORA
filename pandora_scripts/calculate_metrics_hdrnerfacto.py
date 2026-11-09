import subprocess
import argparse
import os
import shutil
import sys


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, required=True, default="/mnt/data/scene/scene_ns/",
                           help="Directory containing input data (e.g. videos, GT, etc.)")
    argparser.add_argument("--checkpoint", type=str, required=True,
                           default="/mnt/workspace/pandora/nerfstudio-HDR/outputs/coffee_room2_ns/pandora-nerfacto/2025-01-22_194346/",
                           help="Directory containing training data (e.g. config.yml and checkpoint info)")
    argparser.add_argument("--metric_name", type=str, required=True, default="unaligned",
                           help="Name of the method, used for naming output directories")
    
    args = argparser.parse_args()

    if not args.input_dir.endswith('/'):
        args.input_dir += '/'

    metrics_dir = os.path.join(args.input_dir, "metrics")
    renders_dir = os.path.join(metrics_dir, "images")
    
    renders_well = os.path.join(renders_dir, args.metric_name)
    if os.path.exists(renders_well):
        shutil.rmtree(renders_well)
    
    renders_fast = os.path.join(renders_dir, args.metric_name + "_fast")
    if os.path.exists(renders_fast):
        shutil.rmtree(renders_fast)
    
    renders_lin = os.path.join(renders_dir, args.metric_name + "_lin")
    if os.path.exists(renders_lin):
        shutil.rmtree(renders_lin)
    
    renders_lin_HDR = os.path.join(renders_dir, args.metric_name + "_lin_HDR")
    if os.path.exists(renders_lin_HDR):
        shutil.rmtree(renders_lin_HDR)
    
    lin_renders = os.path.join(renders_dir, args.metric_name + "_lin_renders")
    if os.path.exists(lin_renders):
        shutil.rmtree(lin_renders)

    gt_dir = os.path.join(args.input_dir, "GT")
    gt_hdr = os.path.join(gt_dir, "GT_hdr")
    if os.path.exists(gt_hdr):
        shutil.rmtree(gt_hdr)
    gt_exr = os.path.join(gt_dir, "GT_exr")
    gt_exr_renders = os.path.join(gt_dir, "GT_exr_renders")
    if os.path.exists(gt_exr_renders):
        shutil.rmtree(gt_exr_renders)

    # 1. Get ground truth transformations:
    print("Running ns-process-data to get GT transformations...")
    subprocess.run([
        "ns-process-data", "pandora-GT-HDR",
        "--data", args.input_dir,
        "--output-dir", metrics_dir,
        "--metadata", os.path.join(args.input_dir, "sfm/reconstruction.json"),
        "--checkpoint", args.checkpoint
    ], check=True)

   # 2. Render panos at GT positions for well and fast exposures:
    print("Rendering well-exposed panos...")
    subprocess.run([
        "ns-render", "camera-path",
        "--load-config", os.path.join(args.checkpoint, "config.yml"),
        "--camera-path-filename", os.path.join(metrics_dir, "GT_transforms.json"),
        "--output-path", renders_well,
        "--output-format", "images"
    ], check=True)

    print("Rendering fast-exposed panos...")
    subprocess.run([
        "ns-render", "camera-path",
        "--load-config", os.path.join(args.checkpoint, "config.yml"),
        "--camera-path-filename", os.path.join(metrics_dir, "GT_transforms.json"),
        "--output-path", renders_fast,
        "--output-format", "images",
        "--rendered_output_names", "rgb_fast"
    ], check=True)

    # 3. Combine both well and fast exposed panos:
    print("Combining well and fast exposure renders...")
    subprocess.run([
        "python", "pandora_scripts/combine.py",
        "--well_dir", renders_well + "/",
        "--fast_dir", renders_fast + "/",
        "--out_dir", renders_lin + "/",
        "--experiment_location", os.path.join(args.input_dir, "data") + "/",
        "--do_linearization"
    ], check=True)

    # 4. Rename the render files:
    print("Step 4a: Renaming render filenames for consistency between GT and renders...")
    render_files = sorted([f for f in os.listdir(renders_lin) if f.endswith('.exr')])
    print("Render files found:", render_files)
    total = len(render_files)
    number_strings = [str(i) for i in range(1, total + 1)]
    lex_order = sorted(number_strings)
    for old_path, num in zip(render_files, lex_order):
        new_filename = f"GT{num}.exr"
        new_path = os.path.join(renders_lin, new_filename)
        old_path = os.path.join(renders_lin, old_path)
        shutil.move(old_path, new_path)
        print(f"Renamed {os.path.basename(old_path)} to {new_filename}")
    
    print("Step 4b: Verifying filename consistency between GT and renders...")
    try:
        gt_files = set(f for f in os.listdir(gt_exr) if f.endswith('.exr'))
        render_files = set(f for f in os.listdir(renders_lin) if f.endswith('.exr'))
        missing_in_renders = gt_files - render_files
        extra_in_renders = render_files - gt_files
        if missing_in_renders or extra_in_renders:
            if missing_in_renders:
                print("Error: The following GT files are missing in the renders directory:", missing_in_renders)
            if extra_in_renders:
                print("Error: There are extra files in the renders directory not found in GT:", extra_in_renders)
            sys.exit(1)
        else:
            print("All GT filenames are present in the renders directory.")
    except Exception as e:
        print("Error during filename verification:", e)
        sys.exit(1)
    
    # 5. Calculate PSNR, SSIM and LPIPS:
    print("Calculating LDR metrics (PSNR, SSIM, LPIPS)...")
    subprocess.run([
        "python", "pandora_scripts/ldr_res.py",
        "--gt_dir", gt_exr + "/",
        "--data_dir", renders_lin + "/"
    ], check=True)

    # 6. Generate GT and result renders with Blender:
    expected_render_dir = gt_exr_renders
    rendered_files_exist = any(
        f.endswith('.exr') for f in os.listdir(expected_render_dir)
    ) if os.path.exists(expected_render_dir) else False

    if not rendered_files_exist:
        print("Rendering GT images with Blender...")
        subprocess.run([
            "blender", "--background", "pandora_scripts/scene_new.blend",
            "-P", "pandora_scripts/hdr_blender.py", "--",
            os.path.join(gt_exr, ""), os.path.join(gt_exr_renders, "")
        ], check=True)
    else:
        print("Skipping Blender render — output files already exist.")

    print("Rendering result images with Blender...")
    subprocess.run([
        "blender", "--background", "pandora_scripts/scene_new.blend",
        "-P", "pandora_scripts/hdr_blender.py", "--",
        renders_lin + "/", lin_renders + "/"
    ], check=True)

    # 7. Calculate si-RMSE, RMSE, RGB angular error and PSNR (LDR rendered):
    print("Calculating additional metrics (si-RMSE, RMSE, angular error, PSNR)...")
    subprocess.run([
        "python", "pandora_scripts/res_table_1.py",
        "--results_dataset_roots", lin_renders + "/",
        "--gt_dataset_root", gt_exr_renders + "/",
        "--output_folder", metrics_dir + "/results_table/",
        "--dataset_names", args.metric_name
    ], check=True)

    # 8. Convert EXR panos to HDR format:
    print("Converting EXR panos to HDR...")
    subprocess.run([
        "python", "pandora_scripts/exr2hdr.py",
        "--hdr_dir", renders_lin,
        "--output_dir", renders_lin_HDR
    ], check=True)
    subprocess.run([
        "python", "pandora_scripts/exr2hdr.py",
        "--hdr_dir", gt_exr,
        "--output_dir", gt_hdr
    ], check=True)

    # 9. Convert EXR panos to LDR format:
    print("Converting EXR panos to LDR visualization...")
    subprocess.run([
        "python", "pandora_scripts/tonemap.py",
        "--data_dir", renders_lin,
        "--out_dir", renders_lin
    ], check=True)
    subprocess.run([
        "python", "pandora_scripts/tonemap.py",
        "--data_dir", gt_exr,
        "--out_dir", gt_exr
    ], check=True)
    subprocess.run([
        "python", "pandora_scripts/tonemap.py",
        "--data_dir", lin_renders,
        "--out_dir", lin_renders
    ], check=True)
    subprocess.run([
        "python", "pandora_scripts/tonemap.py",
        "--data_dir", gt_exr_renders,
        "--out_dir", gt_exr_renders
    ], check=True)

     # 9. Convert EXR panos to LDR format:
    print("Preparing the HTML visualization...")
    subprocess.run([
        "python", "pandora_scripts/results_visualization.py",
        "--gt_dir", gt_exr,
        "--gt_render_dir", gt_exr_renders,
        "--pandora_dir", renders_lin,
        "--pandora_render_dir", lin_renders,
        "--out_dir", args.input_dir,
        "--name", args.metric_name
    ], check=True)
