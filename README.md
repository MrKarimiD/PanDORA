# nerfstudio-HDR — How it works?

This repository supports two HDR NeRF pipelines that share the same capture,
pre-processing, and evaluation workflow, and differ only in the training and
metrics steps:

- **PanDORA** — two-stage training with fast/well exposure alignment.
- **HDR-Nerfacto (`hdr-nerfacto`)** — single-stage training.

Steps that are identical for both methods are written once below. Where the two
methods diverge (setup branch, training, metrics) both variants are shown.

## 0. Setup

> Tip: Open this folder in the VS Code **Dev Container** extension.

```
tmux new -s nerf_studio

docker run -it --rm -v /gel/usr/{USER_NAME}:/mnt/workspace/ -v /home-local2/{USER_NAME}.extra.nobkp/:/mnt/data/ --gpus '"device=0"' -p 8008:8008 -p 6006:6006 lantern_docker:latest /bin/bash

cd /mnt/workspace/lantern/nerfstudio-HDR

git clone https://github.com/MrKarimiD/nerfstudio-HDR.git
conda activate nerfstudio-HDR
export PYTHONPATH=$pwd:$PYTHONPATH
pip install scikit-surgerycore pydub skylibs piq OpenEXR Imath equilib
```

**Select the branch for your method:**

- **HDR-Nerfacto** — check out the `hdr-nerfacto` branch:
    ```
    git checkout -t remotes/origin/hdr-nerfacto
    ```
- **Pandora** — use the default branch (no checkout needed).

Then finish the install:

```
pip install -e .
ns-install-cli

pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
```

If `lang-sam` is not installed, do the following:
```
cd ../lang-segment-anything
pip install .
cd ../nerfstudio-HDR
```

If there is an issue with `equilib`, do the following:
```
pip uninstall equilib
cd ../equilib
python setup.py develop
cd ../nerfstudio-HDR
```

## 1. Data acquisition & pre-processing

1. Setup the apparatus (stick with two Ricoh cameras)
    > Cameras:
    >
    > **Camera 1**: left, well-exposed, serial: YN14100695 \
    > **Camera 3**: right, fast-exposed (under-exposed), serial: YN14111000

    - To connect the other camera:
        - Turn wifi off
        - Close the app
        - Turn wifi on
        - Open the app
        - Connect the other camera
    - Camera setup:
        - Install Ricoh Theta app
        - Connect the camera to the app
        - Change camera parameters: select manual (bottom right)
            - First time: turn on "CT Settings" in Shooting settings
            - Apreture: 2.1
            - Shutter speed: determined below
            - ISO: 800
            - WB: 3500
    - Make sure the cameras point up to avoid having lights in the seam.
    - Make sure to film up and down to get good coverage

2. First capture with well-exposed cameras.
    - Change shutter speed to get a well-exposed image. **Note the shutter speed.**
    - Start the video on one camera. Connect to the other camera following the steps above and start the video on the second camera.
    - Clap at the beginning and at the end to use for synchronization.
    - Go around the scene with the two cameras well-exposed.

3. Second capture with Camera 1 (left) well-exposed and Camera 2 (right) fast-exposed. Change shutter speed of Camera 2 to 1/25000 (fastest exposure) and repeat the steps above

    If calculating **METRICS**: Capture GT HDR brakets
    - Go to settings and change the camera mode shooting method to multiple brackets
    - Set the camera parameters:
        - Apreture: 2.1
        - Shutter speed: Calculate the 11 shutter speeds and select the closest ones on the app in order from smallest to hightes
            ```
            python lantern_scripts/calculate_exposures.py --min 0.00004 --max 0.004
            ```
        - ISO: 800
        - WB: 3500
    - Take 10 different locations

4. Import the files on computer. If on a mac, use Ricoh Theta File Transfer for Mac

5. Get the equirectangular videos by processing them with the Ricoh Theta computer app.
    - Drag and drop the video file into the app window.
    - Make sure both boxes are unchecked.
    - Name files as fallows
        - Camera 1:
            - First capture: left_sfm
            - Second capture: left_e1
        - Camera 2 (fast exposed):
            - First capture: right_sfm
            - Second capture: right_e2

    If calculating **METRICS**: Group GT brackets under GT folder like so
        GT/
            GT1/
                001.jpg
                002.jpg
                ...
            GT2/
                ...
            ...

6. Transfer files on lab machine.

    ```
    scp -r /path/to/source/file /path/to/destination/
    ```

7. Process the videos for OpenSFM. Input the right shutter speed noted above.

    If not calculating metrics:
    ```
    python lantern_scripts/process_videos_for_sfm.py --input_dir /mnt/data/scene/ --shutter_speed 0.004
    ```

    If calculating **METRICS**:
    ```
    python lantern_scripts/process_videos_for_sfm.py --input_dir /mnt/data/scene/ --shutter_speed 0.004 --gt
    ```
    - Resize GT_exr and GT_jpg images to 3840x1920 with GNU Image Manipulation Program
        - Go to Image > Scale Image
        - Transfer those back on lab machine in th GT folder of the scene naming them GT_jpg_small and GT_exr_small
    - Add GT_jpg images into sfm/images/ with their appropriate mask (checkout stick_masks folder)

## 2. Process data using OpenSFM

```
bin/opensfm_run_all /mnt/data/scene/sfm/
```

To view results:
```
python3 viewer/server.py -d /mnt/data/scene/sfm/
```
Check that the camera positions make sense and that the number of clusters is low.

## 3. Process data for NeRF

```
ns-process-data lantern-openSFM  --data /mnt/data/scene/data/  --output-dir /mnt/data/scene/scene_ns/ --metadata  /mnt/data/scene/sfm/reconstruction.json
```

For processing existing dataset that used colmap for camera positions:
```
ns-process-data images --data /mnt/data/garden/images --output-dir /mnt/data/garden_processed --skip-colmap --skip-image-processing --colmap-model-path /mnt/data/garden/sparse/0
```

## 4. Train NeRF

Pick the method you set up in Step 0.

### Option A — PanDORA

1. Run step 1 of PanDORA.

    ```
    ns-train PanDORA --data /mnt/data/scene/scene_ns/ --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 --pipeline.model.lantern_steps 1 --pipeline.datamanager.pixel-sampler.lantern_steps 1 --max-num-iterations 60000
    ```

    1.1. For better alignment between the well and the fast exposed images, run the following command.
    ```
    python lantern_scripts/align_fast_to_well.py --input_dir /mnt/data/scene/scene_ns/ --config outputs/scene_ns/PanDORA/2024-08-07_143013/config.yml
    ```

2. Run step 2 of PanDORA.

    ```
    ns-train PanDORA --data /mnt/data/scene/scene_ns/ --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800 --pipeline.model.lantern_steps 2 --pipeline.datamanager.pixel-sampler.lantern_steps 2 --load-dir /mnt/workspace/lantern/nerfstudio-HDR/outputs/scene_ns/PanDORA/2024-06-10_155559/nerfstudio_models --pipeline.model.apply_mu_law False --max-num-iterations 120000
    ```

### Option B — HDR-Nerfacto (`hdr-nerfacto`)

1. Run hdr-nerfacto.

    ```
    ns-train hdr-nerfacto --data /mnt/data/nerfstudio_ds/real_data/amphitheatre_1112_2/amphitheatre_ns/ --viewer.websocket-port 8008 --pipeline.datamanager.train-num-images-to-sample-from 1800  --max-num-iterations 120000 --viewer.websocket-port 8008
    ```

## 5. View results

To use the interface:
```
ns-viewer --load-config outputs/scene_ns/PanDORA/2024-06-06_191338/config.yml --viewer.websocket-port 8008
```
- For the right side up:
    - Go to SCENE sub-menu.
    - Clic on RESET UP DIRECTION.
- To render a video with the interface command:
    - Go to RENDER sub-menu.
    - Clic on ADD CAMERA to manualy set camera view points.
    - Clic on RENDER to get the command.
        ```
        ns-render camera-path --load-config outputs/scene_ns/PanDORA/2024-06-06_191338/config.yml --camera-path-filename /mnt/data/scene/scene_ns/camera_paths/2024-06-06_191338.json --output-path /mnt/data/scene/renders/2024-06-06_191338.mp4
        ```
    - To render a video fast-exposed: add --rendered-output-names rgb_fast
        ```
        ns-render camera-path --load-config outputs/scene_ns/PanDORA/2024-06-06_191338/config.yml --camera-path-filename /mnt/data/scene/scene_ns/camera_paths/2024-06-06_191338.json --output-path /mnt/data/scene/renders/2024-06-06_191338.mp4 --rendered-output-names rgb_fast
        ```

To get evaluation images (well-exposed):
```
ns-eval --load-config=outputs/scene_ns/PanDORA/2024-06-17_152246/config.yml --output-path=output.json --render_output_path=/mnt/data/scene/eval
```

- To get fast-exposed evaluation images, change these two lines in nerfstudio/lantern/model.py:
    ```
    line 238: if batch["exposure"] != 1.0: # change to == for testing fast-exposed
    line 242: return None, None # comment for testing fast-exposed
    ```

- To get evaluation images and tensorboard:
    ```
    ns-eval --load-config=outputs/scene_ns/PanDORA/2024-06-20_142413/config.yml --output-path=output.json --render_output_path=/mnt/data/scene/eval  --vis viewer+tensorboard
    ```
- To load tensorboard results:
    ```
    tensorboard --logdir=/path/to/file
    ```
    - Connect to http://localhost:6006/

## 6. Calculate metrics

The single-command convenience script differs by method:

- **PanDORA**:
    ```
    python lantern_scripts/calculate_metrics.py --input_dir /mnt/data/coffee_room2/ --checkpoint /mnt/workspace/lantern/nerfstudio-HDR/outputs/coffee_room2_ns/PanDORA/2025-01-22_194346/ --metric_name unaligned
    ```
- **HDR-Nerfacto (`hdr-nerfacto`)**:
    ```
    python lantern_scripts/calculate_metrics_hdrnerfacto.py --input_dir /mnt/data/coffee_room2/ --checkpoint /mnt/workspace/lantern/nerfstudio-HDR/outputs/coffee_room2_ns/PanDORA/2025-01-22_194346/ --metric_name hdr_nerfacto
    ```

then point #8

**or**, run the steps manually (identical for both methods):

1. Get ground truth transformations:
    ```
    ns-process-data lantern-GT-HDR --data /mnt/data/coffee_room2/coffee_room2_ns/ --output-dir /mnt/data/coffee_room2/metrics --metadata /mnt/data/coffee_room2/sfm/reconstruction.json --checkpoint /mnt/workspace/lantern/nerfstudio-HDR/outputs/coffee_room2_ns/PanDORA/2025-01-22_194346/
    ```

2. Render panos at ground truth positions for well and fast exposed:
    ```
    ns-render camera-path --load-config /mnt/workspace/lantern/nerfstudio-HDR/outputs/coffee_room2_ns/PanDORA/2025-01-22_194346/config.yml --camera-path-filename /mnt/data/coffee_room2/metrics/GT_transforms.json --output-path /mnt/data/coffee_room2/metrics/renders/unaligned --output-format images
    ```
    ```
    ns-render camera-path --load-config /mnt/workspace/lantern/nerfstudio-HDR/outputs/coffee_room2_ns/PanDORA/2025-01-22_194346/config.yml --camera-path-filename /mnt/data/coffee_room2/metrics/GT_transforms.json --output-path /mnt/data/coffee_room2/metrics/renders/unaligned_fast --output-format images --rendered_output_names rgb_fast
    ```

3. Combine both well and fast exposed panos:
    ```
    python lantern_scripts/combine.py --well_dir /mnt/data/coffee_room2/metrics/renders/unaligned/ --fast_dir /mnt/data/coffee_room2/metrics/renders/unaligned_fast/ --out_dir /mnt/data/coffee_room2/metrics/renders/unaligned_lin/ --experiment_location /mnt/data/coffee_room2/data/ --do_linearization
    ```

4. Make sure the panos from renders have the same names as the GT panos.

5. Calculate PSNR, SSIM and LPIPS:
    ```
    python lantern_scripts/ldr_res.py --gt_dir /mnt/data/coffee_room2/GT/GT_exr/  --data_dir /mnt/data/coffee_room2/metrics/renders/unaligned_lin/
    ```

6. Generate GT and results renders with blender:
    ```
    blender --background lantern_scripts/scene_new.blend -P lantern_scripts/hdr_blender.py -- /mnt/data/coffee_room2/GT/GT_exr/ /mnt/data/coffee_room2/GT/GT_exr_renders/
    ```
    ```
    blender --background lantern_scripts/scene_new.blend -P lantern_scripts/hdr_blender.py -- /mnt/data/coffee_room2/metrics/renders/unaligned_lin/ /mnt/data/coffee_room2/metrics/renders/unaligned_lin_renders/
    ```

7. Calculate si-RMSE, RMSE, RGB ang. and PSNR (LDR r.). Need to change values in code:
    ```
    python lantern_scripts/res_table_1.py
    ```

8. Convert exr panos to hdr:
    ```
    convert GT1.exr GT1.hdr
    ```

9. Calulate PU-PSNR, HDR-VDP and PU-SSIM following the steps in the following link:
    https://github.com/darthgera123/PanoHDR-NeRF/tree/main/LANet/metrics
    ```
    cd hdrvdp-3.0.6
    ```
    ```
    matlab -batch "metric('../../data/GT_meeting_room/', '../../data/results_meeting_room/')"
    ```

    ```
    cd ../pu21/matlab/examples
    ```
    ```
    matlab -batch "pupsnr('../../../../data/GT_meeting_room/', '../../../../data/results_meeting_room/')"
    ```
