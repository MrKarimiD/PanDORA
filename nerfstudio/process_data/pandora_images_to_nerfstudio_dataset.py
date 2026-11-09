# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processes an image sequence to a nerfstudio compatible dataset."""

import json
import os
import re
import shutil
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pandora_scripts.extract_linearized import apply_correction
from nerfstudio.process_data import colmap_utils, equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import (
    ColmapConverterToNerfstudioDataset,
)
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class PandoraImagesToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process images into a nerfstudio dataset.

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    percent_radius_crop: float = 1.0	
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    # left_colmap_baseline_dir: Optional[Path] = None
    # """Path to the left camera baseline directory."""
    # left_colmap_baseline_mask_dir: Optional[Path] = None
    # """Path to the left camera baseline mask directory."""
    # right_colmap_baseline_dir: Optional[Path] = None
    # """Path to the right camera baseline directory."""
    # right_colmap_baseline_mask_dir: Optional[Path] = None
    # """Path to the right camera baseline mask directory."""
# 
    # left_e1_dir: Optional[Path] = None
    # """Path to the left camera e1 directory (well-exposed)."""
    # left_e1_mask_dir: Optional[Path] = None
    # """Path to the left camera e1 mask directory (well-exposed)."""
    # right_e2_dir: Optional[Path] = None
    # """Path to the right camera e2 directory (fast-exposed)."""
    # right_e2_mask_dir: Optional[Path] = None
    # """Path to the right camera e2 mask directory (fast-exposed)."""

    exposure1: float = None # t = 1/125
    # """Well-exposed exposure value (left camera)."""
# 
    exposure2: float = None # t = 1/25000
    # """Fast-exposed exposure value (right camera)."""

   # exposure1: float = 1.0 # t = 1/125
   # """Well-exposed exposure value (left camera)."""
# 
   # exposure2: float = 0.0032 # t = 1/25000
   # """Fast-exposed exposure value (right camera)."""

    white_balance_setting = 3500

    skip_perspective_transform: bool = False
    """Skip perspective transformation."""

    skip_colmap_to_json: bool = False
    """Skip colmap to json conversion."""

    skip_compactness_constraint: bool = False

    skip_autocenter_constraint: bool = False

    capture_settings_file_name: str = "capture_settings.json"

    skip_linearization: bool = False

    skip_saturation_mask: bool = False

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""
        assert self.skip_image_processing, "Image processing not supported for now"	
        self.is_HDR = True

        # check camera settings
        with open(self.data / self.capture_settings_file_name) as f:
            # read shutter_left, shutter_right, iso_left, iso_right
            data = json.load(f)
            self.white_balance_setting = data["left"]["white_balance"]
            self.exposure1 = 1.0
            self.exposure2 = data["right"]["shutter_speed"] / data["left"]["shutter_speed"]

            
        
        require_cameras_exist = False
        if self.colmap_model_path != ColmapConverterToNerfstudioDataset.default_colmap_path():
            if not self.skip_colmap:
                raise RuntimeError("The --colmap-model-path can only be used when --skip-colmap is not set.")
            if not (self.output_dir / self.colmap_model_path).exists():
                raise RuntimeError(f"The colmap-model-path {self.output_dir / self.colmap_model_path} does not exist.")
            require_cameras_exist = True

        image_rename_map: Optional[dict[str, str]] = None

        sequence_names = ['left_colmap_baseline', 'right_colmap_baseline', 'left_e1', 'right_e2']
        perspective_dirs = []
        perspective_mask_dirs = []

        if self.camera_type == "equirectangular":
            if self.eval_data is not None:	
                raise ValueError("Cannot use eval_data with camera_type equirectangular.")
            if not self.skip_perspective_transform:
                pers_size = equirect_utils.compute_resolution_from_equirect(self.data / sequence_names[0], self.images_per_equirect)
                CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")
            else:
                pers_size = None

            for sequence_name in sequence_names:
                if self.mask_dir:
                    mask_path = self.mask_dir / sequence_name
                    output_mask_dir = self.data / sequence_name  / 'mask_planar_projections'
                else:
                    mask_path = None
                    output_mask_dir = None
                output_dir = self.data / sequence_name / 'planar_projections'
                perspective_dirs.append(output_dir)
                perspective_mask_dirs.append(output_mask_dir)

                if not self.skip_perspective_transform:
                    equirect_utils.generate_planar_projections_from_equirectangular(
                        self.data / sequence_name, pers_size, self.images_per_equirect, mask_path, crop_factor=self.crop_factor
                    )
                    CONSOLE.log(f"Generated {len(os.listdir(output_dir))} images in {output_dir}")

                
            self.camera_type = "perspective"
        else:
            raise ValueError("Camera type must be equirectangular")

        summary_log = []

        # Copy and downscale images
        if not self.skip_image_processing:
            # Copy images to output directory
            #TODO 
            image_rename_map_paths = {}
            for i in range(len(perspective_dirs)):
                image_rename_map_paths.update(process_data_utils.copy_images(
                    perspective_dirs[i],
                    image_dir=self.image_dir,
                    crop_factor=self.crop_factor,
                    verbose=self.verbose,
                    num_downscales=self.num_downscales,
                ))
                
                if self.eval_data is not None:
                    raise NotImplementedError
                    eval_image_rename_map_paths = process_data_utils.copy_images(	
                        self.eval_data,	
                        image_dir=self.image_dir,	
                        crop_factor=self.crop_factor,	
                        image_prefix="frame_eval_",	
                        verbose=self.verbose,	
                        num_downscales=self.num_downscales,	
                    )	
                    image_rename_map_paths.update(eval_image_rename_map_paths)
            
            image_rename_map = dict((a.name, b.name) for a, b in image_rename_map_paths.items())
            num_frames = len(image_rename_map)
            summary_log.append(f"Starting with {num_frames} images")

        else:
           #num_frames = len(process_data_utils.list_images(self.data / sequence_names[0]))
           #if num_frames == 0:
           #    raise RuntimeError("No usable images in the data folder.")
           #summary_log.append(f"Starting with {num_frames} images")
            pass

        image_rename_map = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir) / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir = Path(tmp_dir) / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

            # copy files for colmap in tmp_dir
            for i in range(len(perspective_dirs)):
                # avoid right_e2 (fast-exposed), colmap doesn't work well on it
                if i == 3:
                    continue

                shutil.copytree(perspective_dirs[i], image_dir / sequence_names[i])
            for i in range(len(perspective_mask_dirs)):
                # avoid right_e2 (fast-exposed), colmap doesn't work well on it
                if i == 3:
                    continue

                shutil.copytree(perspective_mask_dirs[i], mask_dir / sequence_names[i])

            # Run COLMAP
            if not self.skip_colmap:
                require_cameras_exist = True
                self.absolute_colmap_path.mkdir(parents=True, exist_ok=True)
                colmap_utils.run_colmap(
                    image_dir=image_dir,
                    colmap_dir=self.absolute_colmap_path,
                    camera_model=process_data_utils.CAMERA_MODELS[self.camera_type],
                    mask_path=mask_dir,
                    gpu=self.gpu,
                    verbose=self.verbose,
                    matching_method=self.matching_method,
                    colmap_cmd=self.colmap_cmd,
                )
                # Colmap uses renamed images
                image_rename_map = None

        # # Export depth maps
        image_id_to_depth_path, log_tmp = self._export_depth()
        summary_log += log_tmp
        incomplete_transforms_file = self.output_dir / "transforms_incomplete.json"

        if not self.skip_colmap_to_json:
            if require_cameras_exist and not (self.absolute_colmap_model_path / "cameras.bin").exists():
                raise RuntimeError(f"Could not find existing COLMAP results ({self.colmap_model_path / 'cameras.bin'}).")
            
            summary_log += self._save_transforms(
                camera_mask_path=Path('masks'),
            )

            shutil.move(str(self.output_dir / 'transforms.json'), incomplete_transforms_file)
            CONSOLE.log(f"Saved initial transforms to {incomplete_transforms_file}")
        original_transforms = load_from_json(incomplete_transforms_file)

        frames = original_transforms["frames"]
        sorted_frames = sorted(frames, key=lambda x: x["file_path"])

        # frames are splitted into 4 groups: left_calib, right_calib, left_e1, right_e2
        # the two first groups are used to infer the basis_change between left and right
        # that basis_change is then applied to the third group (left_e1) to infer the fourth (right_e2)

        grouped_frames = {sequence_name: dict() for sequence_name in ['left_colmap_baseline', 'right_colmap_baseline', 'left_e1']}
        for frame in sorted_frames:
            # use regex
            matches = re.match(r'images/(\w+)/.*_(\d+)_(\d+).png', frame["file_path"])
            if matches:
                sequence_name = matches.group(1)
                frame_idx = int(matches.group(2))
                crop_direction = int(matches.group(3))
                if crop_direction not in grouped_frames[sequence_name].keys():
                    grouped_frames[sequence_name][crop_direction] = dict()
                updated_frame = deepcopy(frame)
                # change frame's file_path, which looks like images/left_colmap_baseline/000000_0_0.png to images/left_colmap_baseline/planar_projections/000000_0.png
                # also change frame's mask_path, which looks like mask/left_colmap_baseline/000000_0_0.png to images/left_colmap_baseline/mask_planar_projections/000000_0.png
                updated_frame["file_path"] = updated_frame["file_path"].replace(f'images/{sequence_name}/', f'images/{sequence_name}/planar_projections/')
                updated_frame["mask_path"] = updated_frame["mask_path"].replace(f'masks/{sequence_name}/', f'images/{sequence_name}/mask_planar_projections/')
                grouped_frames[sequence_name][crop_direction][frame_idx] = updated_frame
            else:
                # warning
                print(f'Warning: {frame["file_path"]} does not match regex')


        # write to text file grouped_frames
        with open(self.output_dir / 'grouped_frames.json', 'w') as f:
            json.dump(grouped_frames, f, indent=4)


        thresh = 0.02
        COMPACTNESS_THRESHOLDS = {
            'left_colmap_baseline': thresh,
            'right_colmap_baseline': thresh,
            'left_e1': thresh,
        }
        deleted_panoramic_frame_indices = dict()
        for sequence_name in ['left_colmap_baseline', 'right_colmap_baseline', 'left_e1']:
            deleted_panoramic_frame_indices[sequence_name] = []
        if not self.skip_compactness_constraint:
            for sequence_name in ['left_colmap_baseline', 'right_colmap_baseline', 'left_e1']:
                panoramic_frame_count = len(grouped_frames[sequence_name][1])
                print(f'panoramic_frame_count: {panoramic_frame_count}')
                panoramic_frames = dict()
                for crop_direction in grouped_frames[sequence_name].keys():
                    for i, frame in grouped_frames[sequence_name][crop_direction].items():
                        if i not in panoramic_frames.keys():
                            panoramic_frames[i] = []
                        panoramic_frames[i].append(frame)
                
                for i, panoramic_frame in panoramic_frames.items():
                    camera_center_mean = np.mean([np.array(frame['transform_matrix'])[0:3, 3] for frame in panoramic_frame], axis=0)
                    camera_distances = [np.linalg.norm(np.array(frame['transform_matrix'])[0:3, 3] - camera_center_mean) for frame in panoramic_frame]
                    camera_center_std = np.std(camera_distances)
                    norm_camera_center_std = np.linalg.norm(camera_center_std)
                    print(f'norm_camera_center_std: {norm_camera_center_std}')
                    if norm_camera_center_std > COMPACTNESS_THRESHOLDS[sequence_name]:
                        print(f'Warning: norm_camera_center_std {norm_camera_center_std} > COMPACTNESS_THRESHOLDS[{sequence_name}] {COMPACTNESS_THRESHOLDS[sequence_name]}')
                        deleted_panoramic_frame_indices[sequence_name].append(i)

                if sequence_name == 'left_e1':
                    # delete from grouped_frames
                    for frame_idx in deleted_panoramic_frame_indices[sequence_name]:
                        for crop_direction in grouped_frames[sequence_name].keys():
                            try:
                                del grouped_frames[sequence_name][crop_direction][frame_idx]
                            except KeyError:
                                print(f'KeyError, frame_idx: {frame_idx}, crop_direction: {crop_direction}')

        if not self.skip_autocenter_constraint:
            for sequence_name in ['left_colmap_baseline', 'right_colmap_baseline', 'left_e1']:
                panoramic_frames = dict()
                for crop_direction in grouped_frames[sequence_name].keys():
                    for i, frame in grouped_frames[sequence_name][crop_direction].items():
                        if i not in panoramic_frames.keys():
                            panoramic_frames[i] = dict()
                        panoramic_frames[i][crop_direction] = frame

                for i, panoramic_frame in panoramic_frames.items():
                    camera_center_mean = np.mean([np.array(frame['transform_matrix'])[0:3, 3] for frame in panoramic_frame.values()], axis=0)
                    # set all the camera centers to the mean
                    for frame in panoramic_frame.values():
                        
                        # frame['transform_matrix'][0:3, 3] = camera_center_mean
                        # to numpy
                        transform_matrix = np.array(frame['transform_matrix'])
                        transform_matrix[0:3, 3] = camera_center_mean
                        # to list
                        frame['transform_matrix'] = transform_matrix.tolist()


                # update grouped_frames
                for crop_direction in grouped_frames[sequence_name].keys():
                    for i, frame in grouped_frames[sequence_name][crop_direction].items():
                        frame['transform_matrix'] = panoramic_frames[i][crop_direction]['transform_matrix']
                        

        # group 3 is empty, we want to infer its transform
        grouped_frames['right_e2'] = dict()
        for crop_direction in range(self.images_per_equirect):
            if crop_direction not in grouped_frames['left_colmap_baseline'].keys() or crop_direction not in grouped_frames['right_colmap_baseline'].keys():
                print(f'Warning: no left_colmap_baseline or right_colmap_baseline for crop_direction {crop_direction}')
                continue
            
            total_basis_change = np.zeros((4, 4))
            basis_change_count = 0
            for frame_idx in grouped_frames['left_colmap_baseline'][crop_direction].keys():
                print(f'crop_direction: {crop_direction}, frame_idx: {frame_idx}')
                if frame_idx in deleted_panoramic_frame_indices['left_colmap_baseline']:
                    continue
                if frame_idx in deleted_panoramic_frame_indices['right_colmap_baseline']:
                    continue
                if frame_idx not in grouped_frames['right_colmap_baseline'][crop_direction].keys():
                    print(f'Warning: no right_colmap_baseline for crop_direction {crop_direction}, frame_idx {frame_idx}')
                    continue
                
                group_0_tranform = np.array(grouped_frames['left_colmap_baseline'][crop_direction][frame_idx]['transform_matrix'])
                group_1_tranform = np.array(grouped_frames['right_colmap_baseline'][crop_direction][frame_idx]['transform_matrix'])
                
                basis_change = np.linalg.inv(group_0_tranform) @ group_1_tranform
                total_basis_change += basis_change
                basis_change_count += 1
            
            basis_change = total_basis_change / basis_change_count

            # group_0_tranform = np.array(grouped_frames['left_colmap_baseline'][crop_direction][0]['transform_matrix'])
            # group_1_tranform = np.array(grouped_frames['right_colmap_baseline'][crop_direction][0]['transform_matrix'])
            # basis_change = np.linalg.inv(group_0_tranform) @ group_1_tranform

            # sometimes, colmap omits frames, so if a frame doesn't have the left_e1 pose, we can't infer the right_e2 pose
            if grouped_frames['left_e1'].get(crop_direction):
                for frame_idx, frame in grouped_frames['left_e1'][crop_direction].items():
                    
                    # apply group_3_tranform to frame, put it in group 3
                    group_3_frame_transform = (np.array(frame['transform_matrix']) @ basis_change).tolist()
                    if crop_direction not in grouped_frames['right_e2'].keys():
                        grouped_frames['right_e2'][crop_direction] = dict()
                    grouped_frames['right_e2'][crop_direction][frame_idx] = {
                        'file_path': frame['file_path'].replace('left_e1', 'right_e2').replace('lhs', 'rhs'), # TODO validate file naming convention
                        'transform_matrix': group_3_frame_transform,
                        'mask_path': frame['mask_path'].replace('left_e1', 'right_e2').replace('lhs', 'rhs'), # TODO validate file naming convention
                    }
            else:
                # warning
                print(f'Warning: no left_e1 for crop_direction {crop_direction}')
        
        # TODO: copy images

        # generate new transforms.json
        new_transforms = deepcopy(original_transforms)
        new_transforms['frames'] = []
        for sequence_name in ['left_e1', 'right_e2']:
            for i, crop_direction in enumerate(range(self.images_per_equirect)):
                if crop_direction not in grouped_frames[sequence_name].keys():
                    print(f'Warning: no {sequence_name} for crop_direction {crop_direction}')
                    continue
                new_transforms['frames'] += grouped_frames[sequence_name][crop_direction].values()
                

        # linearization
        if not self.skip_linearization:
            with open(f'linearize-ricoh-z10-wb-{self.white_balance_setting}.json') as f:
                data = json.load(f)
                CORRECTION_CURVE_TYPE = data["type"]

            for sequence_name in ['left_e1', 'right_e2']:
                for i, crop_direction in enumerate(range(self.images_per_equirect)):
                    if crop_direction not in grouped_frames[sequence_name].keys():
                        print(f'Warning: no {sequence_name} for crop_direction {crop_direction}')
                        continue
                    for frame in grouped_frames[sequence_name][crop_direction].values():
                        print(os.path.join(self.output_dir, frame['file_path']))
                        img = cv2.imread(os.path.join(self.output_dir, frame['file_path']))
                        cv2.imwrite(os.path.join(self.output_dir, frame['file_path'].replace('.png', '_nonlinear.exr')), img.astype(np.float32) / 255)
                        img_corrected = apply_correction((data["B"], data["G"], data["R"]), img, CORRECTION_CURVE_TYPE)
                        print(os.path.join(self.output_dir, frame['file_path'].replace('.png', '_linear.exr')))
                        
                        img_corrected /= 255
                        exposed_img = img_corrected * self.exposure1 if sequence_name == 'left_e1' else img_corrected / self.exposure2
                        cv2.imwrite(os.path.join(self.output_dir, frame['file_path'].replace('.png', '_linear.exr')), exposed_img)

                        #frame['file_path'] = frame['file_path'].replace('.png', '_linear.exr')
 
        # saturation mask
        if not self.skip_saturation_mask:
            for sequence_name in ['left_e1', 'right_e2']:
                for i, crop_direction in enumerate(range(self.images_per_equirect)):
                    if crop_direction not in grouped_frames[sequence_name].keys():
                        print(f'Warning: no {sequence_name} for crop_direction {crop_direction}')
                        continue
                    for frame in grouped_frames[sequence_name][crop_direction].values():
                        img = cv2.imread(os.path.join(self.output_dir, frame['file_path']))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0
                        # left e1 threshold: valid from 0 to 0.95
                        # right e2 threshold: valid from 0.2 to 1
                        # to have mask be =1, all 3 channels must be valid
                        if sequence_name == 'left_e1':
                            min_thresh = 0.0
                            max_thresh = 0.95
                        else:
                            min_thresh = 0.2
                            max_thresh = 1.0
                        mask = np.greater_equal(img, min_thresh) & np.less_equal(img, max_thresh)
                        mask = np.all(mask, axis=2)
                        mask = mask.astype(np.uint8) * 255
                        cv2.imwrite(os.path.join(self.output_dir, frame['mask_path'].replace('.png', '_saturation_mask.png')), mask)
                        # add to frame
                        frame['saturation_mask_path'] = frame['mask_path'].replace('.png', '_saturation_mask.png')

        # change frame file_path to linear exr
        for frame in new_transforms['frames']:
            frame['file_path'] = frame['file_path'].replace('.png', '_linear.exr')

        # generate new transforms.json
        new_transforms['frames'] = []
        for sequence_name in ['left_e1']:#, 'right_e2']:
        # for sequence_name in ['left_e1', 'right_e2']:
        # for sequence_name in ['left_colmap_baseline', 'right_colmap_baseline']:
            for i, crop_direction in enumerate(range(self.images_per_equirect)):
                if crop_direction not in grouped_frames[sequence_name].keys():
                    print(f'Warning: no {sequence_name} for crop_direction {crop_direction}')
                    continue
                new_transforms['frames'] += grouped_frames[sequence_name][crop_direction].values()

        

        # for HDR-Nerfacto, output also the exposures
        exposures_content = {}
        for frame in new_transforms['frames']:
            if 'right_e2' in frame['file_path']:
                exposures_content[frame['file_path']] = self.exposure2
            else:
                exposures_content[frame['file_path']] = self.exposure1
        
        # add the exposures to the transforms.json
        for frame in new_transforms['frames']:
            frame['exposure'] = exposures_content[frame['file_path']]

        with open(self.output_dir / 'exposures.json', "w", encoding="UTF-8") as file:
            json.dump(exposures_content, file, indent=4)        

        with open(self.output_dir / 'transforms.json', "w", encoding="UTF-8") as file:
            json.dump(new_transforms, file, indent=4)

        CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.log(summary)
