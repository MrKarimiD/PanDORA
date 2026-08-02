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

"""Processes an panorama sequence to a nerfstudio compatible dataset."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Tuple
import os
import shutil
import numpy as np
from numpy.linalg import inv
import cv2
from tqdm import tqdm


from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.base_converter_to_nerfstudio_dataset import (
    BaseConverterToNerfstudioDataset,
)
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils import io


def _populate_camera_mask_folder(data_dir, sequence_name):
    """Route per-image masks from the flat <data_dir>/masks/ folder into
    <data_dir>/<sequence_name>/mask/<image_stem>.png, which is where the GT
    planar-projection step expects each mask. Handles source masks stored as
    masks/<camera>/<stem>.png, masks/<camera>_<stem>.png, or masks/<stem>.png."""
    cam_dir = data_dir / sequence_name
    src_dir = data_dir / "masks"
    if not src_dir.is_dir():
        return
    dst_dir = cam_dir / "mask"
    dst_dir.mkdir(exist_ok=True)
    for img in sorted(cam_dir.glob("*.png")):
        stem = img.stem
        for candidate in (
            src_dir / sequence_name / f"{stem}.png",
            src_dir / f"{sequence_name}_{stem}.png",
            src_dir / f"{stem}.png",
        ):
            if candidate.exists():
                shutil.copy(candidate, dst_dir / f"{stem}.png")
                break


def apply_correction(coefs, img, CORRECTION_CURVE_TYPE) -> np.ndarray:
    # check that img is in range 0-255
    assert img.dtype == np.uint8
    assert CORRECTION_CURVE_TYPE in ['gamma']
    img = ((img.astype(np.float32) / 255.0) ** np.array(coefs).T) # * 255.0
    # img = ((img.astype(np.float32) / 255.0) ** coefs[None, None, :]) * 255.0
    return img


def distance(point_array, ref):
    pow_2 = np.power(point_array - ref, 2)
    sum_pow_2 = np.sum(pow_2, axis=-1)
    dist = np.sqrt(sum_pow_2)
    return dist


def opensfm_to_opengl(shot_data):
    rvec = np.array(tuple(map(float, shot_data['rotation'])))
    tvec = np.array(tuple(map(float, shot_data['translation'])))

    # Converting opencv to opengl coordinates
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(rvec).as_matrix()
    
    W2C = np.eye(4)
    W2C[:3, :3] = rot
    W2C[:3, 3] = np.array(tvec)
    
    C2W = inv(W2C)
    C2W[0:3, 1:3] *= -1
    C2W = C2W[np.array([1, 0, 2, 3]), :]
    C2W[2, :] *= -1

    return C2W, rvec, tvec


@dataclass
class OpenSFMToNeRFStudioDataset(BaseConverterToNerfstudioDataset):
    """Process panorama data into a nerfstudio dataset. Skip the COLMAP

    This script does the following:

    1. Crops panoramas to planar images, and restore corresponding perspective camera poses.
    1. Scales images to a specified size.
    2. Converts perpective camera poses into the nerfstudio format.
    """
    
    metadata: Path = ""
    """Path the metadata of the panoramas sequence."""
    num_downscales: int = 0
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    max_dataset_size: int = 300
    """Max number of images to train on. If the dataset has more, images will be sampled approximately evenly. If -1,
    use all images."""
    images_per_equirect: Literal[8, 14] = 8
    """Number of samples per image to take from each equirectangular image.
       Used only when camera-type is equirectangular.
    """
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    """Portion of the image to crop. All values should be in [0,1]. (top, bottom, left, right)"""
    skip_image_processing: bool = False
    """If True, skips copying and downscaling of images and only runs COLMAP if possible and enabled"""
    is_HDR: bool = False
    """If True, process the .exr files as HDR images."""
    use_mask: bool = True
    """If True, process the .exr files with mask."""
    use_exposure: bool = False
    """If True, using different exposures."""

    exposure1: float = None # t = 1/125
    """Well-exposed exposure value (left camera)."""
    exposure2: float = None # t = 1/25000
    """Fast-exposed exposure value (right camera)."""
    capture_settings_file_name: str = "capture_settings.json"

    skip_linearization: bool = False

    skip_saturation_mask: bool = False

    skip_histograms: bool = False

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        mask_dir = self.output_dir / "masks"
        if self.use_mask:
            mask_dir.mkdir(parents=True, exist_ok=True)

        # check camera settings
        with open(self.data / self.capture_settings_file_name) as f:
            # read shutter_left, shutter_right, iso_left, iso_right
            data = json.load(f)
            self.white_balance_setting = data["left"]["white_balance"]
            self.exposure1 = 1.0
            self.exposure2 = data["right"]["shutter_speed"] / data["left"]["shutter_speed"]

        summary_log = []

        if not self.skip_image_processing: 
            openSFM_reconstruction = io.load_from_json(self.metadata)
            frames_names = openSFM_reconstruction[0]['shots'].keys()
            frames_names = sorted(frames_names)

            # Remove GT files from the training sets
            frames_names = list(set(frames_names) - set( [x for x in frames_names if x.startswith('GT')]))
            
            # Computing the basis transform matrix
            rot_list = []
            trn_list = []

            for fname in frames_names:
                if fname.startswith("left_sfm"):
                    shot_data = openSFM_reconstruction[0]['shots'][fname]
                    left_C2W, left_rvec, left_tvec = opensfm_to_opengl(shot_data)
                    
                    right_fname = fname.replace("left_sfm", "right_sfm")
                    if right_fname in frames_names:
                        shot_data = openSFM_reconstruction[0]['shots'][right_fname]
                        right_C2W, right_rvec, right_tvec = opensfm_to_opengl(shot_data)
                        basis_change = np.linalg.inv(left_C2W) @ right_C2W

                        from scipy.spatial.transform import Rotation as R
                        r = R.from_matrix(basis_change[0:3, 0:3])
                        rot = r.as_euler('zyx', degrees=True)
                        rot_list.append(r.as_quat())
                        trn_list.append(basis_change[0:3, 3])
                    else:
                        print("The right one is not registered for " + fname)
            
            # Filtering the data points
            trn_np = np.asarray(trn_list)
            trn_median = np.median(trn_np, 0)
            dist = distance(trn_np, trn_median)
            percentile_95 = np.percentile(dist, 95)
            valid_idx = dist < percentile_95
            
            trn_np = trn_np[valid_idx]
            rot_np = np.asarray(rot_list)[valid_idx]

            basis_change = np.zeros((4, 4))
            # basis_change = solution_w_SVD
            from sksurgerycore.algorithms.averagequaternions import average_quaternions
            avg_quat = average_quaternions(rot_np)
            r = R.from_quat(avg_quat)
            basis_change[0:3, 0:3] = r.as_matrix()
            basis_change[0:3, 3] = np.mean(trn_np, 0)
            basis_change[3, 3] = 1.0
            print("basis_change: ", basis_change)
            
            # Convert OpenSFM coordinates to the NeRFStudio
            translations = []
            for fname in frames_names:
                # if fname.startswith("left_sfm"):
                if fname.startswith("left_e1"):
                    shot_data = openSFM_reconstruction[0]['shots'][fname]
                    _, _, left_tvec = opensfm_to_opengl(shot_data)
                    translations.append(left_tvec)
            
            translations_np = np.asarray(translations)
            left_trn_median = np.median(translations_np, 0)
            dist = distance(translations_np, left_trn_median)
            percentile_95 = np.percentile(dist, 99)
            valid_idx = dist < percentile_95

            # Convert OpenSFM coordinates to the NeRFStudio
            camera_dict = {}
            camera_dict["frames"] = []
            for idx, fname in enumerate([x for x in frames_names if x.startswith('left_e1')]):
                # if fname.startswith("left_sfm"):
                right_fname = fname.replace("left_e1", "right_e2")
                # right_fname = fname.replace("left_sfm", "right_sfm")
                shot_data = openSFM_reconstruction[0]['shots'][fname]
                left_C2W, left_rvec, left_tvec = opensfm_to_opengl(shot_data)
                if valid_idx[idx]:
                    camera_dict["frames"].append(
                    {
                        "file_path": fname.split('.')[0],
                        "transform_matrix": left_C2W.tolist()
                    })
                    
                    right_C2W = left_C2W @ basis_change
                    camera_dict["frames"].append(
                    {
                        "file_path": right_fname.split('.')[0],
                        "transform_matrix": right_C2W.tolist()
                    })
                        
            with open(self.output_dir / 'panoramic_transforms.json', 'w') as f:
                json.dump(camera_dict, f, indent=4)
            
            pers_size = equirect_utils.compute_resolution_from_equirect(self.data / 'left_e1', self.images_per_equirect)
            # pers_size = equirect_utils.compute_resolution_from_equirect(self.data / 'left_sfm', self.images_per_equirect)
            
            cropped_images_filename = []
            cropped_masks_filename = []
            metadata_dict_all = {}
            # for sequence_name in ['left_sfm', 'right_sfm']:
            for sequence_name in ['left_e1', 'right_e2']:
                if self.use_mask:
                    _populate_camera_mask_folder(self.data, sequence_name)
                CONSOLE.log(f"Generating {self.images_per_equirect} {pers_size} sized images per equirectangular image")
                out_dir = equirect_utils.generate_planar_projections_from_equirectangular_GT(
                    self.output_dir / 'panoramic_transforms.json',
                    self.data / sequence_name, pers_size, self.images_per_equirect, 
                    crop_factor=self.crop_factor, 
                    clip_output = False,
                    use_mask = self.use_mask,
                    prefix = sequence_name
                )
                self.camera_type = "perspective"
                metadata_dict = io.load_from_json(out_dir / "transforms.json")
                metadata_dict_all[sequence_name] = metadata_dict
                # Copy images to output directory
                for frame in metadata_dict["frames"]:
                    cropped_images_filename.append(Path(frame["file_path"]))
                    if self.use_mask:
                        cropped_masks_filename.append(Path(frame["mask_path"]))
            
            # Copy images to output directory
            copied_image_paths = process_data_utils.copy_images_list(
                cropped_images_filename, image_dir=image_dir, verbose=self.verbose, num_downscales=self.num_downscales
            )
            copied_image_paths = [Path("images/" + copied_image_path.name) for copied_image_path in copied_image_paths]
            if self.use_mask:
                copied_mask_paths = process_data_utils.copy_images_list(
                    cropped_masks_filename, image_dir=mask_dir, verbose=self.verbose, num_downscales=self.num_downscales
                )
                copied_mask_paths = [Path("masks/" + copied_mask_path.name) for copied_mask_path in copied_mask_paths]
            num_frames = len(copied_image_paths)

            summary_log.append(f"Used {num_frames} images out of {num_frames} total")
            if self.max_dataset_size > 0:
                summary_log.append(
                    "To change the size of the dataset add the argument [yellow]--max_dataset_size[/yellow] to "
                    f"larger than the current value ({self.max_dataset_size}), or -1 to use all images."
                )

            metadata_path = self.output_dir / "transforms.json"
            metadata_dict = {}
            for key in metadata_dict_all["left_e1"].keys() - 'frames':
                metadata_dict[key] = metadata_dict_all["left_e1"][key]
            metadata_dict["frames"] = metadata_dict_all['left_e1']['frames'] + metadata_dict_all['right_e2']['frames']
            # for key in metadata_dict_all["left_sfm"].keys() - 'frames':
            #     metadata_dict[key] = metadata_dict_all["left_sfm"][key]
            # metadata_dict["frames"] = metadata_dict_all['left_sfm']['frames'] + metadata_dict_all['right_sfm']['frames']

            for i, frame in enumerate(metadata_dict["frames"]):
                frame["file_path"] = str(copied_image_paths[i])
                if self.use_mask:
                    frame["mask_path"] = str(copied_mask_paths[i])

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=4)

        transform_data = io.load_from_json(self.output_dir / "transforms.json")
        
        # linearization
        if not self.skip_linearization:
            
            with open(f'linearize-ricoh-z10-wb-{self.white_balance_setting}.json') as f:
                data = json.load(f)
                CORRECTION_CURVE_TYPE = 'gamma'

            for frame in tqdm(transform_data["frames"], desc="Linearization: "):
                img = cv2.imread(os.path.join(self.output_dir, frame['file_path']))
                # import pdb; pdb.set_trace()
                img_corrected = apply_correction((data["b"], data["g"], data["r"]), img, CORRECTION_CURVE_TYPE) # 
                # img_corrected /= 255
                # img_corrected = img_corrected ** 2.2
                # import pdb; pdb.set_trace()
                img_name = frame['file_path'].split('/')[-1]
                exposed_img = np.float32(img_corrected) # * self.exposure1 if img_name.startswith('left_e1') else img_corrected / self.exposure2
                # exposed_img = img_corrected * self.exposure1
                cv2.imwrite(os.path.join(self.output_dir, frame['file_path'].replace('.png', '_linear.exr')), exposed_img)
                # import pdb; pdb.set_trace()

        # saturation mask
        if not self.skip_saturation_mask:
            for frame in tqdm(transform_data["frames"], desc="Computing the Saturation mask: "):
                img = cv2.imread(os.path.join(self.output_dir, frame['file_path']))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                # left e1 threshold: valid from 0 to 0.95
                # right e2 threshold: valid from 0.2 to 1
                # to have mask be =1, all 3 channels must be valid
                img_name = frame['file_path'].split('/')[-1]
                if img_name.startswith('left_e1'):
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
        # for frame in transform_data['frames']:
        #     frame['file_path'] = frame['file_path'].replace('.png', '_linear.exr')

        # for HDR-Nerfacto, output also the exposures
        exposures_content = {}
        for frame in transform_data['frames']:
            if 'right_e2' in frame['file_path']:
                exposures_content[frame['file_path']] = self.exposure2
            else:
                exposures_content[frame['file_path']] = self.exposure1
        
        # add the exposures to the transforms.json
        for frame in transform_data['frames']:
            frame['exposure'] = exposures_content[frame['file_path']]

        with open(self.output_dir / 'exposures.json', "w", encoding="UTF-8") as file:
            json.dump(exposures_content, file, indent=4)        

        with open(self.output_dir / 'transforms.json', "w", encoding="UTF-8") as file:
            json.dump(transform_data, file, indent=4)

        
        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()