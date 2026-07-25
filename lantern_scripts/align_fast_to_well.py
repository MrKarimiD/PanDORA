import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import json
import subprocess
import argparse
import shutil

class ContourTooSmallException(Exception):
    pass

class Image:
    def __init__(self, image, threshold = 5):
        self.image = image.copy()
        self.threshold = threshold
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.thresholded = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        self.contours, _ = cv2.findContours(self.thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

class Contour:
    def __init__(self, contour, well_image_threshold_index = None, min_area = 50.0):
        self.contour = contour
        self.area = cv2.contourArea(contour)
        if self.area < min_area:
            raise ContourTooSmallException(f"Contour area {self.area} is too small.")
        self.centroid = np.mean(contour, axis=0)[0]
        self.relative_position = None
        self.match = None
        self.score = float('inf')
        self.well_image_threshold_index = well_image_threshold_index

    def overlap(self, contour):
        image_shape = (960, 960)
        mask1 = np.zeros(image_shape, dtype=np.uint8)
        mask2 = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(mask1, [self.contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask2, [contour.contour], -1, 255, thickness=cv2.FILLED)
        overlap_mask = cv2.bitwise_and(mask1, mask2)
        overlap_area = np.sum(overlap_mask) / 255
        return overlap_area > 0
    
    def get_matching_score(self, contour):
        dist_shape = cv2.matchShapes(self.contour, contour.contour, cv2.CONTOURS_MATCH_I2, 0.0)
        dist_spatial = np.linalg.norm(np.array(self.centroid) - np.array(contour.centroid))
        return 0.7 * dist_shape + 0.3 * dist_spatial
    
    def match_contour(self, contour, score):
        self.match = contour
        self.score = score
    
    def unmatch_contour(self):
        self.match = None
        self.score = float('inf')

    def get_points(self):
        return self.contour[:, 0, :]


class AlignImages:
    def __init__(self, well_image, fast_image):
        self.fast_image = Image(fast_image)
        self.fast_contours = []
        for contour in self.fast_image.contours:
            try:
                contour_obj = Contour(contour)
                self.fast_contours.append(contour_obj)
            except ContourTooSmallException as e:
                continue
        self.matched_well_contours = [None] * len(self.fast_contours)
        self.well_points_combined = []
        self.fast_points_combined = []
        self.well_image = Image(well_image)
        self.well_images = []
        self.well_thresholds = (230, 235, 240, 245, 250)
        for threshold in self.well_thresholds:
            self.well_images.append(Image(well_image, threshold))
        self.image_height, self.image_width = self.fast_image.image.shape[:2]

    def all_contours_matched(self):
        for contour in self.fast_contours:
            if contour.match == None:
                return False
        return True
    
    def existing_contour_match(self, contour):
        for i, existing_contour in enumerate(self.matched_well_contours):
            if existing_contour != None and contour.overlap(existing_contour):
                return i, self.fast_contours[i]
        return None, None
    
    def delete_unmatched_contours(self):
        for contour_index in range(len(self.fast_contours) -1, -1, -1):
            if self.fast_contours[contour_index].match == None or self.matched_well_contours[contour_index] == None:
                del self.fast_contours[contour_index]
                del self.matched_well_contours[contour_index]
    
    def match_contours(self):
        iterations = 0
        while not self.all_contours_matched():
            if iterations >= 5:
                self.delete_unmatched_contours()
                break
            for contour_index, fast_contour in enumerate(self.fast_contours):
                for threshold_index, _ in enumerate(self.well_thresholds):
                    for well_contour in self.well_images[threshold_index].contours:
                        try:
                            well_contour = Contour(well_contour, threshold_index)
                            score = fast_contour.get_matching_score(well_contour)
                            if score < fast_contour.score:
                                if score >= 200:
                                    continue
                                match_index, match = self.existing_contour_match(well_contour)
                                if match != None:
                                    if score < match.score:
                                        match.unmatch_contour()
                                        self.matched_well_contours[match_index] = None
                                        fast_contour.match_contour(well_contour, score)
                                        self.matched_well_contours[contour_index] = well_contour
                                    else:
                                        continue
                                else:
                                    fast_contour.match_contour(well_contour, score)
                                    self.matched_well_contours[contour_index] = well_contour
                        except ContourTooSmallException as e:
                            continue
            iterations += 1
        return list(zip(self.matched_well_contours, self.fast_contours))

    def match_contours_simplified(self): #something to try out
        for contour_index, fast_contour in enumerate(self.fast_contours):
            for threshold_index, _ in enumerate(self.well_thresholds):
                for well_contour in self.well_images[threshold_index].contours:
                    try:
                        well_contour = Contour(well_contour, threshold_index)
                        score = fast_contour.get_matching_score(well_contour)
                        if score < fast_contour.score and score <= 200:
                                fast_contour.match_contour(well_contour, score)
                                self.matched_well_contours[contour_index] = well_contour
                    except ContourTooSmallException as e:
                        continue
        return list(zip(self.matched_well_contours, self.fast_contours))

    def get_point_descriptors(self, points, image):
        sift = cv2.SIFT_create()
        keypoints = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=20) for p in points]
        keypoints, descriptors = sift.compute(image, keypoints)
        return descriptors

    def match_points(self, well_contour, fast_contour):
        well_points = well_contour.get_points()
        fast_points = fast_contour.get_points()
        well_descriptors = self.get_point_descriptors(well_points, self.well_images[well_contour.well_image_threshold_index].thresholded)
        fast_descriptors = self.get_point_descriptors(fast_points, self.fast_image.thresholded)
        bf = cv2.BFMatcher()
        matches = bf.match(fast_descriptors, well_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        fast_matched_points = [fast_points[m.queryIdx] for m in matches]
        well_matched_points = [well_points[m.trainIdx] for m in matches]
        return well_matched_points, fast_matched_points

    def alignImages(self):
        matched_contours = self.match_contours()
        well_points_combined = []
        fast_points_combined = []
        for well_contour, fast_contour in matched_contours:
            well_matched_points, fast_matched_points = self.match_points(well_contour, fast_contour)
            well_points_combined.extend(well_matched_points)
            fast_points_combined.extend(fast_matched_points)
        # Expose the SIFT/BFMatcher point correspondences (pre-RANSAC) for visualization.
        self.well_points_combined = well_points_combined
        self.fast_points_combined = fast_points_combined
        if len(well_points_combined) >= 4:
            well_points = np.float32(well_points_combined)
            fast_points = np.float32(fast_points_combined)
            H, mask = cv2.findHomography(fast_points, well_points, cv2.RANSAC, 5.0)
            warped_image = cv2.warpPerspective(self.fast_image.image, H, (self.image_width, self.image_height))
        else:
            # print("Not enough points for homography.")
            pass
        return warped_image, H

def save_transformation(json_path, right_filename, matrix):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {'matrices': []}

    found = False
    for item in data['matrices']:
        if item['name'] == right_filename:
            item['matrix'] = matrix.tolist()
            found = True
            break
    
    if not found:
        named_matrix = {'name': right_filename, 'matrix': matrix.tolist()}
        data['matrices'].append(named_matrix)

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def create_fast_transforms_json(transforms_json_path, dataparser_transforms_path, fast_transforms_json_path):
    with open(transforms_json_path, 'r') as f:
        transforms = json.load(f)
    number_of_frames = int(len(transforms['frames'])/2)

    with open(dataparser_transforms_path, 'r') as f:
        dataset_transform = json.load(f)
    scale_factor = dataset_transform['scale']
    normalization_mat = np.eye(4)
    normalization_mat[:3, :] = np.array(dataset_transform['transform'])

    camera_dict = {}
    camera_dict["fps"] = 1
    camera_dict["camera_type"] = "perspective"
    camera_dict["render_width"] = 960
    camera_dict["render_height"] = 960
    camera_dict["smoothness_value"] = 0
    camera_dict["is_cycle"] = False
    camera_dict["camera_path"] = []

    frame_length = 0
    for i in range(number_of_frames, 2*number_of_frames):
        C2W = transforms['frames'][i]['transform_matrix']
        C2W = normalization_mat @ C2W 
        C2W[:3, 3] *= scale_factor
        camera_dict["camera_path"].append(
        {
            "camera_to_world": C2W.flatten().tolist(),
            "fov": 120,
            "aspect": 1,
        })
        frame_length += 1            
    camera_dict["seconds"] = float(frame_length)

    with open(fast_transforms_json_path, 'w') as f:
        json.dump(camera_dict, f, indent=4)

def copy_images(src_folder, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)

    files = os.listdir(src_folder)
    for filename in files:
        if filename.endswith(".png"):
            src_file = os.path.join(src_folder, filename)
            dest_file = os.path.join(dest_folder, filename)
            shutil.copy(src_file, dest_file)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_dir", type=str, default="/mnt/data/scene/scene_ns/", help="Nerfstudio directory of the scene")
    argparser.add_argument("--config", type=str, default="outputs/scene_ns/lantern-nerfacto/2024-06-20_142413/config.yml", help="Nerfstudio config file")
    argparser.add_argument("--no_render", action='store_true', default=False, help="Doesn't render images for alignment if flag is present")
    argparser.add_argument("--no_copy", action='store_true', default=False, help="Doesn't copy the original data if flag is present. Use if already copied.")
    args = argparser.parse_args()

    json_path = os.path.join(args.input_dir, 'alignment_matrices.json')
    transfoms_json_path = os.path.join(args.input_dir, 'transforms.json')
    fast_transforms_json_path = os.path.join(args.input_dir, 'fast_transforms.json')
    dataparser_transforms_path = args.config.replace('config.yml', 'dataparser_transforms.json')
    rendered_for_alignment_path = os.path.join(args.input_dir, 'rendered_for_alignment')
    images_path_aligned = os.path.join(args.input_dir, 'images')
    images_path_unaligned = images_path_aligned + '_unaligned'
    masks_path_aligned = os.path.join(args.input_dir, 'masks')
    masks_path_unaligned = masks_path_aligned + '_unaligned'
    merged_images = os.path.join(images_path_aligned, 'merged')

    if not args.no_render:
        print("Rendering images for alignment...")
        create_fast_transforms_json(transfoms_json_path, dataparser_transforms_path, fast_transforms_json_path)
        subprocess.run(["ns-render", "camera-path", "--load-config=" + args.config, "--camera-path-filename=" + fast_transforms_json_path, "--output-path=" + rendered_for_alignment_path,  "--output-format=images", "--image-format=png"])

    if not args.no_copy:
        print("Copying data...")
        subprocess.run(["mv", images_path_aligned, images_path_unaligned])
        copy_images(images_path_unaligned, images_path_aligned)
        subprocess.run(["mv", masks_path_aligned, masks_path_unaligned])
        copy_images(masks_path_unaligned, masks_path_aligned)
    os.makedirs(merged_images, exist_ok=True)

    print("Aligning images...")
    for i, right_name in enumerate(f for f in os.listdir(images_path_unaligned) if f.startswith("right") and f.endswith(".png")):
        left_rendered_image_path = os.path.join(rendered_for_alignment_path, f"{i + 1:05}.png")
        left_image = cv2.cvtColor(cv2.imread(left_rendered_image_path), cv2.COLOR_BGR2RGB)
        right_path = os.path.join(images_path_unaligned, right_name)
        right_image = cv2.cvtColor(cv2.imread(right_path), cv2.COLOR_BGR2RGB)

        try:
            align_images = AlignImages(left_image, right_image)
            warped_image, matrix = align_images.alignImages()
            print(f"Aligned: {right_name}")

        except Exception as e:
            warped_image = right_image
            matrix = np.zeros((3, 3))
            print(f"Failed to align: {right_name}")

        cv2.imwrite(os.path.join(images_path_aligned, right_name), cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR))
        save_transformation(json_path, right_name, matrix)

        # Masks
        previous_mask_path = os.path.join(masks_path_unaligned, right_name)
        previous_mask = cv2.cvtColor(cv2.imread(previous_mask_path), cv2.COLOR_BGR2GRAY)
        warped_previous_mask = cv2.warpPerspective(previous_mask, matrix, (previous_mask.shape[1], previous_mask.shape[0]),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(os.path.join(masks_path_aligned, right_name), warped_previous_mask)

        previous_saturation_mask_path = previous_mask_path.replace(".png", "_saturation_mask.png")
        previous_saturation_mask = cv2.cvtColor(cv2.imread(previous_saturation_mask_path), cv2.COLOR_BGR2GRAY)
        warped_saturation_masks = cv2.warpPerspective(previous_saturation_mask, matrix, (previous_saturation_mask.shape[1], previous_saturation_mask.shape[0]),
                                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(os.path.join(masks_path_aligned, right_name.replace(".png", "_saturation_mask.png")), warped_saturation_masks)

        # Merged images
        alpha = 50.0/100.0
        merged_before_image = cv2.addWeighted(left_image, 1.0-alpha, right_image, alpha, 0)
        merged_before_masked_image = cv2.bitwise_or(merged_before_image, merged_before_image, mask=previous_mask)
        merged_before_path = os.path.join(merged_images, right_name).replace(".png", "_before.png")
        cv2.imwrite(merged_before_path, cv2.cvtColor(merged_before_masked_image, cv2.COLOR_RGB2BGR))

        merged_after_image = cv2.addWeighted(left_image, 1.0-alpha, warped_image, alpha, 0)
        merged_after_masked_image = cv2.bitwise_or(merged_after_image, merged_after_image, mask=warped_previous_mask)
        merged_after_path = os.path.join(merged_images, right_name).replace(".png", "_after.png")
        cv2.imwrite(merged_after_path, cv2.cvtColor(merged_after_masked_image, cv2.COLOR_RGB2BGR))

    print("All done :D")
