import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

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
        # show_contours(self.well_images[5].thresholded, self.well_images[4].thresholded, matched_contours)
        well_points_combined = []
        fast_points_combined = []
        for well_contour, fast_contour in matched_contours:
            well_matched_points, fast_matched_points = self.match_points(well_contour, fast_contour)
            well_points_combined.extend(well_matched_points)
            fast_points_combined.extend(fast_matched_points)
        # show_points(well_image, fast_image, well_points_combined, fast_points_combined)
        if len(well_points_combined) >= 4:
            well_points = np.float32(well_points_combined)
            fast_points = np.float32(fast_points_combined)
            H, mask = cv2.findHomography(fast_points, well_points, cv2.RANSAC, 5.0)
            warped_image = cv2.warpPerspective(self.fast_image.image, H, (self.image_width, self.image_height))
        else:
            print("Not enough points for homography.")
        return warped_image, H

def show_contours(well_image, fast_image, matched_contours):
    well_image_c = well_image.copy()
    fast_image_c = fast_image.copy()

    for well_contour, fast_contour in matched_contours:
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.drawContours(well_image_c, well_contour.contour, -1, color, 2)
        cv2.drawContours(fast_image_c, fast_contour.contour, -1, color, 2)

    fig, axes = plt.subplots(1, 2, figsize=(120, 120), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes[1].imshow(well_image_c)
    axes[1].axis('off')
    axes[0].imshow(fast_image_c)
    axes[0].axis('off')

    fig.canvas.draw()
    plt.show()

def show_points(well_image, fast_image, well_points, fast_points):
    well_image_c = well_image.copy()
    fast_image_c = fast_image.copy()

    for p1 in well_points:
        cv2.circle(well_image_c, tuple(p1), 2, (0, 255, 0), -1)

    for p2 in fast_points:
        cv2.circle(fast_image_c, tuple(p2), 2, (0, 255, 0), -1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(well_image_c)
    axes[0].axis('off')
    axes[0].set_title('Well-exposed Image with Points')
    axes[1].imshow(fast_image_c)
    axes[1].axis('off')
    axes[1].set_title('Aligned Fast-exposed Image')
    plt.show()

def align(well_image, fast_image):
    align_images = AlignImages(well_image, fast_image)
    return align_images.alignImages()

if __name__ == '__main__':
    index = 0
    path_well = '...'
    path_fast = '...'
    well_images = sorted(os.listdir(path_well))
    fast_images = sorted(os.listdir(path_fast))
    well_image = cv2.cvtColor(cv2.imread(os.path.join(path_well, well_images[index])), cv2.COLOR_BGR2RGB)
    fast_image = cv2.cvtColor(cv2.imread(os.path.join(path_fast, fast_images[index])), cv2.COLOR_BGR2RGB)

    warped_image, H = align(well_image, fast_image)