import argparse
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import PIL.ExifTags
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import warnings 


def apply_correction(coefs, img, CORRECTION_CURVE_TYPE) -> np.ndarray:
    # check that img is in range 0-255
    assert img.dtype == np.uint8
    assert CORRECTION_CURVE_TYPE in ['gamma']
    img = ((img.astype(np.float32) / 255.0) ** np.array(coefs).T) # * 255.0
    # img = ((img.astype(np.float32) / 255.0) ** coefs[None, None, :]) * 255.0
    return img


def hat_weighting_function(pixels, is_lowest = False, is_fastest = False, zmin = 0.0, zmax = 1.0):
    threshold = 0.5 * (zmin + zmax)
    weights = np.zeros(pixels.shape)
    
    assert not(is_lowest and is_fastest)

    # mask_lowerbound = zmin <= pixels <= threshold
    mask_lowerbound = np.greater_equal(pixels, zmin) & np.less_equal(pixels, threshold)
    if not is_lowest:
        weights[mask_lowerbound] = 1.0 * (pixels[mask_lowerbound] - zmin)
    else:
        weights[mask_lowerbound] = 1.0

    # mask_upperbound = threshold <= pixels <= zmax
    mask_upperbound = np.greater_equal(pixels, threshold) & np.less_equal(pixels, zmax)
    if not is_fastest:
        weights[mask_upperbound] = 1.0 * (zmax - pixels[mask_upperbound])
    else:
        weights[mask_upperbound] = 1.0

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/momo/Desktop/Pandora-HDR_NeRF/final_results/meeting_room/GT/HDR_brackets/')
    parser.add_argument('--experiment_location', type=str, required=True)
    args = parser.parse_args()
    
    
    experiment_setup = args.experiment_location + 'capture_settings.json'
    assert os.path.exists(experiment_setup), "!!!! Cannot find the files for exposure setup.  !!!"
    with open(experiment_setup) as f:
        # read shutter_left, shutter_right, iso_left, iso_right
        data = json.load(f)
        reference_exposure = float(data["left"]["shutter_speed"])


    
    GT_directories = [x[0] for x in os.walk(args.data_dir)]
    GT_directories = list(set(GT_directories) - set([args.data_dir]))
    
    for data_dir in GT_directories:
        test_images = []
        print("data_dir: ", data_dir)
        for path in Path(data_dir).rglob('*.JPG'):
            test_images.append(path)

        with open(f'linearize-ricoh-z10-wb-3500.json') as f:
            camera_data = json.load(f)
            CORRECTION_CURVE_TYPE = 'gamma'
        
        img_0 = np.array(Image.open(test_images[0]) )
        weights_all = np.zeros(img_0.shape) # + np.finfo(np.float32).eps
        final_img = np.zeros(img_0.shape)

        fastest_expo = 1000
        lowest_expo = -1000

        data = {}
        
        for pano_addr in tqdm(test_images):
            img_exif = Image.open(pano_addr)
            img = cv2.imread(str(pano_addr))
            exif = {
                PIL.ExifTags.TAGS[k]: v
                for k, v in img_exif._getexif().items()
                if k in PIL.ExifTags.TAGS
            }
            img_corrected = apply_correction((camera_data["b"], camera_data["g"], camera_data["r"]), np.array(img), CORRECTION_CURVE_TYPE)
            # weights = hat_weighting_function(img_corrected)
            # weights_all += weights
            exposure = float(exif['ExposureTime']) / reference_exposure
            data[str(exposure)] = img_corrected
            # final_img += (weights * img_corrected) / exposure

            if exposure < fastest_expo:
                fastest_expo = exposure

            if lowest_expo < exposure:
                lowest_expo = exposure

        for key in data.keys():
            if key == str(lowest_expo):
                weights = hat_weighting_function(data[key], is_lowest = True)
            elif key == str(fastest_expo):
                weights = hat_weighting_function(data[key], is_fastest = True)
            else:
                weights = hat_weighting_function(data[key])
            weights_all += weights
            final_img += (weights * data[key]) / float(key)


        # import pdb; pdb.set_trace()
        # final_img[weights_all == 0] = (1 / fastest_expo)
        # weights_all[weights_all == 0] = 1.0

            
        final_img_normal = np.float32(final_img / weights_all)
        output_name = data_dir.split('/')[-1]
        cv2.imwrite(data_dir + '/'+ output_name + '.exr', final_img_normal )
        cv2.imwrite(data_dir + '/weights_all.exr', np.float32(weights_all) )