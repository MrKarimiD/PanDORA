import argparse
import json
import os

import cv2
import numpy as np

# fonction that apply the correction on the colors on the input images


def apply_correction(coefs, img, CORRECTION_CURVE_TYPE) -> np.ndarray:
    # check that img is in range 0-255
    assert img.dtype == np.uint8
    
    B, G, R = coefs
    original_shape = img.shape
    img = img.astype(np.float32)
    img = img.reshape(-1, 3)
    if CORRECTION_CURVE_TYPE == "deg2":
        img[:, 0] = B[0] + img[:, 0] * B[1] + img[:, 0] ** 2 * B[2]
        img[:, 1] = G[0] + img[:, 1] * G[1] + img[:, 1] ** 2 * G[2]
        img[:, 2] = R[0] + img[:, 2] * R[1] + img[:, 2] ** 2 * R[2]
    if CORRECTION_CURVE_TYPE == "deg3":
        img[:, 0] = B[0] + img[:, 0] * B[1] + \
            img[:, 0] ** 2 * B[2] + img[:, 0] ** 3 * B[3]
        img[:, 1] = G[0] + img[:, 1] * G[1] + \
            img[:, 1] ** 2 * G[2] + img[:, 0] ** 3 * G[3]
        img[:, 2] = R[0] + img[:, 2] * R[1] + \
            img[:, 2] ** 2 * R[2] + img[:, 0] ** 3 * R[3]
    elif CORRECTION_CURVE_TYPE == "deg1":
        img[:, 0] = B[0] + img[:, 0] * B[1]
        img[:, 1] = G[0] + img[:, 1] * G[1]
        img[:, 2] = R[0] + img[:, 2] * R[1]
    elif CORRECTION_CURVE_TYPE == "deg2_no_offset":
        img[:, 0] = img[:, 0] * B[0] + img[:, 0] ** 2 * B[1]
        img[:, 1] = img[:, 1] * G[0] + img[:, 1] ** 2 * G[1]
        img[:, 2] = img[:, 2] * R[0] + img[:, 2] ** 2 * R[1]
    elif CORRECTION_CURVE_TYPE == "deg3_no_offset":
        img[:, 0] = img[:, 0] * B[0] + \
            img[:, 0] ** 2 * B[1] + img[:, 0] ** 3 * B[2]
        img[:, 1] = img[:, 1] * G[0] + \
            img[:, 1] ** 2 * G[1] + img[:, 1] ** 3 * G[2]
        img[:, 2] = img[:, 2] * R[0] + \
            img[:, 2] ** 2 * R[1] + img[:, 2] ** 3 * R[2]
    elif CORRECTION_CURVE_TYPE == "deg1_no_offset":
        img[:, 0] = img[:, 0] * B[0]
        img[:, 1] = img[:, 1] * G[0]
        img[:, 2] = img[:, 2] * R[0]
    img = img.clip(0, 255)
    img = img.reshape(original_shape)
    return img


def extract_frames(pathToVideo, fps, resize=False, width=960, height=480, jpg=False, frames_name=None, out_path=None):

    if out_path is None:
        frames_path = "frames"
    else:
        frames_path = os.path.join(out_path, "frames")

    if not os.path.exists(frames_path):
        os.mkdir(frames_path)

    # extract frames from the path video
    if frames_name is None:
        os.system(
            '''ffmpeg -i "{}" -s {}x{} -vf fps={} {}/out%d.png'''.format(pathToVideo, width, height, str(fps), frames_path))
    else:
        os.system(
            '''ffmpeg -i "{}" -s {}x{} -vf fps={} {}/{}%d.png'''.format(pathToVideo, width, height, str(fps), frames_path, frames_name))

    if resize:
        imgs = os.listdir(frames_path)
        for file in imgs:
            path = os.path.join(frames_path, file)
            frame = cv2.imread(path)
            frame = cv2.resize(frame, (width, height))
            cv2.imwrite(path, frame)

    if jpg:
        imgs = os.listdir(frames_path)
        for file in imgs:
            path = os.path.join(frames_path, file)
            frame = cv2.imread(path)
            if path.find(".png") != -1:
                cv2.imwrite(path.replace(".png", ".jpg"), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(path)
            elif path.find(".JPG") != -1:
                cv2.imwrite(path.replace(".JPG", ".jpg"), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                os.remove(path)


def linearize_frames(out_path=None):
    # inearize all the images in the frame folder
    # parameter of the color curve. stored in a dict in a json file resulting from the macduff_modified.py code
    with open('data.json') as f:
        data = json.load(f)
    CORRECTION_CURVE_TYPE = data["type"]

    if out_path is None:
        frames_path = "frames"
        lin_path = "linearized"
    else:
        frames_path = os.path.join(out_path, "frames")
        lin_path = os.path.join(out_path, "linearized")

    if not os.path.exists(lin_path):
        os.mkdir(lin_path)

    # calling the apply correction fonction for as long as there is images in the frames folder

    dirs = os.listdir(frames_path)
    for file in dirs:
        im = cv2.imread(os.path.join(frames_path, file))
        im_corrected = apply_correction(
            (data["B"], data["G"], data["R"]), im, CORRECTION_CURVE_TYPE)
        cv2.imwrite(lin_path+"/out_lin"+file, im_corrected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract and linearize images')
    parser.add_argument('-e', '--extract', action='store_true',
                        help='if you want to extract frame of video. If so, --path and --fps are required')
    parser.add_argument('-l', '--linearize', action='store_true',
                        help='if you want to linearize the images in the "frames" folder')
    parser.add_argument('-p', '--path', type=str, metavar='',
                        help="path to the video you want to extract frame from")
    parser.add_argument('-o', '--out_path', type=str, metavar='',
                        help="path to the output frames")
    parser.add_argument('-f', '--fps', type=float, metavar='',
                        help="number of frame per second you want to extract from the video")
    parser.add_argument('-r', '--resize', action='store_true',
                        help="resize the extracted frames")
    parser.add_argument('--height', type=int, metavar='',
                        help="height of extracted frames")
    parser.add_argument('-w', '--width', type=int, metavar='',
                        help="width of extracted frames")
    parser.add_argument('--jpg', action='store_true',
                        help='save the frames in jpg format')
    args = parser.parse_args()

    # ============== START MAIN =============
    #
    print("=====================")
    print("        Start        ")
    print("=====================")

    # check up parameters
    if args.extract:
        if args.path == None:
            print("--path is empty, don't forget to input a path")
            exit()
        if args.fps == None:
            print("--fps is empty, don't forget to specify the fps")
            exit()
        if args.fps == 0:
            print("--fps is 0, it has to be a float higher than 0")
            exit()
    if args.extract == False and args.linearize == False:
        print("It did nothing because you didnt ask for neither extract (-e) nor linearize (-l)")

    # ================================================================================================
    # =======================variable to modify - Path to the video===================================
    # ================================================================================================
    pathToVideo = args.path
    # ================================================================================================
    # ================================================================================================
    # ================================================================================================

    if args.extract:
        extract_frames(pathToVideo, args.fps, args.resize,
                       args.width, args.height, args.jpg, None, args.out_path)

    if args.linearize:
        linearize_frames(args.out_path)

    print("=====================")
    print("      The end        ")
    print("=====================")