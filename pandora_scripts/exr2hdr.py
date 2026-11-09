import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
import numpy as np
from pathlib import Path
from hdrio import imwrite
import imageio


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
imageio.plugins.freeimage.download()

def convert(src_addr, trg_addr):
    img = imageio.imread(src_addr)
    imwrite( img[:, :, 0:3], trg_addr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdr_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--change_name', action='store_true')
    args = parser.parse_args()

    hdrs = []
    for path in Path(args.hdr_dir).glob('*.exr'):
        if path.is_file():
            hdrs.append(str(path))
    if len(hdrs) == 0:
        raise Exception("No data available!!!")
    hdrs = sorted(hdrs)

    os.makedirs(args.output_dir, exist_ok=True)
    
    for index, src_addr in enumerate(tqdm(hdrs)):
        src_addr = str(src_addr)
        if args.change_name:
            trg_addr = str(Path(args.output_dir) / f"{index:05d}.hdr")
        else:
            trg_addr = src_addr.replace(args.hdr_dir, args.output_dir)
            trg_addr = trg_addr.replace('.exr', '.hdr')
        convert(src_addr, trg_addr)
