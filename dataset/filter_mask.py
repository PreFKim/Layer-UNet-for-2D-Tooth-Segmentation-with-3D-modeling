import nibabel as nib
import numpy as np

import cv2

import os
import glob
import tqdm
import argparse


def filter_mask(args):
    if os.path.exists(args.save_path)==False:
        os.makedirs(args.save_path)
        
    for filename in tqdm.tqdm(glob.glob(os.path.join(args.img_path,"*"))):
        mask = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        mask[mask<args.threshold] = 0
        mask[mask>=args.threshold] = 1

        cv2.imwrite(
            os.path.join(
                args.save_path,
                f"{os.path.basename(filename)}"
            ), mask)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="./data/img(label)")
    parser.add_argument("--save_path", type=str, default="./data/mask")
    parser.add_argument("--threshold", type=int, default="255")

    args = parser.parse_args()

    filter_mask(args)