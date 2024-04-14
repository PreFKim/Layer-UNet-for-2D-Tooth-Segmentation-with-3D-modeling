import nibabel as nib
import numpy as np

import cv2

import os
import glob
import tqdm
import argparse


def nii2img(args):

    if args.for_label :
        args.save_path = args.save_path + "(label)"

    if os.path.exists(args.save_path)==False:
        os.makedirs(args.save_path)

    for filename in glob.glob(os.path.join(args.nii_path,"*.nii")):
        nii = nib.load(filename).get_fdata()

        mn = args.min if args.min is not None else np.min(img)
        mx = args.max if args.max is not None else np.max(img)

        nii = (nii-mn)/(mx-mn)
        print(filename, nii.shape, np.sum(nii.shape), np.min(nii), np.max(nii))

        for i in tqdm.tqdm(range(nii.shape[2])):
            img = (nii[..., i]*255)

            if args.for_label:
                img = img - 1
                img[img<0] = 0

            img = img.astype(np.uint8)

            cv2.imwrite(
                os.path.join(
                    args.save_path,
                    f"{os.path.basename(filename)}-slice{i:03d}_z.png"
                ), img)
        

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nii_path", type=str, default="./data/nii")
    parser.add_argument("--save_path", type=str, default="./data/img")
    parser.add_argument("--for_label", action="store_true")

    parser.add_argument("--min", type=int, default=None)
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()

    nii2img(args)