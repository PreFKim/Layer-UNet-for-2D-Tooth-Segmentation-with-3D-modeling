import tensorflow as tf
import nibabel as nib
import numpy as np
import meshplot as mp
import cv2, os, tqdm

import skimage.measure
from stl import mesh
import argparse

from losses import Focal_IoU

from cfg import IMAGE_HEIGHT, IMAGE_WIDTH

def dataToMesh(vert, faces):
    mm = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mm.vectors[i][j] = vert[f[j],:]
    return mm

def main(model, args):

    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)

    nii = nib.load(args.nii_path).get_fdata()
    mn = args.min if args.min is not None else np.min(img, (0, 1))
    mx = args.max if args.max is not None else np.max(img, (0, 1))

    nii = (nii - mn) / (mx - mn)

    result = np.zeros_like(nii, dtype= np.float32)

    for i in tqdm.tqdm(range(nii.shape[-2])):

        img = nii[..., i]
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img.reshape(1, IMAGE_HEIGHT, IMAGE_WIDTH, 1)
        
        pred = model.predict(img, verbose=0)[0]

        if type(model.output) == list :
            num_of_output = len(model.output)
            pred = sum(pred) / num_of_output

        pred[pred < args.threshold] = 0.0
        pred[pred >= args.threshold] = 1.0
        result[..., i] = pred

    vertices,faces,_,_ = skimage.measure.marching_cubes(result)
    #Save mesh to file (.stl)
    mm = dataToMesh(vertices, faces)
    mm.save(os.path.join(args.save_path, os.path.basename(args.nii_path)))

    print(f'{os.path.join(args.save_path, os.path.basename(args.nii_path))}에 저장되었습니다.')
    

if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, default="./experiments/example.ckpt")
    parser.add_argument("--nii_path", type=str, default="./data/nii/CT01.nii")
    parser.add_argument("--save_path", type=str, default="./result")
    parser.add_argument("--min", type=int, default=None)
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.checkpoints, custom_objects = {'Focal_IoU':Focal_IoU})

    main(model, args.video_path)