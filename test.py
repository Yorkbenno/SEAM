# import argparse
import os
from PIL import Image
import numpy as np
import png

if __name__ == '__main__':
    mask_folder_path = 'train_pseudo_mask'
    gt_path = 'processed_train_mask'

    # for mask_name in os.listdir(gt_path):
    #     mask = np.asarray(Image.open(os.path.join(gt_path, mask_name)))
    #     # this three steps, convert tumor to 0, background to 2
    #     mask[mask > 0] = 1
    #     palette = [(0, 64, 128), (243, 152, 0)]
    #     with open(os.path.join(gt_path, f'{mask_name.split(".")[0]}.png'), 'wb') as f:
    #         w = png.Writer(mask.shape[1], mask.shape[0],palette=palette, bitdepth=8)
    #         w.write(f, mask.astype(np.uint8))
    
    intersection = [0, 0]
    union = [0, 0]
    for im_name in os.listdir(mask_folder_path):
        train_mask_name = im_name.split('.')[0] + "_anno.png"
        pred = np.array(Image.open(os.path.join(mask_folder_path, im_name)))
        real = np.array(Image.open(os.path.join(gt_path, train_mask_name)))
        for i in range(2):
            # if i in pred:
            inter = np.sum(np.logical_and(pred == i, real == i))
            u = np.sum(np.logical_or(pred == i, real == i))
            intersection[i] += inter
            union[i] += u

    eps = 1e-7
    total = 0
    for i in range(2):
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
    print(total / 2)