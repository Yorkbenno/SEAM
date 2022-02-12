import argparse
import json
import os
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import voc12.data
from torch.utils.data import DataLoader
from tool import pyutils, imutils, torchutils
import importlib
from torchvision import transforms
import torch
import torch.nn.functional as F
import shutil
from multiprocessing import Array, Process
# from utils.util import chunks
import random

def chunks(lst, num_workers=None, n=None):
    """
    a helper function for seperate the list to chunks

    Args:
        lst (list): the target list
        num_workers (int, optional): Default is None. When num_workers are not None, the function divide the list into num_workers chunks
        n (int, optional): Default is None. When the n is not None, the function divide the list into n length chunks

    Returns:
        llis: a list of small chunk lists
    """
    chunk_list = []
    if num_workers is None and n is None:
        print("the function should at least pass one positional argument")
        exit()
    elif n == None:
        n = int(np.ceil(len(lst)/num_workers))
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list
    else:
        for i in range(0, len(lst), n):
            chunk_list.append(lst[i:i + n])
        return chunk_list

def online_cut_patches(im, im_size=96, stride=32):
    """
    function for crop the image to subpatches, will include corner cases
    the return position (x,y) is the up left corner of the image
    Args:
        im (np.ndarray): the image for cropping
        im_size (int, optional): the sub-image size. Defaults to 56.
        stride (int, optional): the pixels between two sub-images. Defaults to 28.
    Returns:
        (list, list): list of image reference and list of its corresponding positions
    """
    im_list = []
    position_list = []

    h, w = im.shape[:2]
    if h < im_size:
        h_ = np.array([0])
    else:
        h_ = np.arange(0, h - im_size + 1, stride)
        if h % stride != 0:
            h_ = np.append(h_, h-im_size)

    if w < im_size:
        w_ = np.array([0])
    else:
        w_ = np.arange(0, w - im_size + 1, stride)
        if w % stride != 0:
            w_ = np.append(w_, w - im_size)

    if len(im.shape) == 3:
        for i in h_:
            for j in w_:   	
                temp = Image.fromarray(np.uint8(im[i:i+im_size,j:j+im_size,:].copy()))
                im_list.append(temp)
                position_list.append((i,j))
    else:
        for i in h_:
            for j in w_:   	
                temp = Image.fromarray(np.uint8(im[i:i+im_size,j:j+im_size].copy()))
                im_list.append(temp)
                position_list.append((i,j))
    return im_list, position_list

def multiscale_online_crop(im, im_size, stride):
    """
    first resize the image to different scales, then crop according to `im_size`

    Returns:
        scale_im_list: the image list
        scale_position_list: the images position
    """
    # im = Image.fromarray(im)
    # im = np.asarray(im)
    im_list, position_list = online_cut_patches(im, im_size, stride)

    return im_list, position_list

def crop_validation_images(dataset_path, gt_path, bg_path, side_length, stride, validation_folder_name):
    """
    if the scales are not modified, this function can run only once.
    crop the validation images to reduce the validation time
    the output is in `validation_cam_folder_name/crop_images`
    images are stored according to the image name

    Args:
        dataset_path (str): the validation dataset path
        side_length (int): the crop size
        stride (int): the distance between two crops
        scales (list): a list of scales to crop
        validation_cam_folder_name (str): the destination to store the validation cam
    """
    images = os.listdir(dataset_path)
    if not os.path.exists(f'{validation_folder_name}/img'):
        os.mkdir(f'{validation_folder_name}/img')
    if not os.path.exists(f'{validation_folder_name}/mask'):
        os.mkdir(f'{validation_folder_name}/mask')
    if not os.path.exists(f'{validation_folder_name}/background-mask'):
        os.mkdir(f'{validation_folder_name}/background-mask')
    
    with open(f'../WSSS4LUAD/val_image_label/groundtruth.json') as f:
        big_labels = json.load(f)
    
    for image in tqdm(images):
        if int(image.split('.')[0]) < 31:
            label = big_labels[image]
            image_path = os.path.join(dataset_path, image)
            gt_mask_path = os.path.join(gt_path, image)
            bg_mask_path = os.path.join(bg_path, image)
            shutil.copyfile(image_path, f'{validation_folder_name}/img/{image.split(".")[0]}_{label}.png')
            shutil.copyfile(gt_mask_path, f'{validation_folder_name}/mask/{image.split(".")[0]}_{label}.png')
            shutil.copyfile(bg_mask_path, f'{validation_folder_name}/background-mask/{image.split(".")[0]}_{label}.png')
        else:
            image_path = os.path.join(dataset_path, image)
            gt_mask_path = os.path.join(gt_path, image)
            bg_mask_path = os.path.join(bg_path, image)
            im = np.asarray(Image.open(image_path))
            im_list, position_list = multiscale_online_crop(im, side_length, stride)
            gt_im = np.asarray(Image.open(gt_mask_path))
            gt_list, _ = multiscale_online_crop(gt_im, side_length, stride)
            bg_im = np.asarray(Image.open(bg_mask_path))
            bg_list, _ = multiscale_online_crop(bg_im, side_length, stride)
            label = big_labels[image]

            for j in range(len(im_list)):
                im_list[j].save(f'{validation_folder_name}/img/{image.split(".")[0]}_{position_list[j]}_{label}.png')
            for j in range(len(gt_list)):
                gt_list[j].save(f'{validation_folder_name}/mask/{image.split(".")[0]}_{position_list[j]}_{label}.png')
            for j in range(len(bg_list)):
                bg_list[j].save(f'{validation_folder_name}/background-mask/{image.split(".")[0]}_{position_list[j]}_{label}.png')

def prepare_wsss(side_length: int, stride: int) -> None:
    """
    offline crop the images into wsss_valid_out_cam/crop_images

    Args:
        side_length (int): the crop image length
        stride (int): the steps for cutting a new image
    """
    validation_folder_name = 'wsss_valid'
    validation_dataset_path = '../WSSS4LUAD/Dataset_wsss/2.validation/img'
    gt_path = '../WSSS4LUAD/Dataset_wsss/2.validation/mask'
    bg_path = '../WSSS4LUAD/Dataset_wsss/2.validation/background-mask'
    if not os.path.exists(validation_folder_name):
        os.mkdir(validation_folder_name)

    print('crop validation set images ...')
    crop_validation_images(validation_dataset_path, gt_path, bg_path, side_length, stride, validation_folder_name)
    print('cropping finishes!')

def prepare_crag(side_length: int, stride: int) -> None:
    """
    offline crop the images into crag_valid_out_cam/crop_images

    Args:
        side_length (int): the crop image length
        stride (int): the steps for cutting a new image
    """
    validation_folder_name = 'crag_valid'
    validation_dataset_path = '../WSSS4LUAD/Dataset_crag/2.validation/img'
    gt_path = '../WSSS4LUAD/Dataset_crag/2.validation/mask'
    # bg_path = '../WSSS4LUAD/Dataset_crag/2.validation/background-mask'
    if not os.path.exists(validation_folder_name):
        os.mkdir(validation_folder_name)

    print('crop validation set images ...')
    crop_crag_validation_images(validation_dataset_path, gt_path, side_length, stride, validation_folder_name)
    print('cropping finishes!')

def crop_crag_validation_images(dataset_path, gt_path, side_length, stride, validation_folder_name):

    images = os.listdir(dataset_path)
    if not os.path.exists(f'{validation_folder_name}/img'):
        os.mkdir(f'{validation_folder_name}/img')
    if not os.path.exists(f'{validation_folder_name}/mask'):
        os.mkdir(f'{validation_folder_name}/mask')
    
    for image in tqdm(images):
        image_path = os.path.join(dataset_path, image)
        gt_mask_path = os.path.join(gt_path, image)

        im = np.asarray(Image.open(image_path))
        im_list, position_list = multiscale_online_crop(im, side_length, stride)
        gt_im = np.asarray(Image.open(gt_mask_path))
        gt_list, _ = multiscale_online_crop(gt_im, side_length, stride)

        for j in range(len(im_list)):
            im_list[j].save(f'{validation_folder_name}/img/{image.split(".")[0]}_{position_list[j]}_[1, 1].png')
        for j in range(len(gt_list)):
            gt_list[j].save(f'{validation_folder_name}/mask/{image.split(".")[0]}_{position_list[j]}_[1, 1].png')

def get_overall_crag_valid_score(pred_image_path, groundtruth_path, num_workers=5, mask_path=None, num_class=3):

    image_names = list(map(lambda x: x.split('.')[0], os.listdir(pred_image_path)))
    random.shuffle(image_names)
    image_list = chunks(image_names, num_workers)

    def f(intersection, union, image_list):
        gt_list = []
        pred_list = []

        for im_name in image_list:
            cam = np.load(os.path.join(pred_image_path, f"{im_name}.npy"), allow_pickle=True).astype(np.uint8).reshape(-1)
            groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}.png")).reshape(-1)
            
            if mask_path:
                mask = np.asarray(Image.open(mask_path + f"/{im_name}.png")).reshape(-1)
                cam = cam[mask == 0]
                groundtruth = groundtruth[mask == 0]
            
            gt_list.extend(groundtruth)
            pred_list.extend(cam)

        pred = np.array(pred_list)
        real = np.array(gt_list)
        for i in range(num_class):
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                intersection[i] += inter
                union[i] += u

    intersection = Array("d", [0] * num_class)
    union = Array("d", [0] * num_class)
    p_list = []
    for i in range(len(image_list)):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    eps = 1e-7
    total = 0
    for i in range(num_class):
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
    return total / num_class

def get_overall_valid_score(pred_image_path, groundtruth_path, num_workers=5, mask_path=None, num_class=3):
    """
    get the scores with validation groundtruth, the background will be masked out
    and return the score for all photos

    Args:
        pred_image_path (str): the prediction require to test, npy format
        groundtruth_path (str): groundtruth images, png format
        num_workers (int): number of process in parallel, default is 5.
        mask_path (str): the white background, png format
        num_class (int): default is 3.

    Returns:
        float: the mIOU score
    """
    image_names = list(map(lambda x: x.split('.')[0], os.listdir(pred_image_path)))
    random.shuffle(image_names)
    image_list = chunks(image_names, num_workers)

    def f(intersection, union, image_list):
        gt_list = []
        pred_list = []

        for im_name in image_list:
            cam = np.load(os.path.join(pred_image_path, f"{im_name}.npy"), allow_pickle=True).astype(np.uint8).reshape(-1)
            groundtruth = np.asarray(Image.open(groundtruth_path + f"/{im_name}.png")).reshape(-1)
            
            if mask_path:
                mask = np.asarray(Image.open(mask_path + f"/{im_name}.png")).reshape(-1)
                cam = cam[mask == 0]
                groundtruth = groundtruth[mask == 0]
            
            gt_list.extend(groundtruth)
            pred_list.extend(cam)

        pred = np.array(pred_list)
        real = np.array(gt_list)
        for i in range(num_class):
            if i in pred:
                inter = sum(np.logical_and(pred == i, real == i))
                u = sum(np.logical_or(pred == i, real == i))
                intersection[i] += inter
                union[i] += u

    intersection = Array("d", [0] * num_class)
    union = Array("d", [0] * num_class)
    p_list = []
    for i in range(len(image_list)):
        p = Process(target=f, args=(intersection, union, image_list[i]))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()

    eps = 1e-7
    total = 0
    for i in range(num_class):
        class_i = intersection[i] / (union[i] + eps)
        total += class_i
    return total / num_class

if __name__ == "__main__":
    # prepare_crag(224, 224)
    result = get_overall_valid_score("validoutcampred", "wsss_valid/mask", mask_path="wsss_valid/background-mask")
    print(result)