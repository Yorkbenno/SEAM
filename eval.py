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

    h, w, _ = im.shape
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

    for i in h_:
        for j in w_:   	
            temp = Image.fromarray(np.uint8(im[i:i+im_size,j:j+im_size,:].copy()))
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
    im = Image.fromarray(im)

    im = np.asarray(im)
    im_list, position_list = online_cut_patches(im, im_size, stride)

    return im_list, position_list

def crop_validation_images(dataset_path, side_length, stride, validation_cam_folder_name):
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
    if not os.path.exists(f'{validation_cam_folder_name}/crop_images'):
        os.mkdir(f'{validation_cam_folder_name}/crop_images')
    with open(f'../WSSS4LUAD/val_image_label/groundtruth.json') as f:
        big_labels = json.load(f)
    for image in tqdm(images):
        if not os.path.exists(f'{validation_cam_folder_name}/crop_images/{image.split(".")[0]}'):
            os.mkdir(f'{validation_cam_folder_name}/crop_images/{image.split(".")[0]}')
        image_path = os.path.join(dataset_path, image)
        im = np.asarray(Image.open(image_path))
        im_list, position_list = multiscale_online_crop(im, side_length, stride)
        label = big_labels[image]
        for j in range(len(im_list)):
            im_list[j].save(f'{validation_cam_folder_name}/crop_images/{image.split(".")[0]}/{position_list[j]}{label}.png')

def prepare_wsss(side_length: int, stride: int) -> None:
    """
    offline crop the images into wsss_valid_out_cam/crop_images

    Args:
        side_length (int): the crop image length
        stride (int): the steps for cutting a new image
    """
    print('start processing validation and test images...')

    validation_cam_folder_name = 'wsss_valid_out_cam'
    validation_dataset_path = '../WSSS4LUAD/Dataset_wsss/2.validation/img'
    if not os.path.exists(validation_cam_folder_name):
        os.mkdir(validation_cam_folder_name)

    print('crop validation set images ...')
    crop_validation_images(validation_dataset_path, side_length, stride, validation_cam_folder_name)
    print('cropping finishes!')

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", default=224, type=int)
    parser.add_argument("--stride", default=224, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--network", default="network.resnet38_SEAM", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--out_cam", default=None, type=str)
    parser.add_argument("--out_crf", default=None, type=str) 
    parser.add_argument("--out_cam_pred", default=None, type=str)
    parser.add_argument("--out_cam_pred_alpha", default=0.26, type=float)
    args = parser.parse_args()
    
    # prepare_wsss(args.crop_size, args.stride)

    crf_alpha = [4,24]
    model = getattr(importlib.import_module(args.network), 'Net')()
    model.load_state_dict(torch.load(args.weights))

    model.eval()
    model.cuda()

    validation_image_path = "wsss_valid_out_cam/crop_images/"
    inter_union = np.zeros((2,3))
    for im_index in tqdm(os.listdir(validation_image_path)):
        images_path = os.path.join(validation_image_path, im_index)

        infer_dataset = voc12.data.MyClsValDatasetMSF(images_path,
                                                    scales=[0.5, 1.0, 1.5, 2.0],
                                                    inter_transform=transforms.Compose(
                                                        [np.asarray,
                                                            model.normalize,
                                                            imutils.HWC_to_CHW]))

        infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=False)

        n_gpus = torch.cuda.device_count()
        model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))

        cams_list = []
        position_list = []
        for img_name, img_list, label, position in infer_data_loader:
            img_name = img_name[0]
            label = label[0]


            img_path = os.path.join(images_path, img_name)
            orig_img = np.asarray(Image.open(img_path))
            orig_img_size = orig_img.shape[:2]

            cam_list = []
            for i, img in enumerate(img_list):
                def _work(i, img):
                    with torch.no_grad():
                        with torch.cuda.device(i%n_gpus):
                            _, cam = model_replicas[i%n_gpus](img.cuda())
                            cam = F.upsample(cam[:,1:,:,:], orig_img_size, mode='bilinear', align_corners=False)[0]
                            cam = cam.cpu().numpy() * label.clone().view(3, 1, 1).numpy()
                            if i % 2 == 1:
                                cam = np.flip(cam, axis=-1)
                            return cam
                
                thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list)), batch_size=12, prefetch_size=0, processes=args.num_workers)
                cam_list = thread_pool.pop_results()
                sum_cam = np.sum(cam_list, axis=0)
                sum_cam[sum_cam < 0] = 0
                cam_max = np.max(sum_cam, (1,2), keepdims=True)
                cam_min = np.min(sum_cam, (1,2), keepdims=True)
                sum_cam[sum_cam < cam_min + 1e-5] = 0
                norm_cam = (sum_cam-cam_min - 1e-5) / (cam_max - cam_min + 1e-5)
                cams_list.append(norm_cam)
                position_list.append(position)

        position_list = np.concatenate(position_list)
        complete_img = np.asarray(Image.open(f'/home/yzouag2/WSSS4LUAD/Dataset_wsss/2.validation/img/{im_index}.png'))
        w, h, _ = complete_img.shape
        sum_cam = np.zeros((3, w, h))
        sum_counter = np.zeros_like(sum_cam)
        
        for k in range(len(cams_list)):
            y, x = position_list[k][0], position_list[k][1]
            crop = cams_list[k]
            sum_cam[:, y:y+args.crop_size, x:x+args.crop_size] += crop
            sum_counter[:, y:y+args.crop_size, x:x+args.crop_size] += 1
        sum_counter[sum_counter < 1] = 1
        norm_cam = sum_cam / sum_counter

        cam_dict = {}
        for i in range(3):
            if label[i] > 1e-5:
                cam_dict[i] = norm_cam[i]

        if args.out_cam is not None:
            np.save(os.path.join(args.out_cam, im_index + '.npy'), cam_dict)

        if args.out_cam_pred is not None:
            bg_score = [np.ones_like(norm_cam[0])*args.out_cam_pred_alpha]
            pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)
            pred = pred.astype(np.uint8)
            np.save(os.path.join(args.out_cam_pred, im_index + '.npy'), pred)
            
            pred = pred.reshape(-1)
            groundtruth = np.asarray(Image.open(f'/home/yzouag2/WSSS4LUAD/Dataset_wsss/2.validation/mask/{im_index}.png')).reshape(-1)
            mask = np.asarray(Image.open(f'/home/yzouag2/WSSS4LUAD/Dataset_wsss/2.validation/background-mask/{im_index}.png')).reshape(-1)
            pred = pred[mask == 0]
            groundtruth = groundtruth[mask == 0]

            for i in range(3):
                if i in pred:
                    inter = sum(np.logical_and(pred == i, groundtruth == i))
                    u = sum(np.logical_or(pred == i, groundtruth == i))
                    inter_union[0,i] += inter
                    inter_union[1,i] += u

        def _crf_with_alpha(cam_dict, alpha):
            v = np.array(list(cam_dict.values()))
            bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
            bgcam_score = np.concatenate((bg_score, v), axis=0)
            crf_score = imutils.crf_inference(complete_img, bgcam_score, labels=bgcam_score.shape[0])

            n_crf_al = dict()

            n_crf_al[0] = crf_score[0]
            for i, key in enumerate(cam_dict.keys()):
                n_crf_al[key+1] = crf_score[i+1]

            return n_crf_al

        if args.out_crf is not None:
            for t in crf_alpha:
                crf = _crf_with_alpha(cam_dict, t)
                folder = args.out_crf + ('_%.1f'%t)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                np.save(os.path.join(folder, im_index + '.npy'), crf)
    
    eps = 1e-7
    total = 0
    for i in range(3):
        class_i = inter_union[0,i] / (inter_union[i] + eps)
        total += class_i
    print(total / 3)

