import numpy as np
from PIL import Image
import os
import png

def process_mask(mask_folder_path, destination):
    for mask_name in os.listdir(mask_folder_path):
        mask = np.asarray(Image.open(os.path.join(mask_folder_path, mask_name))).copy()
        
        mask[mask > 0] = 1
        palette = [(64, 128, 0), (0, 64, 128)]
        with open(os.path.join(destination, f'{mask_name.split(".")[0]}.png'), 'wb') as f:
            w = png.Writer(mask.shape[1], mask.shape[0], palette=palette, bitdepth=8)
            w.write(f, mask.astype(np.uint8))

def online_cut_patches(im, im_size, stride):
    """
    function for crop the image to subpatches, will include corner cases
    the return position (x,y) is the up left corner of the image
    Args:
        im (np.ndarray): the image for cropping
        im_size (int): the sub-image size.
        stride (int): the pixels between two sub-images.
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
    
def glas_join_crops_back(cropped_cam_path: str, origin_ims_path: str, side_length: int, stride: int, is_train: bool) -> None:
    """
    merge the cropped image mask to the original image size and save in the `cropped_cam_path` folder

    Args:
        cropped_cam_path (str): the predicted crop masks path
        origin_ims_path (str): the original image path
        side_length (int): the crop size
        stride (int): the step between crop images
        is_train: whether this function is performed on the training set
    """
    partial_image_list = os.listdir(cropped_cam_path)
    # make a dict to tract wich images are in a group and should be merged back
    ims_dict = {}
    for partial_mask in partial_image_list:
        _, corresponding_im, index = partial_mask.split('_')
        index = int(index.split('-')[0])
        if is_train:
            if f'train_{corresponding_im}.bmp' not in ims_dict:
                ims_dict[f'train_{corresponding_im}.bmp'] = {}
            ims_dict[f'train_{corresponding_im}.bmp'][index] = os.path.join(cropped_cam_path, partial_mask)
        else:
            if f'{corresponding_im}.bmp' not in ims_dict:
                ims_dict[f'{corresponding_im}.bmp'] = {}
            ims_dict[f'{corresponding_im}.bmp'][index] = os.path.join(cropped_cam_path, partial_mask)

    # merge images to the size in validation set part
    for origin_im in os.listdir(origin_ims_path):
        im = np.asarray(Image.open(os.path.join(origin_ims_path, origin_im)))
        complete_mask = np.zeros((im.shape[0], im.shape[1]))
        sum_counter = np.zeros_like(complete_mask)
        _, position_list = online_cut_patches(im, im_size=side_length, stride=stride)

        for i in range(len(position_list)):
            partial_mask = np.load(ims_dict[origin_im][i], allow_pickle=True)
            position = position_list[i]
            complete_mask[position[0]:position[0]+side_length, position[1]:position[1]+side_length] += partial_mask
            sum_counter[position[0]:position[0]+side_length, position[1]:position[1]+side_length] += 1

        complete_mask = np.rint(complete_mask / sum_counter)
        palette = [(64, 128, 0), (0, 64, 128)]
        with open(os.path.join(cropped_cam_path, f'{origin_im.split(".")[0]}.png'), 'wb') as f:
            w = png.Writer(complete_mask.shape[1], complete_mask.shape[0], palette=palette, bitdepth=8)
            w.write(f, complete_mask.astype(np.uint8))

def process_mask(mask_folder_path, destination):
    for mask_name in os.listdir(mask_folder_path):
        mask = np.asarray(Image.open(os.path.join(mask_folder_path, mask_name))).copy()
        # this three steps, convert tumor to 0, background to 2
        mask[mask > 0] = 1
        palette = [(0, 64, 128), (64, 128, 0)]
        with open(os.path.join(destination, f'{mask_name.split(".")[0]}.png'), 'wb') as f:
            w = png.Writer(mask.shape[1], mask.shape[0], palette=palette, bitdepth=8)
            w.write(f, mask.astype(np.uint8))       
            
process_mask("/tmp/sccam_merge", "sccam_mask")
# glas_join_crops_back("outcampred", "/home/yyubm/OEEM/classification/glas/1.training/origin_ims", 112, 56, True)
# new_dir = "train_pseudo_mask"
# old_dir = "outcampred"
# for filename in os.listdir(old_dir):
#     if filename.endswith(".png"):
#         os.rename(os.path.join(old_dir, filename), os.path.join(new_dir, filename))
# process_mask("train_mask", "processed_train_mask")