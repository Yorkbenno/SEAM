import os
from PIL import Image
import numpy as np
# from utils.util import online_cut_patches
import png

pseudo_mask_path = 'validoutcampred'
origin_ims_path = '../WSSS4LUAD/Dataset_crag/3.testing/img'
destination = 'crag_seam_testPseudoMask'

if not os.path.exists(destination):
    os.mkdir(destination)

ims_dict = {}

for partial_mask in os.listdir(pseudo_mask_path):
    im_index, s = partial_mask.split('_')
    im_index = int(im_index)
    # position = s.split('-')[0]
    if im_index not in ims_dict:
        ims_dict[im_index] = []
    ims_dict[im_index].append(os.path.join(pseudo_mask_path, partial_mask))


for origin_im in os.listdir(origin_ims_path):
    im = np.asarray(Image.open(os.path.join(origin_ims_path, origin_im)))
    complete_mask = np.zeros((im.shape[0], im.shape[1]))
    sum_counter = np.zeros_like(complete_mask)
    im_index = int(origin_im.split('.')[0])

    for im_path in ims_dict[im_index]:
        partial_mask = np.load(im_path, allow_pickle=True)
        position_path = im_path.split('_')[-1].split('-')[0][1:-1].split(',')
        position = tuple((int(position_path[0]), int(position_path[1])))
        # print(position)
        complete_mask[position[0]:position[0]+112, position[1]:position[1]+112] += partial_mask
        sum_counter[position[0]:position[0]+112, position[1]:position[1]+112] += 1

    complete_mask = np.rint(complete_mask / sum_counter)
    palette = [(0, 64, 128), (64, 128, 0), (243, 152, 0), (255, 255, 255)]
    with open(os.path.join(destination, f'{origin_im.split(".")[0]}.png'), 'wb') as f:
        w = png.Writer(complete_mask.shape[1], complete_mask.shape[0], palette=palette, bitdepth=8)
        w.write(f, complete_mask.astype(np.uint8))