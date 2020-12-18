import numpy as np
import math
import cv2
import random
import os


def generate_image_list(args):
    filenames = os.listdir(args.train_data_dir)
    num_imgs = len(filenames)
    num_ave_aug = int(math.floor(args.augment_num/num_imgs))
    rem = args.augment_num - num_ave_aug*num_imgs
    lucky_seq = [True]*rem + [False]*(num_imgs-rem)
    random.shuffle(lucky_seq)

    img_list = [
        (os.sep.join([args.train_data_dir, filename]), num_ave_aug+1 if lucky else num_ave_aug)
        for filename, lucky in zip(filenames, lucky_seq)
    ]
    return img_list

if cfg.aug_dir and cfg.do_aug:
    img_list = generate_image_list(cfg)
    augment_images(img_list, cfg)

dataset_dir = cfg.aug_dir if cfg.aug_dir else cfg.train_data_dir
file_list = glob(dataset_dir + '/*')
num_valid_data = int(np.ceil(len(file_list) * 0.2))
data_train = data_flow(file_list[:-num_valid_data], cfg.batch_size, cfg.grayscale)
data_valid = data_flow(file_list[-num_valid_data:], cfg.batch_size, cfg.grayscale)
