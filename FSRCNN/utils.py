#!/Anaconda3/envs/tf110-gpu-py36/python python
# coding:utf-8

import cv2
import os
import glob
import h5py
from skimage import io as skio
from skimage import color as skco
import random
from math import floor
import struct
import scipy.misc
import tensorflow as tf
from PIL import Image
from scipy.misc import imread
import numpy as np
from multiprocessing import Pool, Lock, active_children
from scipy.misc.pilutil import imread, imsave, imresize
import pdb
import scipy.ndimage
import warnings
warnings.filterwarnings("ignore")

FLAGS = tf.app.flags.FLAGS


def prepare_data(sess, dataset):
    # 选择训练集或测试集下的图片数据
    if FLAGS.is_train:
        data_dir = os.path.join(os.getcwd(), dataset)
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")

    data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))
    return data


def make_data(sess, checkpoint_dir, lr_data, hr_data):
    # 将输入数据制作成.h5文件格式
    if FLAGS.is_train:
        save_path = os.path.join(os.getcwd(), '{}/train.h5'.format(checkpoint_dir))
    else:
        save_path = os.path.join(os.getcwd(), '{}/test.h5'.format(checkpoint_dir))

    with h5py.File(save_path, 'w') as hf:
        hf.create_dataset('lr_data', data=lr_data)
        hf.create_dataset('hr_data', data=hr_data)


def read_data(path):
    # 读取.h5格式的文件，提取其中储存的数据
    with h5py.File(path, 'r') as hf:
        lr_data = np.array(hf.get('lr_data'))
        hr_data = np.array(hf.get('hr_data'))
        return lr_data, hr_data


def merge(images, size):
    # 对输出结果的图像块进行合并，形成输出图
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def modcrop(image, scale=3):
    # 裁剪图片为缩放倍数scale的大小
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def modcrop_same(num_x, num_y, image, lr_size, hr_size, scale=3):
    # 将bicubic和原图等裁剪为输出图的大小
    image_size, label_size, scale = lr_size, hr_size, scale
    x_loc0 = scale * (0 + floor((image_size - 1) / 2)) - floor((label_size + scale) / 2 - 1)
    y_loc0 = scale * (0 + floor((image_size - 1) / 2)) - floor((label_size + scale) / 2 - 1)
    x_loc = x_loc0
    y_loc = y_loc0
    num1 = label_size * num_x
    num2 = label_size * num_y

    if len(image.shape) == 3:
        h = num1 + x_loc
        w = num2 + y_loc
        image1 = image[x_loc:h, y_loc:w, :]
    else:
        h = num1 + x_loc
        w = num2 + y_loc
        image1 = image[x_loc:h, y_loc:w]
    return image1


def modcrop_same_2(num_x, num_y, image, lr_size, hr_size, scale=3):
    # 将bicubic和原图裁剪为输出图的大小
    image_size, label_size, scale = lr_size, hr_size, scale
    label_padding = label_size // scale  # eg. 21 / 3 = 7 # 原来的代码

    num1 = label_padding * scale
    num2 = label_size * num_x
    num3 = label_size * num_y

    if len(image.shape) == 3:
        h = num2 + num1
        w = num3 + num1
        image1 = image[num1:h, num1:w, :]
    else:
        h = num2 + num1
        w = num3 + num1
        image1 = image[num1:h, num1:w]
    return image1


def array_image_save(array, image_path, color_im):
    """
    将数组转为图片格式并保存
    """
    image = Image.fromarray(array)
    if color_im == 1 and image.mode != 'RGB':
        image = image.convert('RGB')
    elif color_im == 0:
        image = image.convert('L')
    image.save(image_path)
    # print("Saved image: {}".format(image_path))


def color_image_save(src_image, im_cb_out, im_cr_out):
    """
    生成彩色图，并进行像素防溢出警告处理
    """
    back2src = cv2.merge([src_image * 255, im_cb_out * 255, im_cr_out * 255])
    pic = skco.ycbcr2rgb(back2src)
    # pic = skco.convert_colorspace(back2src, 'YCbCr', 'RGB')
    # 去掉超出0-255的像素
    pic = pic * 255
    pic *= (pic > 0)
    pic = pic * (pic <= 255) + 255 * (pic > 255)
    pic = pic.astype(np.uint8)

    dst_image = pic

    return dst_image


def preprocess_train(path, scale=3):
    """
    获取原始HR图
    将原始HR图像读取Y通道，并裁剪为缩放因子的倍数，且下采样得到需要的输入灰度图
    """
    im_rgb = skio.imread(path)              # 读取图片，格式是RGB
    im_rgb_mod = modcrop(im_rgb, scale)     # 将输入图裁剪为scale的倍数
    # 按照彩色图和灰度图进行操作
    if len(im_rgb_mod.shape) == 3:
        # 将rgb通道转为ycbcr通道，并分离出来
        im_ycbcr = skco.rgb2ycbcr(im_rgb_mod)   # 将RGB转为Ycbcr
        im_y_hr = im_ycbcr[:, :, 0] / 255.0     # 获取y通道
    else:
        im_y_hr = im_rgb_mod / 255.0

    # 对y通道，即灰度图进行上下采样处理
    im_y_down, im_y_bic = bicubic_down_up(im_y_hr, scale)

    # im_y_down = cv2.resize(im_y_hr, (int(im_y_hr.shape[1] / scale), int(im_y_hr.shape[0] / scale)),
    #                         interpolation=cv2.INTER_CUBIC)
    """
    输出变量说明：
    im_y_hr：                输入HR图；
    im_y_down：              模型网络的输入图；
    """
    return im_y_hr, im_y_down


def preprocess_test(path, scale=3):
    """
    获取原始HR图
    将原始HR图像读取Y通道，并裁剪为缩放因子的倍数，且下采样得到需要的输入灰度图
    """
    im_rgb = skio.imread(path)              # 读取图片，格式是RGB
    im_rgb_mod = modcrop(im_rgb, scale)     # 将输入图裁剪为scale的倍数
    # 按照彩色图和灰度图进行操作
    if len(im_rgb_mod.shape) == 3:
        # 将rgb通道转为ycbcr通道，并分离出来
        im_ycbcr = skco.rgb2ycbcr(im_rgb_mod)   # 将RGB转为Ycbcr
        im_y_hr = im_ycbcr[:, :, 0] / 255.0     # 获取y通道
        im_cb_hr = im_ycbcr[:, :, 1] / 255.0    # 获取cb通道
        im_cr_hr = im_ycbcr[:, :, 2] / 255.0    # 获取cr通道
        # 对cb,cr通道进行下采样和上采样
        im_cb_down, im_cb_bic = bicubic_down_up(im_cb_hr, scale)
        im_cr_down, im_cr_bic = bicubic_down_up(im_cr_hr, scale)
        # im_cb_down = cv2.resize(im_cb_hr, (int(im_cb_hr.shape[1] / scale), int(im_cb_hr.shape[0] / scale)),
        #                     interpolation=cv2.INTER_CUBIC)
        # im_cr_down = cv2.resize(im_cr_hr, (int(im_cr_hr.shape[1] / scale), int(im_cr_hr.shape[0] / scale)),
        #                     interpolation=cv2.INTER_CUBIC)
        # # 对已经下采用的cb,cr通道进行上采样，以便最后整合获得彩色图
        # im_cb_bic = cv2.resize(im_cb_down, (int(im_cb_hr.shape[1]), int(im_cb_hr.shape[0])),
        #                     interpolation=cv2.INTER_CUBIC)
        # im_cr_bic = cv2.resize(im_cr_down, (int(im_cr_hr.shape[1]), int(im_cr_hr.shape[0])),
        #                     interpolation=cv2.INTER_CUBIC)
        # 用于判断是否为彩色图
        color_im = 1
    else:
        im_y_hr = im_rgb_mod / 255.0
        im_cb_bic = 0
        im_cr_bic = 0
        color_im = 0        # 用于判断是否为彩色图

    # 对y通道，即灰度图进行上下采样处理
    im_y_down, im_y_bic = bicubic_down_up(im_y_hr, scale)

    # im_y_down = cv2.resize(im_y_hr, (int(im_y_hr.shape[1] / scale), int(im_y_hr.shape[0] / scale)),
    #                         interpolation=cv2.INTER_CUBIC)
    # im_y_bic = cv2.resize(im_y_down, (int(im_y_hr.shape[1]), int(im_y_hr.shape[0])),
    #                        interpolation=cv2.INTER_CUBIC)

    # cv2.imshow('im_y_down image', im_y_down)
    # cv2.imshow('im_y_bic image', im_y_bic)
    # cv2.waitKey(0)
    """
    输出变量说明：
    im_y_hr：                输入HR图；
    im_y_down：              模型网络的输入图；
    im_y_bic：               作为对比的bicubic图；
    im_cb_bic和im_cr_bic：    最后用于合成彩色图；
    color_im：               用于判断是否彩色图，方便后续操作
    """
    return im_y_hr, im_y_down, im_y_bic, im_cb_bic, im_cr_bic, color_im


def bicubic_down_up(im_hr, scale=3):
    im_hr_origin = Image.fromarray(im_hr)
    im_bic_down = im_hr_origin.resize((int(im_hr.shape[1] / scale), int(im_hr.shape[0] / scale)), Image.BICUBIC)
    im_bic_up = im_bic_down.resize((int(im_hr.shape[1]), int(im_hr.shape[0])), Image.BICUBIC)
    # 转为array格式
    (im_bic_down_width, im_bic_down_height) = im_bic_down.size
    im_bic_down_array = np.array(list(im_bic_down.getdata())).astype(np.float).reshape((im_bic_down_height, im_bic_down_width))
    (im_bic_up_width, im_bic_up_height) = im_bic_up.size
    im_bic_up_array = np.array(list(im_bic_up.getdata())).astype(np.float).reshape((im_bic_up_height, im_bic_up_width))
    return im_bic_down_array, im_bic_up_array


def train_input_setup(config):
    """
    读取训练集图片，并制作sub-images(即，patches图像块)
    并保存为.h5文件格式
    """
    # 初始化
    sess = config.sess
    lr_size, hr_size, scale, stride = config.LR_size, config.HR_size, config.scale, config.stride
    sub_lr_sequence, sub_hr_sequence = [], []
    padding = abs(hr_size - lr_size) // 2  # eg. for 3x: (21 - 11) / 2 = 5 表示整数除法,返回不大于结果的一个最大的整数
    label_padding = hr_size // scale  # eg. for 3x: 21 / 3 = 7

    # 加载数据集的路径
    data = prepare_data(sess, dataset=config.TrainData_dir)

    # 遍历训练集图片，制作训练数据集
    for i in range(len(data)):
        input_hr_, input_lr_ = preprocess_train(data[i], scale)

        if len(input_lr_.shape) == 3:
            h, w, _ = input_lr_.shape
        else:
            h, w = input_lr_.shape

        for x in range(0, h - lr_size + 1, stride):
            for y in range(0, w - lr_size + 1, stride):

                x_loc = scale * (x + floor((lr_size - 1) / 2)) - floor((hr_size + scale) / 2 - 1)  # 3x+15-11
                y_loc = scale * (y + floor((lr_size - 1) / 2)) - floor((hr_size + scale) / 2 - 1)

                sub_lr = input_lr_[x: lr_size + x, y: lr_size + y]
                sub_hr = input_hr_[x_loc: hr_size + x_loc, y_loc: hr_size + y_loc]

        # if 2 * hr_size / scale - 1 / 3 > lr_size + padding - 1:
        #     bu_ding = int(2 * hr_size / scale - 1)
        # else:
        #     bu_ding = int(lr_size + padding - 1)
        #
        # for x in range(0, h - bu_ding, stride):
        #     for y in range(0, w - bu_ding, stride):
        #         sub_lr = input_lr_[x + padding: x + padding + lr_size, y + padding: y + padding + lr_size]
        #
        #         x_loc, y_loc = x + label_padding, y + label_padding
        #         sub_hr = input_hr_[x_loc * scale: x_loc * scale + hr_size, y_loc * scale: y_loc * scale + hr_size]

                sub_lr = sub_lr.reshape([lr_size, lr_size, 1])
                sub_hr = sub_hr.reshape([hr_size, hr_size, 1])

                sub_lr_sequence.append(sub_lr)
                sub_hr_sequence.append(sub_hr)

    arr_lr_data = np.asarray(sub_lr_sequence)
    arr_hr_data = np.asarray(sub_hr_sequence)

    make_data(sess, config.checkpoint_dir, arr_lr_data, arr_hr_data)


def test_input_setup(config, i, data_img):
    """
    Read image files, make their sub-images, and save them as a h5 file format.
    """
    sess = config.sess
    image_size, label_size, stride, scale = config.LR_size, config.HR_size, config.stride, config.scale

    data = data_img

    sub_input_sequence, sub_label_sequence = [], []
    padding = abs(image_size - label_size) // 2     # eg. (21 - 11) / 2 = 5 # 原来的代码
    label_padding = label_size // scale             # eg. 21 / 3 = 7 # 原来的代码

    pic_index = i  # Index of image based on lexicographic order in data folder
    im_y_hr, im_y_down, im_y_bic, im_cb_bic, im_cr_bic, color_im = preprocess_test(data[pic_index], config.scale)
    input_, label_ = im_y_down, im_y_hr

    if len(input_.shape) == 3:
        h, w, _ = input_.shape
    else:
        h, w = input_.shape

    nx, ny = 0, 0

    for x in range(0, h - image_size + 1, stride):
        nx += 1
        ny = 0
        for y in range(0, w - image_size + 1, stride):
            ny += 1
            x_loc = scale * (x + floor((image_size - 1) / 2)) - floor((label_size + scale) / 2 - 1)
            y_loc = scale * (y + floor((image_size - 1) / 2)) - floor((label_size + scale) / 2 - 1)

            sub_input = input_[x: image_size + x, y: image_size + y]
            sub_label = label_[x_loc: label_size + x_loc, y_loc: label_size + y_loc]

    # if 2*label_size / scale - 1/3 > image_size + padding - 1:  # 防止超出边界
    #     bu_ding = int(2*label_size / scale - 1)
    # else:
    #     bu_ding = int(image_size + padding - 1)
    #
    # for x in range(0, h - bu_ding, stride):
    #     nx += 1
    #     ny = 0
    #     for y in range(0, w - bu_ding, stride):  # (-image_size-padding)是为了不让input_出界
    #         ny += 1
    #         sub_input = input_[x + padding: x + padding + image_size, y + padding: y + padding + image_size]
    #         x_loc, y_loc = x + label_padding, y + label_padding
    #         sub_label = label_[x_loc * scale: x_loc * scale + label_size, y_loc * scale: y_loc * scale + label_size]

            sub_input = sub_input.reshape([image_size, image_size, 1])
            sub_label = sub_label.reshape([label_size, label_size, 1])

            sub_input_sequence.append(sub_input)
            sub_label_sequence.append(sub_label)

    arrdata = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)
    # 制作h5格式的数据文件
    make_data(sess, config.checkpoint_dir, arrdata, arrlabel)

    # 把各类通道裁剪成与最终的输出图尺寸相同大小
    if color_im == 1:
        im_cb_bic_same = modcrop_same(nx, ny, im_cb_bic, image_size, label_size, scale)
        im_cr_bic_same = modcrop_same(nx, ny, im_cr_bic, image_size, label_size, scale)
    else:
        im_cb_bic_same = 0
        im_cr_bic_same = 0
    im_y_hr_same = modcrop_same(nx, ny, im_y_hr, image_size, label_size, scale)
    im_y_bic_same = modcrop_same(nx, ny, im_y_bic, image_size, label_size, scale)

    return nx, ny, im_y_hr_same, im_y_bic_same, im_cb_bic_same, im_cr_bic_same, color_im
