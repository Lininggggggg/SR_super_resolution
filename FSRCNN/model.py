#!/Anaconda3/envs/tf110-gpu-py36/python python
# coding:utf-8

from utils import (
    modcrop_same,
    read_data,
    train_input_setup,
    test_input_setup,
    merge,
    array_image_save,
    prepare_data,
    color_image_save
)
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pdb
import glob
import scipy.misc
import cv2
from skimage import io as skio
from skimage import color as skco
from skimage import img_as_ubyte
from skimage import color
import warnings

warnings.filterwarnings("ignore")


class FSRCNN(object):
    def __init__(self, sess, config):
        self.sess = sess
        self.c_dim = 1  # 图像channels维度

        self.is_train = config.is_train
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.scale = config.scale
        self.stride = config.stride

        self.var_d = config.var_d
        self.var_s = config.var_s
        self.var_m = config.var_m

        self.learning_rate = config.learning_rate
        self.momentum = config.momentum

        self.TrainData_dir = config.TrainData_dir
        self.TestData_dir = config.TestData_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.output_dir = config.output_dir

        # LR和SR的patch缩放因子的比例大小scale_factors，*2，*3，*4
        # scale_factors = [[10, 19], [7, 19], [6, 21]]  # 论文应该是这样
        # 实际操作上应该是下面这样，+5-1=4是由于卷积层第一层padding为0，而其他层有padding
        # 但论文里又说padding（全）为0，所以此处存疑
        # scale_factors = [[10+5-1, 19], [7+5-1, 19], [6+5-1, 21]]  # 自己理解的应该是这样
        # 但论文里说的19,19,21是caffe的反卷积层的原因，可能此处tensorflow没这个问题，所以为20,21,24
        scale_factors = [[10+5-1, 20], [7+5-1, 21], [6+5-1, 24]]
        # scale_factors = [[10+5-1, 19], [7+5-1, 19], [6+5-1, 21]]  # 测试之后发现不行

        self.LR_size, self.HR_size = scale_factors[self.scale - 2]  # 缩放scale对应的图像尺寸

        if not self.is_train:
            self.stride = [10, 7, 6][self.scale - 2]
            # self.stride = [14, 11, 10][self.scale - 2]  # 此处确实不是11，因为stride在测试时候是设置为反卷积输入尺寸

        self.build_model()

    def build_model(self):
        # 卷积网络参数初始化
        self.LRs = tf.placeholder(tf.float32, [None, self.LR_size, self.LR_size, self.c_dim], name='LRs')
        self.HRs = tf.placeholder(tf.float32, [None, self.HR_size, self.HR_size, self.c_dim], name='HRs')
        # Batch size differs in training vs testing
        self.batch = tf.placeholder(tf.int32, shape=[], name='batch')

        # 构建CNN的每层的权重w和偏差b
        d, s, m = self.var_d, self.var_s, self.var_m

        expand_weight = 'w{}'.format(m+3)
        expand_bias = 'b{}'.format(m+3)
        deconv_weight = 'w{}'.format(m+4)
        deconv_bias = 'b{}'.format(m+4)

        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, 1, d], stddev=0.0378, dtype=tf.float32), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, d, s], stddev=0.3536, dtype=tf.float32), name='w2'),
            expand_weight: tf.Variable(tf.random_normal([1, 1, s, d], stddev=0.189, dtype=tf.float32), name=expand_weight),
            deconv_weight: tf.Variable(tf.random_normal([9, 9, 1, d], stddev=0.001, dtype=tf.float32), name=deconv_weight)
        }  # 反卷积层的权重deconv_weight本来应该是[9, 9, d, 1]的，但tensorflow对卷积和反卷积的卷积核输入输出通道位置刚好设置相反而已，不是算法层面上的问题

        self.biases = {
            'b1': tf.Variable(tf.zeros([d]), name='b1'),
            'b2': tf.Variable(tf.zeros([s]), name='b2'),
            expand_bias: tf.Variable(tf.zeros([d]), name=expand_bias),
            deconv_bias: tf.Variable(tf.zeros([1]), name=deconv_bias)
        }

        for i in range(3, m+3):
            shrink_weight = 'w{}'.format(i)
            shrink_bias = 'b{}'.format(i)
            self.weights[shrink_weight] = tf.Variable(tf.random_normal([3, 3, s, s], stddev=0.1179, dtype=tf.float32), name=shrink_weight)
            self.biases[shrink_bias] = tf.Variable(tf.zeros([s]), name=shrink_bias)

        # 模型输出，用来求损失函数和测试
        self.SRpred = self.model()

        # 损失函数
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.HRs-self.SRpred), reduction_indices=0))
        self.saver = tf.train.Saver()

    def run(self):
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss)

        tf.initialize_all_variables().run()

        if self.load(self.checkpoint_dir):
            print("[*] Load SUCCESS")
        else:
            print("[!] Load failed...")

        if self.is_train:
            self.run_train()
        else:
            self.run_test()

    def run_train(self):
        # 读取训练集图片，并制作训练用的h5文件
        start_time = time.time()
        print("Beginning training setup...")
        train_input_setup(self)
        print("Training setup took {} seconds.".format(time.time()-start_time))

        # 读取h5文件
        data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "train.h5")
        train_lr_data, train_hr_data = read_data(data_dir)
        print("Total setup time took {} seconds.".format(time.time()-start_time))

        # 开始训练
        print("Training...")
        start_time = time.time()
        start_average, end_average, counter = 0, 0, 0

        for ep in range(self.epoch):
            # Run by batch images
            batch_idxs = len(train_lr_data) // self.batch_size
            batch_average = 0
            for idx in range(0, batch_idxs):
                batch_images = train_lr_data[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_labels = train_hr_data[idx * self.batch_size: (idx + 1) * self.batch_size]

                counter += 1
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.LRs: batch_images, self.HRs: batch_labels, self.batch: self.batch_size})
                batch_average += err

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                        % ((ep+1), counter, time.time() - start_time, err))

                # Save every 500 steps
                if counter % 500 == 0:
                    self.save(self.checkpoint_dir, counter)

            # 下面语句是储存前20%和后20%的训练loss，以便最后说明提升多少
            batch_average = float(batch_average) / batch_idxs
            if ep < (self.epoch * 0.2):
                start_average += batch_average
            elif ep >= (self.epoch * 0.8):
                end_average += batch_average

        # Compare loss of the first 20% and the last 20% epochs
        start_average = float(start_average) / (self.epoch * 0.2)
        end_average = float(end_average) / (self.epoch * 0.2)
        print("Start Average: [%.6f], End Average: [%.6f], Improved: [%.2f%%]" \
            % (start_average, end_average, 100 - (100*end_average/start_average)))

    def run_test(self):
        # 读取测试数据集图片
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), self.TestData_dir)), "Set5")
        data = sorted(glob.glob(os.path.join(data_dir, "*.bmp")))
        data_img = prepare_data(self, dataset=self.TestData_dir)

        # 生成列表记录每次的psnr，准备用于psnr的平均值求取
        all_psnr_bic, all_psnr_fsr = [], []

        # 遍历路径下的图片，并进行测试
        for i in range(len(data)):
            # 制作测试用的h5文件，并读取测试图和各类输入图
            nx, ny, im_y_hr_same, im_y_bic_same, im_cb_bic_same, im_cr_bic_same, color_im = test_input_setup(self, i, data_img)
            data_dir = os.path.join('./{}'.format(self.checkpoint_dir), "test.h5")
            test_data, test_label = read_data(data_dir)

            # 开始测试
            print("Testing...")
            start_time = time.time()
            result = self.SRpred.eval({self.LRs: test_data, self.HRs: test_label, self.batch: nx * ny})
            print("Took %.3f seconds" % (time.time() - start_time))

            # 对输出结果图片转换size储存形式
            result = merge(result, [nx, ny])
            result = result.squeeze()                   # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉

            # 对输出结果、bicubic、原图去除边界，提取干净数据
            height, weight = result.shape[0], result.shape[1]
            result = result[2:height-2, 2:weight-2]
            # result = result[1:height-3, 1:weight-3]   # matlab官方在*3上的做法，但此处效果不对，应该是卷积核的区别
            im_y_hr_same = im_y_hr_same[2:height-2, 2:weight-2]
            im_y_bic_same = im_y_bic_same[2:height-2, 2:weight-2]
            # 对输出图片进行批量格式化命名
            dst00 = os.path.join(os.path.abspath(self.output_dir), '00' + format(str(i), '0>3s') + '_BIC.bmp')
            dst10 = os.path.join(os.path.abspath(self.output_dir), '00' + format(str(i), '0>3s') + '_FSR.bmp')
            dst20 = os.path.join(os.path.abspath(self.output_dir), '00' + format(str(i), '0>3s') + '_HR.bmp')
            # 保存各类测试结果图
            array_image_save(im_y_bic_same * 255, dst00, color_im)    # 保存bicubic输出灰度图
            array_image_save(im_y_hr_same * 255, dst20, color_im)     # 保存hr输出灰度图
            array_image_save(result * 255, dst10, color_im)           # 保存fsrcnn输出灰度图

            # 计算每次bicubic和fsrcnn测试结果的psnr
            psnr_bic = self.psnr_computer(im_y_hr_same, im_y_bic_same)
            psnr_fsr = self.psnr_computer(im_y_hr_same, result)
            all_psnr_bic.append(psnr_bic)
            all_psnr_fsr.append(psnr_fsr)
            print('PSNR for Bicubic:', psnr_bic, 'dB\n' 'PSNR for FSRCNN:', psnr_fsr, 'dB\n')

            if color_im == 1:
                # 对cb,cr通道去除边界，提取干净数据
                im_cb_bic_same = im_cb_bic_same[2:height-2, 2:weight-2]
                im_cr_bic_same = im_cr_bic_same[2:height-2, 2:weight-2]
                # 将y，cb，cr通道转为rgb通道，并对溢出值进行处理
                bic_color = color_image_save(im_y_bic_same, im_cb_bic_same, im_cr_bic_same)
                fsr_color = color_image_save(result, im_cb_bic_same, im_cr_bic_same)
                hr_color = color_image_save(im_y_hr_same, im_cb_bic_same, im_cr_bic_same)
                # 对输出图片进行批量格式化命名
                dst01 = os.path.join(os.path.abspath(self.output_dir), '00' + format(str(i), '0>3s') + '_BIC_color.bmp')
                dst11 = os.path.join(os.path.abspath(self.output_dir), '00' + format(str(i), '0>3s') + '_FSR_color.bmp')
                dst21 = os.path.join(os.path.abspath(self.output_dir), '00' + format(str(i), '0>3s') + '_HR_color.bmp')
                # 保存各类测试结果图
                skio.imsave(dst01, bic_color)              # 保存bicubic输出彩色图
                skio.imsave(dst11, fsr_color)              # 保存hr输出彩色图
                skio.imsave(dst21, hr_color)               # 保存fsrcnn输出彩色图

        # 计算总的bicubic和fsrcnn测试结果的平均psnr
        mean_psnr_bic, mean_psnr_fsr = np.mean(all_psnr_bic), np.mean(all_psnr_fsr)
        print('Mean PSNR for Bicubic:', mean_psnr_bic, 'dB\n' 'Mean PSNR for FSRCNN:', mean_psnr_fsr, 'dB\n')

    def model(self):
        """
        网络模型结构说明
        """
        # Feature Extraction
        conv_feature = self.prelu(tf.nn.conv2d(self.LRs, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1'], 1)

        # Shrinking
        conv_shrink = self.prelu(tf.nn.conv2d(conv_feature, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2'], 2)

        # Mapping (# mapping layers = m)
        prev_layer, m = conv_shrink, self.var_m
        for i in range(3, m + 3):
            weights, biases = self.weights['w{}'.format(i)], self.biases['b{}'.format(i)]
            prev_layer = self.prelu(tf.nn.conv2d(prev_layer, weights, strides=[1, 1, 1, 1], padding='SAME') + biases, i)

        # Expanding
        expand_weights, expand_biases = self.weights['w{}'.format(m + 3)], self.biases['b{}'.format(m + 3)]
        conv_expand = self.prelu(tf.nn.conv2d(prev_layer, expand_weights, strides=[1, 1, 1, 1], padding='VALID') + expand_biases, 7)

        # Deconvolution
        deconv_output = [self.batch, self.HR_size, self.HR_size, self.c_dim]
        deconv_stride = [1,  self.scale, self.scale, 1]
        deconv_weights, deconv_biases = self.weights['w{}'.format(m + 4)], self.biases['b{}'.format(m + 4)]
        conv_deconv = tf.nn.conv2d_transpose(conv_expand, deconv_weights, output_shape=deconv_output, strides=deconv_stride, padding='SAME') + deconv_biases

        return conv_deconv

    def prelu(self, _x, i):
        """
        PreLU tensorflow implementation
        """
        # allsa = _x.get_shape()[-1]  # _x.get_shape()[-1]就是倒数一个维数，如（？,7,7,56）中的56，即输出维度
        alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5  # (_x - abs(_x)) * 0.5相当于求输出值和0值之间的最小值，alphas是训练的系数

        return pos + neg

    def save(self, checkpoint_dir, step):
        model_name = "FSRCNN.model"
        model_dir = "%s_%s" % ("fsrcnn", self.HR_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        """
        加载数据文件h5
        """
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("fsrcnn", self.HR_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def psnr_computer(self, im1, im2):  # 计算结果的PSNR
        """
        计算psnr
        """
        diff = im1*255 - im2*255
        get_rmse = np.sqrt(np.mean(diff**2))
        get_psnr = 20 * np.log10(255 / get_rmse)
        return get_psnr











