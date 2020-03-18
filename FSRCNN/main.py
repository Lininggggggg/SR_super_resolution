#!/Anaconda3/envs/tf110-gpu-py36/python python
# coding:utf-8

from model import FSRCNN
import utils
import numpy as np
import tensorflow as tf
import pprint
import os
import warnings

warnings.filterwarnings("ignore")

flags = tf.app.flags
# 参数
flags.DEFINE_boolean("is_train", True, "训练还是测试[True]")
# flags.DEFINE_boolean("is_train", False, "训练还是测试[True]")  # 注意：换测试数据集时，model和utils有两处'Set5'要改
flags.DEFINE_integer("epoch", 10, "训练迭代次数[100]")
flags.DEFINE_integer("batch_size", 128, "单次训练的batch类型[128]")
flags.DEFINE_integer("scale", 3, "缩放因子2、3或4[3]")
flags.DEFINE_integer("stride", 4, "预处理层的步长[4]")

# 论文里网络层的三个敏感参数
flags.DEFINE_integer("var_d", 56, "LR特征尺寸,敏感变量d[56]")
flags.DEFINE_integer("var_s", 12, "缩小滤镜的数量,敏感变量s[12]")
flags.DEFINE_integer("var_m", 4, "映射深度,敏感变量m[4]")

# ?学习率的设置
flags.DEFINE_float("learning_rate", 0.001, "初始学习率（结合SGD）[0.001]")
flags.DEFINE_float("momentum", 0.9, "SGD参数设置[0.9]")

# 数据提取和储存的所在位置
flags.DEFINE_string("TrainData_dir", "TrainImage", "训练数据存放位置")
# flags.DEFINE_string("TrainData_dir", "expand_image", "训练数据存放位置")
flags.DEFINE_string("TestData_dir", "TestImage", "测试数据存放位置")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "数据提取位置")
flags.DEFINE_string("output_dir", "test_result", "图片测试结果存放位置")

FLAGS = flags.FLAGS
pp = pprint.PrettyPrinter()


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    with tf.Session() as sess:
        fsrcnn = FSRCNN(sess, config=FLAGS)
        fsrcnn.run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    tf.app.run()
