# -*- coding: utf-8 -*-
# @Time : 20-6-4 上午9:12
# @Author : zhuying
# @Company : Minivision
# @File : default_config.py
# @Software : PyCharm
# --*-- coding: utf-8 --*--
"""
default config for training
"""

import torch
from datetime import datetime
from easydict import EasyDict
from src.utility import make_if_not_exist, get_width_height, get_kernel


def get_default_config():
    conf = EasyDict() # conf为EasyDict类型，方便我们调用其属性值

    # ----------------------training---------------
    conf.lr = 1e-1 # 指定学习率
    # [9, 13, 15]
    conf.milestones = [10, 15, 22]  # 定义学习率变化的epoch
    conf.gamma = 0.1 # 学习率变化因子
    conf.epochs = 25 # 定义训练epoch的数目
    conf.momentum = 0.9 # 动量系数
    conf.batch_size = 1024

    # model
    conf.num_classes = 3 # 类别数目：1-真脸 0-电子屏幕 2-其他材质的攻击
    conf.input_channel = 3
    conf.embedding_size = 128

    # dataset
    conf.train_root_path = './datasets/rgb_image' # 数据集所在路径

    # save file path
    conf.snapshot_dir_path = './saved_logs/snapshot'

    # log path
    conf.log_path = './saved_logs/jobs'
    # 更新tensorboard的周期
    conf.board_loss_every = 10 
    # 保存模型的周期
    conf.save_every = 30 

    return conf


def update_config(args, conf):
    conf.devices = args.devices
    conf.patch_info = args.patch_info
    w_input, h_input = get_width_height(args.patch_info) # 分割patch_info，提取宽度和高度信息
    conf.input_size = [h_input, w_input]
    conf.kernel_size = get_kernel(h_input, w_input) # 返回用于傅里叶分解的核的大小
    conf.device = "cuda:{}".format(conf.devices[0]) if torch.cuda.is_available() else "cpu"

    # resize fourier image size
    conf.ft_height = 2*conf.kernel_size[0]
    conf.ft_width = 2*conf.kernel_size[1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    job_name = 'Anti_Spoofing_{}'.format(args.patch_info)
    log_path = '{}/{}/{} '.format(conf.log_path, job_name, current_time) 
    snapshot_dir = '{}/{}'.format(conf.snapshot_dir_path, job_name) # 暂存模型的路径

    make_if_not_exist(snapshot_dir) # 如果不存在，则创建该文件夹
    make_if_not_exist(log_path)

    conf.model_path = snapshot_dir
    conf.log_path = log_path
    conf.job_name = job_name
    return conf
