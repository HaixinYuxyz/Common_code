import os
import datetime

# 以时间建立保存checkpoint的文件夹
dt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
save_folder = os.path.join('./output', dt)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 使用torchsummary来打印网络结构，net为实例化后的网络，input_channels为输入channelinput_size，input_size为输入尺寸
from torchsummary import summary

summary(net, (input_channels, input_size, input_size))

# 生成一个txt文件，保存网络的结构和参数
import sys

f = open(os.path.join(save_folder, 'arch.txt'), 'w')
sys.stdout = f
summary(net, (input_channels, input_size, input_size))
print(net)
sys.stdout = sys.__stdout__
f.close()

# 保存训练时候的Log记录,同时在文件和控制台进行输出
import logging
import time


def log_creater(log_file_dir):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    # set two handlers
    fileHandler = logging.FileHandler(os.path.join(log_file_dir, 'log.log'), mode='w')
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    return logger
