import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

# 将点云与红外数据配准
filepath = r'D:\AntiUAV\dataset_infrared\SPAD\SPAD_np_xyz'
filename = r'.npy'


# 1、提取点云数据与图像数据中的关键点
# 2、
def read_pc(filepath):
    for filename in os.listdir(filepath):
        pc = np.load(filepath + filename)  # 单个是N*3的


    return pc
