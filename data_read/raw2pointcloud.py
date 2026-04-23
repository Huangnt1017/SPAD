import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# import open3d as o3d


data_size = 2  # 每个uint16数据占据2个字节
# 设置时间截取阈值
timeThreshold = 190

# 选择帧数进行统计拟合
startFrame = 1
endFrame = 1000
# 重塑数据为64*64的数组
numPixels = 64 * 64
numFrames = 2000

offset_byte = startFrame * numPixels * data_size
toread_byte = (endFrame - startFrame) * numPixels * data_size


def raw2n3(filename):
    # 读取数据
    with open(filename, 'rb') as f:
        f.seek(offset_byte)
        binary_data = f.read(toread_byte)
        num_data_read = len(binary_data) // data_size
        import struct
        uint16_list = struct.unpack(f'{num_data_read}H', binary_data)
        raw_data = np.array(uint16_list, dtype=np.uint16)
    # print(raw_data.shape)
    data_gpu = raw_data.reshape((endFrame - startFrame, 64, 64))
    # 读取的时候是channel,height,width，注意channel与matlab的相反的
    # print(data_gpu.shape)
    # pc = np.zeros(shape=(0, 3), dtype=int)
    # 将单光子的(x,y,frame)的格式转为点云格式(x,y,z)，仅保留小于阈值的
    condition = (data_gpu < timeThreshold) & (data_gpu > 0)
    # print(condition)
    indices = np.argwhere(condition)
    # print(indices[:, 1:])
    pc = np.hstack((indices[:, 1:] + 1, data_gpu[condition].reshape((-1, 1))))
    # print(pc[:55,:])
    return pc, data_gpu  # N*3, xyz


def n3_filter(point_cloud):  # input = N*3
    # 统计处理（滤波）
    # print(point_cloud.shape)
    # 阈值处理，回波光子数量小于一定比例的帧数时直接剔除
    # 将重复的点云只保留一个坐标（已在unique操作中完成）
    unique_rs_pc, counts = np.unique(point_cloud, axis=0, return_counts=True)
    # print(unique_rs_pc[:50,:])
    # print(counts)
    # 使用布尔索引找到满足条件的唯一点云和它们的计数
    # 这里的条件是计数大于等于 active_point
    mask = counts >= active_point
    filtered_points = unique_rs_pc[mask]
    filtered_counts = counts[mask]
    indx = np.column_stack((filtered_points, filtered_counts))
    # print(filtered_points.shape)
    # print(counts.shape)
    # print(np.sum(filtered_counts))
    # print(indx[:100,:])
    return indx  # output = N*4


def plot_pc(pc):
    # a = (pc[:, 3] - np.min(pc[:, 3]))/(np.max(pc[:, 3]) - np.min(pc[:, 3]))
    # print(a.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc = ax.scatter(pc[:, 1], pc[:, 2],  np.abs(pc[:,0]-65),c=pc[:, 3], s=2, cmap='viridis')
    # np.abs(pc[:, 0]-65),
    # 设置坐标轴
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    # 设置观察视角
    ax.view_init(elev=30, azim=-45)
    # ax.dist = 1000
    # 设置颜色条，用(2是距离，3是强度)作为颜色（即点的强度）
    cbar = fig.colorbar(pc, location='left', shrink=0.6, fraction=0.05)
    cbar.set_label('Length of point')
    return plt.show(block=True)


def read_pc(file):
    return np.loadtxt(file, delimiter=',')


# 最远点采样
def pc_down_sample(pc, num):
    if pc.shape[0] < num:
        raise ValueError("原始数组的行数少于目标采样点数，无法进行采样。")
    N, C = pc.shape
    # 初始化一个数组用于存储num个采样点的索引的位置，大小为num
    pc_ds = np.zeros((num, pc.shape[1]))
    # 用于记录所有点到一个点的距离
    distance = np.ones(N) * 1e10
    # 随机选择一个起始点作为最远点
    farthest = np.random.randint(0, N)
    # 开始迭代，直到达到目标采样点数
    for i in range(num):
        pc_ds[i] = pc[farthest]  # 当前点设为最远
        centroid = pc[farthest, :3]  # 计算距离
        dist = np.linalg.norm(pc[:, :3] - centroid, axis=1)
        np.minimum(distance, dist, out=distance)  # 只保留最小
        farthest = np.argmax(distance)  # 获得索引
    # print(pc_ds.shape)
    return pc_ds


def save_pc(pc):
    return np.savetxt('F:\\20241205_data\\spad\\pc\\1.txt', pc, fmt='%d', delimiter=',')


# 文件路径
# filename1 = r'E:\\essay\\硕士\\研一\\SPAD数据\\20250430dataset\\2025-04-30-pc\\2025-04-30_18-47-35_Delay-0_Width-200-1-3.txt'
# f = r'E:\\essay\\硕士\\研一\\SPAD数据\\20250430dataset\\2025-04-30-pc\\C\\2025-04-30_18-50-39_Delay-0_Width-200-10-12.txt'
active_point = int((endFrame - startFrame) * 0.03)
np.set_printoptions(threshold=np.inf)
if __name__ == '__main__':
    # 可视化txt文件
    file = r'D:\\PYproject\\SPAD\\HMC\\2025-04-30_18-51-28_Delay-0_Width-200-11-13.txt'
    pc = read_pc(file)
    print(file)
    # pc = read_pc('E:/PyCharm Community Edition 2021.2.3/read_spad/HMC/A1.txt')
    intensities = pc[:,3]
    print(np.percentile(intensities, 90))
    mask = pc[:, 3] >= np.percentile(intensities, 90)
    print(pc[mask].shape)
    # print(pc.shape)
    plot_pc(pc[mask])
    # 可视化raw文件
    # start = time.time()
    # rs_pc,data = raw2n3(f)
    # keep_indx = n3_filter(rs_pc)
    # print('keep_indx shape:', keep_indx.shape)
# #     # data[data>190] = 0
#     data = data[:,40,30]
#     bins = 200
#     hist, edges = np.histogram(data, bins=bins, density=True)  # 密度直方图
#
# # 绘制直方图
#     plt.bar(edges[:-1], hist, width=edges[1] - edges[0], edgecolor='black')
#     plt.xlabel('Value')
#     plt.ylabel('Density')
#
#     plt.show()

    # keep_indx1 = pc_down_sample(keep_indx, 5000)
    # # print('keep_indx1 shape:', keep_indx1.shape)
    # end = time.time()
    # print(end - start, 's')
    # plot_pc(keep_indx)
    # plot_pc(keep_indx)
    # save_pc(keep_indx)  # 保存的就是降采样之后的


