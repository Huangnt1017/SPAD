import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import open3d as o3d

# 文件路径
filename1 = r'D:\AntiUAV\dataset_infrared\SPAD\2024101112194700.Raw'
data_size = 2  # 每个uint16数据占据2个字节
# 设置时间截取阈值
timeThreshold = 200

# 选择帧数进行统计拟合
startFrame = 500
endFrame = 1000
# 重塑数据为64*64的数组
numPixels = 64 * 64
numFrames = 20000
# print('data_gpu', data_gpu)
# 假设回波光子数量超过timeThreshold*0.0
active_point = int((endFrame - startFrame) * 0.05)
# print(active_point)
# 选择需要读取的数据，偏移和读取的字节数
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
    return pc  # N*3, xyz


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
    # 归一化强度
    # 避免出现0
    max_s = max(pc[:,3])
    min_s = min(pc[:,3])
    epsilon = 1e-8
    normalized_strength = (pc[:,3] - min_s)/(max_s-min_s+epsilon)
    alpha_normalized = normalized_strength
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc = ax.scatter(pc[:, 1], pc[:, 2], pc[:, 0],
                    c=normalized_strength,
                    s=5,
                    cmap='jet')
    # 设置坐标轴
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    # 设置观察视角
    ax.view_init(elev=30, azim=-45)
    # 设置颜色条，用(2是距离，3是强度)作为颜色（即点的强度）
    cbar = fig.colorbar(pc, location='left', shrink=0.6, fraction=0.05)
    cbar.set_label('Length of point')
    return plt.show(block=True)


def save_pc(pc):
    return np.save('D:\\AntiUAV\\dataset_infrared\\SPAD\\SPAD_np_xyz\\X1.npy', pc)


# 转ply保存
def pc2ply(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return o3d.io.write_point_cloud("out.ply", pcd)


def o3d_draw(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    return o3d.visualization.draw_geometries([pcd])



np.set_printoptions(threshold=np.inf)
start = time.time()
rs_pc = raw2n3(filename1)
keep_indx = n3_filter(rs_pc)
# print(keep_indx[400:,:])
# save_pc(keep_indx[:,:3])
end = time.time()
print(end - start, 's')
# o3d_draw(keep_indx[:,:3])
plot_pc(keep_indx)
