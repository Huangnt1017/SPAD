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
endFrame = 480 * 10
# 重塑数据为64*64的数组
numPixels = 64 * 64

offset_byte = startFrame * numPixels * data_size
toread_byte = (endFrame - startFrame) * numPixels * data_size
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

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


def n3_filter(point_cloud, active_point=0):  # input = N*3
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


from matplotlib.colors import LinearSegmentedColormap, Normalize
def plot_pc(pc, mode):
    # npy/npz 里常是 uint16，做减法前先转有符号，避免下溢导致坐标异常
    pc = np.asarray(pc)
    xyz = pc[:, :3].astype(np.int32, copy=False)
    intensity = pc[:, 3].astype(np.int32, copy=False)

    # a = (pc[:, 3] - np.min(pc[:, 3]))/(np.max(pc[:, 3]) - np.min(pc[:, 3]))
    # print(a.shape)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    # ax.grid(False)
    # ax.set_axis_off()
    # 设置坐标轴
    # 移除数字刻度
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X')
    # # 轴上标签
    if mode == 'all':
        # 定义浅蓝和深红的RGB值（0-255转换为0-1）
        light_blue = (113/255, 178/255, 255/255)   # 浅蓝
        dark_red   = (255/255, 0/255, 0/255)       # 深红

        # 创建自定义colormap，从浅蓝到深红
        cmap_custom = LinearSegmentedColormap.from_list('lightblue_to_darkred', [light_blue, dark_red])

        # 设置归一化，将强度1映射到浅蓝，强度5映射到深红
        norm = Normalize(vmin=1, vmax=750)
        i = intensity.astype(np.float32) ** 0.7  # 非线性化的强度
        denom = i.max() - i.min()
        if denom > 0:
            al = (i - i.min()) / denom  # 对每个点设置透明度，用在alpha
        else:
            al = 0.6
    # 去除网格
        pc = ax.scatter(xyz[:, 1], xyz[:, 2], np.abs(xyz[:, 0] - 65), c=intensity,
                        s=2, cmap=cmap_custom, alpha=al)
        ax.set_xlim(0, 64)
        ax.set_ylim(0, 190)
        ax.set_zlim(0, 64)
        ax.view_init(elev=10, azim=-7)
    elif mode == 'ds':
        # 定义浅蓝和深红的RGB值（0-255转换为0-1）
        light_blue = (113/255, 178/255, 255/255)   # 浅蓝
        dark_red   = (255/255, 0/255, 0/255)       # 深红

        # 创建自定义colormap，从浅蓝到深红
        cmap_custom = LinearSegmentedColormap.from_list('lightblue_to_darkred', [light_blue, dark_red])

        # 设置归一化，将强度1映射到浅蓝，强度5映射到深红
        norm = Normalize(vmin=1, vmax=1550)
        pc = ax.scatter(xyz[:, 1], xyz[:, 2], np.abs(xyz[:, 0] - 65), c=intensity,
                        s=2, cmap=cmap_custom, alpha=0.5)
        ax.set_xlim(0, 64)
        # ax.set_ylim(40, 70)
        ax.set_zlim(0, 64)
        ax.view_init(elev=10, azim=-135)
        # ax.view_init(elev=45, azim=-43)
    elif mode == 'n3':
        pc = ax.scatter(xyz[:, 1], xyz[:, 2], np.abs(xyz[:, 0] - 65), color=(72/255,16/255,96/255),
                        s=5, alpha=0.7)
        ax.view_init(elev=2, azim=-89)

    # 显示范围

    # 设置坐标轴比例
    # ax.set_box_aspect((10, 26, 10))
    # 设置观察视角

    # ax.dist = 1000
    # 设置颜色条，用(2是距离，3是强度)作为颜色（即点的强度）
    cbar = fig.colorbar(pc, location='left', shrink=0.5, fraction=0.05, pad=0.03)
    cbar.set_label('Intensity of points')
    # cbar.set_ticks([])  # 去掉 colorbar 的刻度
    # cbar.ax.set_yticklabels([])  # 去掉 colorbar 的数字
    return plt.show()


def plot_pc2d(pc):
    plt.close('all')
    x = pc[:, 0].astype(int)
    y = pc[:, 1].astype(int)
    z = pc[:, 2].astype(int)
    i = pc[:, 3].astype(int)
    # 遍历所有点，记录每个(x, y)的最大i
    max_i_map = np.zeros((64, 64))
    z_map = np.zeros((64, 64))
    # 遍历所有点，更新max_i_map
    for xi, yi, zi, ii in zip(x, y, z, i):
        # 由于x和y的范围是1-64，需要转换为0-63的索引
        idx_x = xi - 1
        idx_y = yi - 1
        if ii > max_i_map[idx_x, idx_y]:
            max_i_map[idx_x, idx_y] = ii
            z_map[idx_x, idx_y] = zi

    # 归一化图像到0-1范围
    # image_normalized = max_i_map / max_i_map.max()
    # 显示强度图像
    plt.figure(figsize=(10, 8))
    plt.imshow(max_i_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label=' ')
    plt.title('')
    plt.xlabel(' ')
    plt.ylabel(' ')
    plt.xticks([])
    plt.yticks([])
    # 显示网格
    plt.grid(True, linestyle='-', linewidth=0.5)
    plt.show()

    # 显示深度图像
    plt.figure(figsize=(10, 8))
    plt.imshow(z_map, cmap='viridis', interpolation='nearest', vmin=0, vmax=190)
    plt.colorbar(label='')
    plt.title('')
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    # 去除刻度数字
    plt.xticks([])
    plt.yticks([])
    # 显示网格
    plt.grid(True, linestyle='-', linewidth=0.5)
    plt.show()


def read_pc(file_path):
    """通用加载：支持 .txt / .npy / .npz"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        arr = np.loadtxt(file_path, delimiter=',', dtype=np.int32)
    elif ext == '.npy':
        arr = np.load(file_path)
    elif ext == '.npz':
        with np.load(file_path) as npz:
            # 假设只存了一个键 'pc'，若无则取第一个
            arr = npz[list(npz.keys())[0]]
    else:
        raise ValueError(f"不支持的文件格式: {ext}")

    if arr.dtype.kind == 'u':
        return arr.astype(np.int32, copy=False)
    return arr.astype(np.int32, copy=False)

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


def pc_down_sample_intensity_topk(pc, num):
    """
    根据强度(pc[:, 3])降采样，保留强度最高的num个点。
    输入输出均为(N, 4)的numpy数组
    """
    if pc.shape[0] <= num:
        return pc

    # 获取根据强度排序的索引 (argsort默认升序，[::-1]转为降序)
    # pc[:, 3] 是强度通道
    sorted_indices = np.argsort(pc[:, 3])[::-1]

    # 取前num个索引
    topk_indices = sorted_indices[:num]

    # 返回对应的点云数据
    return pc[topk_indices]


def save_pc(pc):
    return np.savetxt('E:\\essay\\硕士\\研一\\SPAD数据\\20250430dataset\\S1.txt', pc, fmt='%d', delimiter=',')


import math
from typing import Tuple, List, Dict
def augment_region_shift_rotate(points: np.ndarray,
                               region_box: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                               shift: Tuple[int, int, int]) -> Tuple[np.ndarray, Dict]:
    """
    将指定区域的点云进行旋转（基于区域中心）和位移，并与原位置进行交换（Swap）。
    旋转角度在函数内部定义为 [-pi/6, pi/6]。
    平移后的XYZ轴范围限定在:
    X: [0, 64], Y: [0, 64], Z: [60, 100]
    若平移后超出边界，则将其限制在边界处。

    Args:
        points: (N, 4) 点云数据
        region_box: 源区域 ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        shift: 偏移量 (dx, dy, dz)

    Returns:
        (augmented_points, label_info): 增强后的点云和对应的标签信息
    """
    if points.shape[0] == 0:
        return points, {}

    # 1. 定义旋转角度 (弧度)
    rot_rad = np.random.uniform(-np.pi/6, np.pi/6)

    # 解包源区域坐标范围
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = region_box
    dx, dy, dz = shift

    # 获取原始宽高深
    w = x_max - x_min
    h = y_max - y_min
    d = z_max - z_min

    # -------------------------------------------------------------
    # 2. 边界检查与修正 logic
    # -------------------------------------------------------------
    # 目标区域： X [0, 64], Y [0, 64], Z [60, 100]
    LIMIT_X = (0, 64)
    LIMIT_Y = (0, 64)
    LIMIT_Z = (60, 100)

    # 计算理论目标位置
    tgt_x_min = x_min + dx
    tgt_x_max = tgt_x_min + w

    tgt_y_min = y_min + dy
    tgt_y_max = tgt_y_min + h

    tgt_z_min = z_min + dz
    tgt_z_max = tgt_z_min + d

    # 修正 dx: 保证物体在 X [0, 64] 内
    # 左边界检查
    if tgt_x_min < LIMIT_X[0]:
        dx += (LIMIT_X[0] - tgt_x_min)
    # 右边界检查 (如果物体宽度大于64，优先保左边界或者居中，这里假设w < 64)
    tgt_x_max = x_max + dx # Recalculate with adjusted dx
    if tgt_x_max > LIMIT_X[1]:
        dx -= (tgt_x_max - LIMIT_X[1])

    # 修正 dy: 保证物体在 Y [0, 64] 内
    if tgt_y_min < LIMIT_Y[0]:
        dy += (LIMIT_Y[0] - tgt_y_min)
    tgt_y_max = y_max + dy
    if tgt_y_max > LIMIT_Y[1]:
        dy -= (tgt_y_max - LIMIT_Y[1])

    # 修正 dz: 保证物体在 Z [60, 100] 内
    if tgt_z_min < LIMIT_Z[0]:
        dz += (LIMIT_Z[0] - tgt_z_min)
    tgt_z_max = z_max + dz
    if tgt_z_max > LIMIT_Z[1]:
        dz -= (tgt_z_max - LIMIT_Z[1])

    # 如果修正后导致 min > max (即物体尺寸大过容器)，需要额外处理，这里暂不考虑极端情况

    # 更新修正后的目标范围
    tgt_x_min, tgt_x_max = x_min + dx, x_max + dx
    tgt_y_min, tgt_y_max = y_min + dy, y_max + dy
    tgt_z_min, tgt_z_max = z_min + dz, z_max + dz

    # -------------------------------------------------------------
    # 3. 数据处理 logic
    # -------------------------------------------------------------

    # 获取坐标引用
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # 确定源区域掩码
    src_mask = (x >= x_min) & (x < x_max) & \
               (y >= y_min) & (y < y_max) & \
               (z >= z_min) & (z < z_max)
    src_indices = np.where(src_mask)[0]

    # 确定目标区域掩码 (用于Swap交换)
    tgt_mask = (x >= tgt_x_min) & (x < tgt_x_max) & \
               (y >= tgt_y_min) & (y < tgt_y_max) & \
               (z >= tgt_z_min) & (z < tgt_z_max)
    tgt_indices = np.where(tgt_mask)[0]

    # 复制数据
    new_points = points.copy()

    # 处理源区域点：先局部旋转，再平移
    if len(src_indices) > 0:
        src_subset = new_points[src_indices].copy()

        # 计算区域中心 (用于旋转轴心)
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2

        # 局部坐标
        rel_y = src_subset[:, 1] - center_y
        rel_z = src_subset[:, 2] - center_z

        # 旋转 (绕Y轴转，只改变X和Z? 原始代码是rot_y和rot_z，这里似乎是绕X轴旋转?
        # 但根据变量名 rel_y/rel_z 像是绕X轴。如果是绕Y轴应该是 x,z 操作。
        # 这里的命名有点混淆，以此处逻辑为准：在 y-z 平面上旋转)
        c, s = np.cos(rot_rad), np.sin(rot_rad)
        rot_y = rel_y * c + rel_z * s
        rot_z = -rel_y * s + rel_z * c

        # 恢复绝对坐标 (在原位旋转结果)
        src_subset[:, 1] = np.round(rot_y + center_y).astype(int)
        src_subset[:, 2] = np.round(rot_z + center_z).astype(int)

        # 应用平移
        src_subset[:, 0] += dx
        src_subset[:, 1] += dy
        src_subset[:, 2] += dz

        # Swap操作：
        # 1. 目标区域的原有点 -> 移回源区域 (反向平移，不做旋转)
        if len(tgt_indices) > 0:
            new_points[tgt_indices, 0] -= dx
            new_points[tgt_indices, 1] -= dy
            new_points[tgt_indices, 2] -= dz

        # 2. 源区域的点 -> 放入目标区域
        new_points[src_indices] = src_subset

    # -----------------------------------------------------------------
    # 构造标签信息
    # -----------------------------------------------------------------
    label_info = {
        "label": "augmented_object",
        "x_range": [int(tgt_x_min), int(tgt_x_max)],
        "y_range": [int(tgt_y_min), int(tgt_y_max)],
        "z_range": [int(tgt_z_min), int(tgt_z_max)],
        "shift_applied": [int(dx), int(dy), int(dz)]
    }

    return new_points, label_info


# 文件路径 E:\\essay\\硕士\\研一\\SPAD数据\\20250430dataset\\2025-04-30-pc\\G\\2025-04-30_18-53-59_Delay-0_Width-200-3-5.txt
# filename1 = r'E:\\essay\\硕士\\研一\\SPAD数据\\20250430dataset\\2025-04-30-pc\\2025-04-30_18-47-35_Delay-0_Width-200-1-3.txt'

# f = r'F:\\20250418\\2025-04-18_16-32-29_Delay-0_Width-200.raw'
# active_point = int((endFrame - startFrame) * 0.03)
np.set_printoptions(threshold=np.inf)
if __name__ == '__main__':
    # # 可视化txt文件
    # rawfile = r'E:/essay/硕士/研一/SPAD数据/20260320/2026-03-20_16-39-17_Delay-0_Width-200.raw'
    # position,_ = raw2n3(rawfile)
    # pc = n3_filter(position)
    # # pc[:, 3] = np.where(pc[:, 3] >= 550, 550, pc[:, 3])
    # mask1 = pc[:, 3] >= 30
    # pc1 = pc[mask1]
    # mask2 = (pc1[:, 2] >=58) & (pc1[:, 2] <=63)
    # pc2 = pc1[mask2]
    # print(pc1.shape)
    # weights = pc[:, 3] ** 0.7
    # weights1 = weights/np.sum(weights)
    # indices = np.random.choice(pc.shape[0], 4096*20, p=weights1, replace=False)
    # pc_rd = pc[indices, :]
    # plot_pc(pc_rd,'all')
    pc = read_pc(r'D:\\PYproject\\SPADdata\\20250430\\lclof_results\\4_completed\\A_completed.txt')
    # pc_d = pc_down_sample_intensity_topk(pc,1024)
    # plot_pc2d(pc)
    mask1 = pc[:, 3] >= 50
    pc1 = pc[mask1]
    mask2 = (pc1[:, 2] >= 55) & (pc1[:, 2] <= 65)
    pc2 = pc1[mask2]
    # dx,dy,dz = np.random.randint(-20,20),np.random.randint(-5,30),np.random.randint(-20,15)
    # print(dx,dy,dz)
    # new_p, _ = augment_region_shift_rotate(pc, ((20,35),(5,25),(80,85)), (dx,dy,dz))
    # print(rot_rad*180/3.14)
    # plot_pc(pc_d, 'ds')
    # plot_pc(pc1, 'ds')
    # mask2 = (pc1[:, 2] >= 75) & (pc1[:, 2] <= 90)
    # pc2 = pc1[mask2]
    # mask3 = (pc2[:, 0] >= 20) & (pc2[:, 0] <= 35)
    # pc3 = pc2[mask3]
    print('pc shape:', pc.shape)
    plot_pc(pc, 'ds')
    # plot_pc2d(pc)
    # 可视化raw文件，保存
    # start = time.time()
    # rs_pc,data = raw2n3(f)
    # keep_indx = n3_filter(rs_pc, active_point=0)
    # save_pc(keep_indx)  # 保存的就是降采样之后的
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
