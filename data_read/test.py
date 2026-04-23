import numpy as np
import torch
import matplotlib.pyplot as plt

# 读取文件数据
file_path = r'D:\AntiUAV\dataset_infrared\SPAD\2024101211250400_yizi.Raw'

with open(file_path, 'rb') as file:
    raw_data = np.fromfile(file, dtype=np.uint8)

# 获取数据长度
data_len = len(raw_data)

# 检查数据长度是否为偶数，以确保低8位和高8位可以成对处理
if data_len % 2 != 0:
    raise ValueError('数据长度不是偶数，无法成对处理低8位和高8位数据。')

# 将数据的奇数位（低8位）和偶数位（高8位）提取出来
low8Bits = raw_data[0::2]  # 奇数位为低8位
high8Bits = raw_data[1::2]  # 偶数位为高8位

# 将低8位和高8位合并为16位数据
combined_data = np.uint16(low8Bits) + (np.uint16(high8Bits) << 8)

# 确定行数
num_rows = len(combined_data) // 4096

# 检查数据是否可以重构为每行 4096 个数据的矩阵
if len(combined_data) % 4096 != 0:
    raise ValueError('数据长度无法整除4096，无法构建矩阵。')

# 将数据重构为矩阵 (num_rows x 4096)
matrix_data = np.reshape(combined_data, (num_rows,4096))

# 只处理前 2000 行
matrix_data = np.int16(matrix_data[:2000, :])
# print(matrix_data)
# 将数据转换为 PyTorch 张量，并迁移到 GPU 上
tensor_data = torch.tensor(matrix_data, dtype=torch.float32).cuda()

print("数据已成功读取并处理为张量，张量大小为：", tensor_data.shape)

# 这里不修改 matrix_data 的任何操作
matrix_data = torch.tensor(matrix_data, dtype=torch.long).cuda()

# 创建一个最终的累加张量，大小为 (64, 64, 4096)，用于存储频率
final_tensor = torch.zeros((64, 64, 4096), dtype=torch.float32, device='cuda')

# 定义64x64空间范围
x_range = 64
y_range = 64
z_range = 4096

# 计算 x 和 y 的索引，利用向量化操作
x_indices = torch.arange(4096, device='cuda') % x_range  # x 索引
y_indices = (torch.arange(4096, device='cuda') // x_range) % y_range  # y 索引

# 我们不再显式创建稀疏张量，而是直接将稀疏点累加到最终张量
# 使用 index_put_ 进行批量操作

for row in matrix_data:
    # row 作为 z 方向的索引
    z_values = row  # z_value 本身就是4096维度上的坐标

    # 使用索引操作，直接将1累加到对应的 (x, y, z_value) 位置
    final_tensor.index_put_((x_indices, y_indices, z_values), torch.ones_like(z_values, dtype=torch.float32),
                            accumulate=True)

# 打印最终张量的大小
print("累加后的张量大小：", final_tensor.shape)
import numpy as np
import torch
from vispy import app, scene
from vispy.scene.visuals import Volume  # 正确导入 Volume

# 假设 final_tensor 已经计算完成，并在 CUDA 中
# 提取 final_tensor 中的 64x64x200 部分，并将其从 GPU 迁移回 CPU
final_tensor_cpu = final_tensor[:, :, :200].cpu().numpy()
print(final_tensor_cpu)
# 归一化 final_tensor_cpu 到 [0, 1] 之间
norm_tensor = (final_tensor_cpu - final_tensor_cpu.min()) / (final_tensor_cpu.max() - final_tensor_cpu.min())

def plot_pc(pc):
    print(pc.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc = ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=pc[:, 2], s=3, cmap='viridis')
    # 设置坐标轴
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 设置颜色条，用重复次数作为颜色（即点的强度）
    cbar = fig.colorbar(pc)
    cbar.set_label('Length of point')
    return plt.show()

plot_pc(final_tensor_cpu)

#
#
#
# # 创建 Vispy canvas，设置背景为白色
# canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white', size=(800, 600))
#
# # 创建 viewbox
# view = canvas.central_widget.add_view()
# view.camera = scene.cameras.TurntableCamera(fov=60, elevation=30, azimuth=30)
#
# # 创建体积渲染对象，并设置烟雾效果
# volume = Volume(norm_tensor, clim=(0, 1), method='mip', threshold=0.2)
# view.add(volume)
#
# # 开始运行 Vispy
# app.run()
#
