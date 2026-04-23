import os

# 环境设置
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '3'

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

"""
此代码采样结果为整数且去重
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The .grad attribute of a Tensor that is not a leaf Tensor")


class KDEApproximator:
    def __init__(self, xyz, intensities, bandwidth=1.0, device=device, max_points=1000000):
        # 如果数据点太多，随机采样一部分用于计算
        if len(xyz) > max_points:
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[idx]
            intensities = intensities[idx]
            print(f"Subsampled to {max_points} points for KDE")

        # 将数据转换为PyTorch张量并移到指定设备
        self.xyz = torch.tensor(xyz, dtype=torch.float32, device=device)
        self.intensities = torch.tensor(intensities, dtype=torch.float32, device=device)
        self.bandwidth = torch.tensor(bandwidth, device=device)
        self.device = device
        print(f"KDE initialized with {len(xyz)} points on {device}, bandwidth={bandwidth}")

        # 计算数据范围
        x_min, x_max = self.xyz[:, 0].min().item(), self.xyz[:, 0].max().item()
        y_min, y_max = self.xyz[:, 1].min().item(), self.xyz[:, 1].max().item()
        z_min, z_max = self.xyz[:, 2].min().item(), self.xyz[:, 2].max().item()
        print(f"Data range: x({x_min:.1f}-{x_max:.1f}), y({y_min:.1f}-{y_max:.1f}), z({z_min:.1f}-{z_max:.1f})")

    def evaluate(self, points, batch_size=1024):
        """评估点云强度 - 使用流式处理避免OOM"""
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=self.device)

        if points.ndim == 1:
            points = points.unsqueeze(0)
        if points.size(1) != 3:
            points = points.view(-1, 3)

        num_points = points.size(0)
        estimated_intensity = torch.zeros(num_points, device=self.device)

        # 使用流式处理计算距离
        for i in range(0, num_points, batch_size):
            batch_end = min(i + batch_size, num_points)
            batch_points = points[i:batch_end]

            # 计算当前批次与所有数据点的距离
            distances_sq = torch.sum((batch_points.unsqueeze(1) - self.xyz.unsqueeze(0)) ** 2, dim=2)

            # 高斯核函数
            weights = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))
            weights_sum = torch.sum(weights, dim=1) + 1e-10
            batch_intensity = torch.sum(weights * self.intensities, dim=1) / weights_sum

            estimated_intensity[i:batch_end] = batch_intensity

            # 清理中间变量
            del distances_sq, weights
            torch.cuda.empty_cache()

        return estimated_intensity.cpu().numpy()

    def evaluate_with_gradient(self, points, batch_size=128):  # 更小的批大小避免OOM
        """带梯度的强度评估 - 优化实现"""
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=self.device, requires_grad=True)
        elif not points.requires_grad:
            points = points.detach().clone().requires_grad_(True)

        if points.ndim == 1:
            points = points.unsqueeze(0)
        if points.size(1) != 3:
            points = points.view(-1, 3)

        num_points = points.size(0)
        estimated_intensity = torch.zeros(num_points, device=self.device)
        gradients = torch.zeros_like(points)

        # 流式处理计算梯度和强度
        for i in range(0, num_points, batch_size):
            batch_end = min(i + batch_size, num_points)
            batch_points = points[i:batch_end].detach().clone().requires_grad_(True)

            # 计算当前批次与所有数据点的距离
            diff = batch_points.unsqueeze(1) - self.xyz.unsqueeze(0)
            distances_sq = torch.sum(diff ** 2, dim=2)

            # 高斯核函数
            weights = torch.exp(-distances_sq / (2 * self.bandwidth ** 2))
            weights_sum = torch.sum(weights, dim=1, keepdim=True) + 1e-10
            normalized_weights = weights / weights_sum

            # 加权平均强度
            batch_intensity = torch.sum(normalized_weights * self.intensities, dim=1)
            estimated_intensity[i:batch_end] = batch_intensity.detach()

            # 计算梯度
            batch_intensity.sum().backward(retain_graph=False)
            gradients[i:batch_end] = batch_points.grad.detach().clone()

            # 清理中间变量
            del diff, distances_sq, weights, normalized_weights, batch_intensity
            torch.cuda.empty_cache()

        return estimated_intensity.cpu().numpy(), gradients.cpu().numpy()


class HMCSampler:
    def __init__(self, kde_approximator, initial_temp=20.0, min_temp=0.1,
                 n_levels=8, l=20, step_size=0.2, step_size_rate=0.5, burn_in_points=5):  # 增加到8个层级
        self.kde = kde_approximator
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.n_levels = n_levels
        self.burn_in_points = burn_in_points
        self.device = kde_approximator.device
        self.num_steps = l
        self.step_size = step_size
        self.step_size_rate = step_size_rate
        print(f"HMC Sampler initialized with {n_levels} levels")

        # 边界定义 [min_x, min_y, min_z], [max_x, max_y, max_z]
        self.bounds_min = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        self.bounds_max = torch.tensor([64.0, 64.0, 190.0], device=self.device)

        # 边界容差
        self.boundary_tolerance = 1e-3

    def U(self, q):
        """势能函数U = -ln(I+ε)"""
        intensity = self.kde.evaluate(q)
        return -np.log(intensity + 1e-8)

    def grad_U(self, q):
        """计算势能梯度：∇U = -∇I / (I + ε)"""
        intensities, gradient = self.kde.evaluate_with_gradient(q)
        intensities = intensities.reshape(-1, 1)
        ea = 1e-8
        grad_u = -gradient / (intensities + ea)
        return torch.tensor(grad_u, device=self.device, dtype=torch.float32)

    def reflect_boundary(self, q_t):
        """批量边界反射 - 优化实现"""
        # 对每个维度应用边界反射
        for i in range(3):
            # 下界反射
            mask_low = q_t[:, i] < self.bounds_min[i]
            if torch.any(mask_low):
                q_t[mask_low, i] = 2 * self.bounds_min[i] - q_t[mask_low, i]

            # 上界反射
            mask_high = q_t[:, i] > self.bounds_max[i]
            if torch.any(mask_high):
                q_t[mask_high, i] = 2 * self.bounds_max[i] - q_t[mask_high, i]

        # 确保在边界内
        q_t = torch.max(q_t, self.bounds_min + self.boundary_tolerance)
        q_t = torch.min(q_t, self.bounds_max - self.boundary_tolerance)
        return q_t

    def leapfrog(self, q, p, step_size, n_steps):
        """批量蛙跳积分 - 使用KDE梯度引导"""
        # 转换为张量
        q_t = torch.tensor(q, dtype=torch.float32, device=self.device)
        p_t = torch.tensor(p, dtype=torch.float32, device=self.device)

        # 初始半步动量更新
        grad_U = self.grad_U(q_t)
        p_t = p_t - 0.5 * step_size * grad_U

        # 主循环
        for i in range(int(n_steps)):
            # 位置全步更新
            q_t = q_t + step_size * p_t
            q_t = self.reflect_boundary(q_t)

            # 动量全步更新（除了最后一步）
            if i < n_steps - 1:
                grad_U = self.grad_U(q_t)
                p_t = p_t - step_size * grad_U

        # 最后半步动量更新
        grad_U = self.grad_U(q_t)
        p_t = p_t - 0.5 * step_size * grad_U

        return q_t.cpu().numpy(), p_t.cpu().numpy()

    def metropolis_hastings_step(self, q0, p0, q1, p1):
        """执行Metropolis-Hastings接受步骤"""
        # 计算势能 (原始值)
        U0 = self.U(q0)
        U1 = self.U(q1)

        # 计算动能
        K0 = np.sum(p0 ** 2, axis=1) / 2.0
        K1 = np.sum(p1 ** 2, axis=1) / 2.0

        # 计算哈密尔顿量
        H0 = U0 + K0
        H1 = U1 + K1

        # 计算接受概率
        delta_H = H1 - H0
        accept_prob = np.exp(-delta_H)
        accept_prob = np.minimum(1.0, accept_prob)

        # 生成随机数并决定是否接受
        rand = np.random.rand(len(q0))
        accepted = rand < accept_prob

        # 创建接受点数组
        accepted_points = np.where(accepted[:, None], q1, q0)

        return accepted_points, accepted

    def get_unique_integer_points(self, points):
        """获取唯一的整数坐标点"""
        # 四舍五入取整
        int_points = np.round(points).astype(int)

        # 确保在边界内
        int_points = np.clip(int_points, [1, 1, 1], [64, 64, 190])

        # 去重
        unique_points = np.unique(int_points, axis=0)

        return unique_points

    def select_initial_points(self, original_data):
        """使用KNN聚类选择初始点 - 按3-3-2比例采样"""
        # 筛选高强度点
        intensities = original_data[:, 3]
        high_intensity_mask = intensities >= 110
        high_intensity_points = original_data[high_intensity_mask, :3]
        high_intensities = intensities[high_intensity_mask]  # 保存对应的强度值
        print(f"Found {len(high_intensity_points)} high-intensity points for clustering")

        # 如果高亮点不足，使用所有点
        if len(high_intensity_points) < 8:
            print(f"Warning: Only {len(high_intensity_points)} high intensity points. Using all points.")
            high_intensity_points = original_data[:, :3]
        # 1. 增强Z轴权重 - 特征缩放
        # 计算各维度范围
        x_range = max(high_intensity_points[:, 0].max() - high_intensity_points[:, 0].min(), 1)
        y_range = max(high_intensity_points[:, 1].max() - high_intensity_points[:, 1].min(), 1)
        z_range = max(high_intensity_points[:, 2].max() - high_intensity_points[:, 2].min(), 1)

        # 创建缩放后的点集 - 增强Z轴重要性
        scaled_points = high_intensity_points.copy()
        scaled_points[:, 0] /= x_range  # X轴归一化
        scaled_points[:, 1] /= y_range  # Y轴归一化
        scaled_points[:, 2] /= (z_range * 0.2)  # Z轴权重增强 (减小分母相当于增加权重)
        # 创建权重向量（归一化强度）
        weights = high_intensities / np.max(high_intensities)
        # 2. KNN聚类 - 使用KMeans替代KNN聚类
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, init='k-means++')
        kmeans.fit(scaled_points, sample_weight=weights)
        labels = kmeans.labels_

        # 3. 按3-3-2比例从每个聚类中采样点
        initial_points = []
        cluster_counts = [0, 0, 0]
        sample_counts = [3, 2, 3]  # 每类采样点数

        for cluster_idx in range(3):
            cluster_indices = np.where(labels == cluster_idx)[0]
            cluster_points = high_intensity_points[cluster_indices]

            # 如果聚类点数不足，使用所有点
            if len(cluster_points) < sample_counts[cluster_idx]:
                print(f"Warning: Cluster {cluster_idx} has only {len(cluster_points)} points. Using all points.")
                selected = cluster_points
            else:
                # 随机选择指定数量的点
                selected_indices = np.random.choice(len(cluster_points), sample_counts[cluster_idx], replace=False)
                selected = cluster_points[selected_indices]

            initial_points.append(selected)
            cluster_counts[cluster_idx] = len(selected)

        initial_points = np.vstack(initial_points)
        print(f"Selected {len(initial_points)} initial points via KNN clustering (cluster counts: {cluster_counts})")
        return initial_points

    def sample(self, original_data):
        """层级式采样 - 8级链式采样 (8->16->32->64->128->256->512->1024)"""
        # 使用KNN选择初始点
        current_points = self.select_initial_points(original_data)
        print(f"Starting with {len(current_points)} seed points")
        print(current_points)
        final_point = None
        # 逐级采样
        for level in range(self.n_levels):
            # 计算当前温度（退火过程）
            # t = max(self.min_temp, self.initial_temp * (0.7 ** level))
            # self.current_temp = t
            print(f"\n{'=' * 50}\nProcessing level {level + 1}/{self.n_levels}")
            print(f"Current seed points: {len(current_points)}")
            # print(f"Current temperature: {t:.4f}")

            # 计算目标点数
            target_points = len(current_points) * 2  # 每级翻倍
            print(f"Target points for this level: {target_points}")

            # 尝试最多4次以获得足够的不重复点
            attempt = 0
            max_attempts = 4
            best_points = None
            best_count = 0

            while attempt < max_attempts:
                attempt += 1
                print(f"\nAttempt {attempt}/{max_attempts} for level {level + 1}")

                # 每个种子点生成2条链
                n_seeds = len(current_points)
                n_chains = 2
                total_chains = n_seeds * n_chains

                # 准备初始位置 - 每个种子点重复2次（对应两条链）
                initial_positions = np.repeat(current_points, n_chains, axis=0)

                # 初始化当前位置为初始位置
                current_positions = initial_positions.copy()

                # 动态调整步长和步数
                n_steps = self.num_steps - 3 * level
                step_size = self.step_size * (self.step_size_rate ** level)
                print(f"Number of steps: {int(n_steps):.4f}")
                print(f"Step size: {step_size:.4f}")

                # 执行burn-in阶段
                burn_in = self.burn_in_points - level
                # print(f"Starting burn-in phase with {n_steps} steps per chain")
                for burn_in_step in range(burn_in):
                    # 为每条链采样新的动量
                    momenta = np.random.normal(0, 1, size=(total_chains, 3))

                    # 执行HMC步骤
                    proposed_positions, proposed_momenta = self.leapfrog(current_positions, momenta, step_size, n_steps)

                    # 执行Metropolis-Hastings接受步骤
                    new_positions, accepted = self.metropolis_hastings_step(
                        current_positions, momenta, proposed_positions, proposed_momenta
                    )

                    # 更新当前位置为接受后的位置
                    current_positions = new_positions

                    # 打印当前burn-in步骤的接受率
                    acceptance_rate = np.mean(accepted) * 100
                    print(
                        f"Burn-in step {burn_in_step + 1}/{burn_in}: acceptance rate = {acceptance_rate:.2f}%")

                    # 清理内存
                    del momenta, proposed_positions, proposed_momenta
                    gc.collect()
                    torch.cuda.empty_cache()

                # burn-in阶段结束，保留最后的位置作为新点
                new_points = current_positions

                # 修剪到有效范围
                new_points = np.clip(new_points, [1, 1, 1], [64, 64, 190])

                # 获取唯一整数点
                unique_points = self.get_unique_integer_points(new_points)
                unique_count = len(unique_points)

                print(f"Generated {unique_count} unique integer points (target: {target_points})")

                # 记录最佳结果（最接近目标点数的尝试）
                if unique_count >= target_points or (best_points is None or unique_count > best_count):
                    best_points = unique_points
                    best_count = unique_count

                    # 如果已经达到目标点数，提前结束尝试
                    if unique_count >= target_points:
                        print(f"Target reached at attempt {attempt}, stopping early")
                        break

                # 更新当前种子点为本次尝试的接受点（浮点数）用于下一次尝试
                current_points = new_points

                # 显式清理内存
                del initial_positions, new_points
                gc.collect()
                torch.cuda.empty_cache()

            # 选择最佳尝试结果
            if best_points is None:
                best_points = np.empty((0, 3))
                best_count = 0

            # 如果点数超过目标，随机采样目标点数
            if best_count > target_points:
                indices = np.random.choice(best_count, target_points, replace=False)
                current_points = best_points[indices]
            else:
                current_points = best_points
                print(f"Warning: Only {best_count} unique points generated (target: {target_points})")

            # 存储当前层级的点（只保留最终结果）
            if level == self.n_levels - 1:
                final_point = current_points.copy()

            # 打印层级完成信息
            print(f"Level {level + 1} completed. Generated {len(current_points)} unique integer points.")

        # 返回所有层级的点
        return final_point


def plot_3d_points(points, intensities, title="3D Point Cloud"):
    """绘制3D点云 - 固定坐标轴范围，不裁剪点"""
    print('Plot ', points.shape)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if len(points) > 0:
        # 绘制点云
        sc = ax.scatter(points[:, 1], points[:, 2], np.abs(points[:, 0] - 65),
                        c=intensities, cmap='viridis', s=3, alpha=0.8)

        # 添加颜色条
        cbar = fig.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label('Intensity', fontsize=12)
    else:
        print("Warning: No points to plot.")

    # 设置坐标轴标签和范围
    ax.set_xlabel('Y', fontsize=12, labelpad=10)
    ax.set_ylabel('Z', fontsize=12, labelpad=10)
    ax.set_zlabel('X', fontsize=12, labelpad=10)

    # # 固定坐标轴范围
    # ax.set_xlim(0, 64)
    # ax.set_ylim(0, 190)
    # ax.set_zlim(0, 64)

    # 设置视角
    ax.view_init(elev=20, azim=-60)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)

    # 设置标题
    ax.set_title(title, fontsize=14, pad=15)

    plt.tight_layout()
    plt.show()


def main(input_file):
    print("Starting HMC sampling process...")
    start_time = time.time()

    # 1. 加载数据
    print(f"Loading data from {input_file}")
    try:
        original_data = np.loadtxt(input_file, delimiter=',')
        print(f"Loaded {len(original_data)} points")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # 2. 初始化KDE
    print("Initializing KDE...")
    try:
        kde = KDEApproximator(original_data[:, :3], original_data[:, 3], bandwidth=1.0, device=device)
    except Exception as e:
        print(f"Error initializing KDE: {e}")
        raise

    # 3. 初始化采样器
    print("Initializing HMC sampler...")
    try:
        sampler = HMCSampler(kde, initial_temp=10.0, min_temp=1.0, n_levels=7,
                             l=35, step_size=0.4, step_size_rate=0.75, burn_in_points=10)
    except Exception as e:
        print(f"Error initializing s sampler: {e}")
        raise

    # 4. 执行采样
    print("Starting sampling process...")
    try:
        samples = sampler.sample(original_data)
        print(f"Total sampled points: {len(samples)}")
    except Exception as e:
        raise TypeError(f"Error during sampling: {e}")
        # print(f"Error during sampling: {e}")
        # samples = np.array([])

    # 打印采样点信息
    if len(samples) > 0:
        print(f"Sample points shape: {samples.shape}")
        print(f"First 5 points: {samples[:5]}")
        print(f"Last 5 points: {samples[-5:]}")

        # 检查唯一性
        unique_samples = np.unique(samples, axis=0)
        print(f"Unique points: {len(unique_samples)}")

        # 如果有重复点，显示重复数量
        if len(samples) != len(unique_samples):
            print(f"Warning: {len(samples) - len(unique_samples)} duplicate points found")

    # 5. 评估采样点强度（仅用于可视化）
    print("Evaluating sample intensities for visualization...")
    if len(samples) > 0:
        try:
            sample_intensities = kde.evaluate(samples)
            print(f"Intensities min: {np.min(sample_intensities):.4f}, max: {np.max(sample_intensities):.4f}")
        except Exception as e:
            print(f"Error evaluating intensities: {e}")
            sample_intensities = np.zeros(len(samples))
    else:
        sample_intensities = np.array([])
        print("Warning: No samples to evaluate.")
    print('points intensities is:', sample_intensities[:5])
    # 计算并打印总时间
    total_time = time.time() - start_time
    print(f"Process completed in {total_time:.2f} seconds")

    # 6. 可视化 - 不裁剪点
    print("Visualizing results without cropping...")
    try:
        plot_3d_points(samples, sample_intensities, "Sampled Points")
    except Exception as e:
        print(f"Error during visualization: {e}")

    # # 7.保存结果-最后1024个
    si_r = sample_intensities.reshape(-1, 1)
    points4 = np.hstack((samples, si_r))
    np.savetxt('A1.txt', points4, delimiter=',', fmt='%d')


if __name__ == '__main__':
    input_file = 'D:\\PYproject\\SPAD\\HMC\\2025-04-30_18-47-35_Delay-0_Width-200-1-3.txt'
    main(input_file)
