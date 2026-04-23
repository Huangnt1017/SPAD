import os

# 环境设置
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '3'

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings
import gc
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The .grad attribute of a Tensor that is not a leaf Tensor")


class RBFApproximator:
    def __init__(self, original_data, epsilon=1.0, device=device):
        self.original_data = original_data
        self.epsilon = torch.tensor(epsilon, device=device)
        self.device = device

        # 预计算强度网格
        self.intensity_grid = torch.zeros((65, 65, 191), device=device)
        for point in original_data:
            x, y, z, i = point
            self.intensity_grid[int(x), int(y), int(z)] = i

        print(f"RBFApproximator initialized with grid")

    def get_surrounding_points(self, points):
        int_points = torch.round(points).long()
        offsets = torch.tensor([
            [dx, dy, dz] for dx in [-2, -1, 0, 1, 2] for dy in [-2, -1, 0, 1, 2] for dz in [-2, -1, 0, 1, 2]
            if not (dx == 0 and dy == 0 and dz == 0)
        ], device=device, dtype=torch.long)

        surrounding_coords = int_points.unsqueeze(1) + offsets.unsqueeze(0)

        # 边界裁剪
        x_coords = surrounding_coords[..., 0].clamp(1, 64)
        y_coords = surrounding_coords[..., 1].clamp(1, 64)
        z_coords = surrounding_coords[..., 2].clamp(1, 190)

        # 直接索引预计算网格
        surrounding_intensities = self.intensity_grid[x_coords, y_coords, z_coords]

        return surrounding_coords.float(), surrounding_intensities

    def _compute_rbf_components(self, batch_points):
        """计算RBF组件：权重、系数和位置差（公共部分）"""
        # 获取周围点
        surrounding_coords, surrounding_intensities = self.get_surrounding_points(batch_points)

        # 计算距离和权重
        diff = batch_points.unsqueeze(1) - surrounding_coords
        distances_sq = torch.sum(diff ** 2, dim=2)
        weights = torch.exp(-(self.epsilon ** 2) * distances_sq)

        # 计算矩阵A
        surrounding_diff = surrounding_coords.unsqueeze(2) - surrounding_coords.unsqueeze(1)
        surrounding_distances_sq = torch.sum(surrounding_diff ** 2, dim=3)
        A = torch.exp(-(self.epsilon ** 2) * surrounding_distances_sq) + \
            torch.eye(124, device=self.device).unsqueeze(0) * 1e-5

        # 求解系数
        w = torch.linalg.solve(A, surrounding_intensities.unsqueeze(-1)).squeeze(-1)

        return weights, w, diff, surrounding_coords, surrounding_intensities

    def evaluate(self, points, batch_size=4096):
        """评估点云强度 - 使用RBF局部拟合"""
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=self.device)

        if points.ndim == 1:
            points = points.unsqueeze(0)
        if points.size(1) != 3:
            points = points.view(-1, 3)

        num_points = points.size(0)
        estimated_intensity = torch.zeros(num_points, device=self.device)

        # 批量处理
        for i in range(0, num_points, batch_size):
            batch_end = min(i + batch_size, num_points)
            batch_points = points[i:batch_end]

            # 使用公共方法计算RBF组件
            weights, w, diff, surrounding_coords, surrounding_intensities = \
                self._compute_rbf_components(batch_points)

            # 计算强度
            batch_intensity = torch.sum(weights * w, dim=1)
            estimated_intensity[i:batch_end] = batch_intensity

            # 清理中间变量
            del weights, w, diff, surrounding_coords, surrounding_intensities
            torch.cuda.empty_cache()

        return estimated_intensity

    def evaluate_with_gradient(self, points, batch_size=4096):
        """带梯度的强度评估"""
        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32, device=self.device)

        if points.ndim == 1:
            points = points.unsqueeze(0)
        if points.size(1) != 3:
            points = points.view(-1, 3)

        num_points = points.size(0)
        estimated_intensity = torch.zeros(num_points, device=self.device)
        gradients = torch.zeros((num_points, 3), device=self.device)

        # 分批处理
        for i in range(0, num_points, batch_size):
            batch_end = min(i + batch_size, num_points)
            batch_points = points[i:batch_end]

            # 使用公共方法计算RBF组件
            weights, w, diff, surrounding_coords, surrounding_intensities = \
                self._compute_rbf_components(batch_points)

            # 计算强度
            batch_intensity = torch.sum(weights * w, dim=1)
            estimated_intensity[i:batch_end] = batch_intensity

            # 计算梯度
            grad_factor = -2 * (self.epsilon ** 2) * weights.unsqueeze(2) * w.unsqueeze(2)
            batch_grad = torch.sum(grad_factor * diff, dim=1)
            gradients[i:batch_end] = batch_grad

            # 主动释放内存
            del weights, w, diff, surrounding_coords, surrounding_intensities, grad_factor
            torch.cuda.empty_cache()

        return estimated_intensity, gradients


class HMCSampler:
    def __init__(self, rbf_approximator, initial_temp=20.0, min_temp=0.1,
                 l=20, step_size=0.2, step_size_rate=0.5, burn_in_points=5):
        self.rbf = rbf_approximator
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.min_temp = min_temp

        self.burn_in_points = burn_in_points
        self.device = rbf_approximator.device
        self.num_steps = l
        self.step_size = step_size
        self.step_size_rate = step_size_rate

        # 边界定义 [min_x, min_y, min_z], [max_x, max_y, max_z]
        self.bounds_min = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        self.bounds_max = torch.tensor([64.0, 64.0, 190.0], device=self.device)

        # 边界容差
        self.boundary_tolerance = 1e-3

    def U(self, q):
        """势能函数U = -ln(I+ε)"""
        intensity = self.rbf.evaluate(q)
        u = -torch.log(torch.tensor(intensity.clone().detach(), device=self.device) + 1e-5)
        return u

    def grad_U(self, q):
        """计算势能梯度：∇U = -∇I / (I + ε)"""
        intensities, gradient = self.rbf.evaluate_with_gradient(q)
        intensities1 = intensities.clone().detach().unsqueeze(1)
        ea = 1e-8
        grad_u = gradient.clone().detach() / (intensities1 + ea)
        return grad_u

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
        """批量蛙跳积分 - 使用RBF梯度引导"""
        # 转换为张量
        q_t = q.clone().detach()
        p_t = p.clone().detach()

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

        return q_t, p_t

    def metropolis_hastings_step(self, q0, p0, q1, p1):
        """执行Metropolis-Hastings接受步骤
        输入全是torch.tensor
        """
        # 计算势能 (-ln(I))
        U0 = self.U(q0)
        U1 = self.U(q1)

        # 计算动能
        K0 = torch.sum(p0 ** 2, dim=1) / 2.0
        K1 = torch.sum(p1 ** 2, dim=1) / 2.0

        # 计算哈密尔顿量
        H0 = U0 + K0
        H1 = U1 + K1
        # print(H0, H1)
        # 计算接受概率
        delta_H = H1 - H0
        # 添加强度偏置函数，accept_prob受强度变化影响
        intensity_bias = torch.tanh(3*(U0 - U1))  # 归一化
        # print(intensity_bias)
        acc1 = torch.exp(-delta_H)
        acc2 = acc1 * (1 + intensity_bias)
        # 添加温度调节，随着iteration的增大而降低温度平滑
        # temperature = 10
        # acc3 = acc2 * temperature
        accept_prob = torch.min(torch.ones_like(delta_H), acc2)

        # 生成随机数并决定是否接受
        rand = torch.rand_like(accept_prob)
        accepted = rand < accept_prob

        # 创建接受点数组
        accepted_points = torch.where(accepted.unsqueeze(1), q1, q0)

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

    def select_initial_points(self, original_data, n_points=16, ratio=(3, 3, 2),
                              intensity_percentile=90, bandwidth=1.0,
                              min_peak_distance=4.0, random_state=42):
        """
        自适应Z轴分层的初始点选择方法 (简化版)

        参数:
        original_data: 原始数据 (N, 4) [x, y, z, intensity]
        n_points: 目标点数 (默认16)
        ratio: 各层采样比例 (默认6:6:4)
        intensity_percentile: 高亮点筛选百分位 (默认99.85%)
        bandwidth: KDE带宽参数 (默认3.0)
        min_peak_distance: 峰值最小距离 (默认4.0)
        random_state: 随机种子

        返回:
        选择的初始点坐标 (n_points, 3)
        """
        np.random.seed(random_state)

        # 1. 筛选高亮点
        intensities = original_data[:, 3]
        i_t = np.percentile(intensities, intensity_percentile)
        mask = intensities >= i_t
        high_points = original_data[mask]

        # 处理高亮点不足的情况
        if len(high_points) < n_points:
            print(f"使用所有点: 高亮点不足 ({len(high_points)} < {n_points})")
            high_points = original_data
            intensities = original_data[:, 3]
        else:
            intensities = high_points[:, 3]

        z_values = high_points[:, 2]

        # 2. 使用KDE检测Z轴密度峰值
        kde = gaussian_kde(z_values, bw_method=bandwidth)
        x = np.linspace(np.min(z_values), np.max(z_values), int(np.max(z_values) - np.min(z_values)))
        density = kde(x)

        # 寻找峰值 (峰间最小距离=min_peak_distance)
        peaks, _ = find_peaks(density, distance=min_peak_distance)
        peak_z = x[peaks]
        print(peak_z)

        # 3. 基于峰值分层 (最多3层)
        n_layers = min(3, len(peak_z))
        if n_layers == 0:  # 无峰值情况
            layers = [(np.min(z_values), np.max(z_values))]
            # n_layers = 3(一般来说)
        else:
            # 按Z值排序峰值
            peak_z_sorted = np.sort(peak_z[:n_layers])
            layers = []

            # 确定层边界
            for i in range(n_layers):
                if i == 0:
                    lower = np.min(z_values)
                else:
                    lower = (peak_z_sorted[i - 1] + peak_z_sorted[i]) / 2

                if i == n_layers - 1:
                    upper = np.max(z_values)
                else:
                    upper = (peak_z_sorted[i] + peak_z_sorted[i + 1]) / 2

                layers.append((lower, upper))
        print(layers)
        print(f"检测到 {n_layers} 个Z轴分层:")
        for i, (low, high) in enumerate(layers):
            print(f"  层 {i}: Z ∈ [{low:.1f}, {high:.1f}]")

        # 4. 将点分配到各层
        layer_points = [[] for _ in range(n_layers)]
        layer_intensities = [[] for _ in range(n_layers)]

        for point, intensity in zip(high_points, intensities):
            z = point[2]
            for i, (low, high) in enumerate(layers):
                if low <= z <= high:
                    layer_points[i].append(point[:3])
                    layer_intensities[i].append(intensity)
                    break

        # 5. 调整采样比例
        if n_layers != len(ratio):
            # 等比例分配
            ratio = [sum(ratio) // n_layers] * n_layers
            # 确保总数正确
            # ratio[0] += n_points - sum(ratio)
        print(ratio)
        # 6. 分层采样
        selected_points = []
        total_points = n_points
        # n_needed = [int(total_points * ratio[0] / sum(ratio)),
        #             int(total_points * ratio[1] / sum(ratio)),
        #             int(total_points * ratio[2] / sum(ratio))]
        for i in range(n_layers):
            points_arr = np.array(layer_points[i])
            intensities_arr = np.array(layer_intensities[i])

            if len(points_arr) == 0:
                print(f"层 {i} 无点可采")
                raise
            n_needed = int(total_points * ratio[i] / sum(ratio))
            # 计算该类90%强度阈值
            intensity_threshold = np.percentile(intensities_arr, 90)

            # 筛选高强度点 (类内强度前10%)
            high_intensity_mask = intensities_arr >= intensity_threshold
            high_intensity_points = points_arr[high_intensity_mask]
            high_intensities = intensities_arr[high_intensity_mask]
            # (min(ratio[i], len(points_arr)))

            # 优先从高强度点中选取
            if len(high_intensity_points) >= n_needed:
                # 使用强度加权随机选择
                weights = high_intensities / np.sum(high_intensities)
                selected_idx = np.random.choice(
                    len(high_intensity_points),
                    n_needed,
                    replace=False,
                    p=weights
                )
                selected = high_intensity_points[selected_idx]
            else:
                # 高强度点不足，先取所有高强度点
                selected = high_intensity_points

                # 再从剩余点中补足
                remaining_needed = n_needed - len(selected)
                remaining_mask = ~high_intensity_mask
                remaining_points = points_arr[remaining_mask]
                remaining_intensities = intensities_arr[remaining_mask]

                if len(remaining_points) > 0:
                    # 使用强度加权随机选择
                    weights = remaining_intensities / np.sum(remaining_intensities)
                    selected_idx = np.random.choice(
                        len(remaining_points),
                        min(remaining_needed, len(remaining_points)),
                        replace=False,
                        p=weights
                    )
                    selected = np.vstack([selected, remaining_points[selected_idx]])

            selected_points.append(selected)
            print(f"层 {i}: 选取 {len(selected)} 个点 (需求: {n_needed})")

        # 7. 合并结果
        if selected_points:
            all_points = np.vstack(selected_points)
        else:
            all_points = np.empty((0, 3))

        # 8. 点数不足时补全
        if len(all_points) < n_points:
            shortage = n_points - len(all_points)
            print(f"点数不足 ({len(all_points)} < {n_points}), 补全 {shortage} 个点")

            # 从所有点中随机选择补足
            weights = intensities / np.sum(intensities)
            selected_idx = np.random.choice(
                len(high_points),
                shortage,
                replace=False,
                p=weights
            )
            additional_points = high_points[selected_idx, :3]
            all_points = np.vstack([all_points, additional_points])

        # 确保点数准确
        if len(all_points) > n_points:
            all_points = all_points[:n_points]

        print(f"最终选取 {len(all_points)} 个初始点")
        return all_points

    def sample(self, original_data, target_points=1024, chain=8):
        """并行8条马尔科夫链采样，直到收集到1024个有效点"""
        # 创建坐标-强度映射字典 (用于快速查找)
        self.coord_intensity_map = {}
        for point in original_data:
            x, y, z, i = point
            key = (int(x), int(y), int(z))
            self.coord_intensity_map[key] = i

        # 使用KNN选择8个初始点
        initial_seeds = self.select_initial_points(original_data, n_points=chain)
        print(f"Starting with {len(initial_seeds)} seed points")
        print(initial_seeds)
        # 初始化8条链的当前位置
        current_positions = initial_seeds.copy()
        current_positions = torch.tensor(current_positions, device=device)
        # 初始化采样点集合
        collected_points = set()
        collected_samples = []

        # 1. Burn-in阶段 (不记录点)
        print(f"Starting burn-in phase with {self.burn_in_points} steps per chain")
        for step in tqdm(range(self.burn_in_points)):
            # 为每条链采样新的动量
            momenta = torch.randn(chain, 3, device=self.device) * 3.0

            # 执行HMC步骤
            proposed_positions, proposed_momenta = self.leapfrog(
                current_positions, momenta, self.step_size, self.num_steps
            )

            # 执行Metropolis-Hastings接受步骤
            new_positions, accepted = self.metropolis_hastings_step(
                current_positions, momenta, proposed_positions, proposed_momenta
            )

            # 更新当前位置为接受后的位置
            current_positions = new_positions

            # 打印当前接受率
            # acceptance_rate = np.mean(accepted) * 100
            # print(f"Burn-in step {step+1}/{self.burn_in_points}: acceptance rate = {acceptance_rate:.2f}%")

            # 清理内存
            del momenta, proposed_positions, proposed_momenta
            gc.collect()
            torch.cuda.empty_cache()

        print("Burn-in phase completed. Starting sampling phase...")

        # 2. 正式采样阶段
        max_iterations = 10000
        iteration = 0

        with tqdm(total=target_points, desc="Collecting points") as pbar:
            while len(collected_samples) < target_points and iteration < max_iterations:
                iteration += 1
                ss = np.random.choice([0.005, 0.006, 0.011, 0.007, 0.009, 0.013, 0.015, 0.008, 0.010, 0.012, 0.014])
                ns = np.random.choice([25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])
                # 为每条链采样新的动量
                momenta = torch.randn(chain, 3, device=self.device) * torch.tensor([3.0, 3.0, 3.0], device=self.device)

                # 执行HMC步骤
                proposed_positions, proposed_momenta = self.leapfrog(
                    current_positions, momenta, ss, ns)

                # 执行Metropolis-Hastings接受步骤
                new_positions, accepted = self.metropolis_hastings_step(
                    current_positions, momenta, proposed_positions, proposed_momenta)

                # 更新当前位置为接受后的位置
                current_positions = new_positions

                # 只在接受新点时处理采样点
                if iteration % 1 == 0:
                    for i in range(chain):
                        if accepted[i]:
                            # 将连续点坐标转换为整数坐标
                            int_point = np.round(new_positions.cpu().numpy()[i]).astype(int)
                            x, y, z = int_point

                            # 确保在边界内
                            if (1 <= x <= 64) and (1 <= y <= 64) and (1 <= z <= 190):
                                # 检查强度是否为0
                                key = tuple(int_point)
                                intensity = self.coord_intensity_map.get(key, 0.0)

                                if intensity > 0:
                                    # 检查是否为新点
                                    if key not in collected_points:
                                        collected_points.add(key)
                                        collected_samples.append(int_point)
                                        pbar.update(1)

                                        # 达到目标点数则退出
                                        if len(collected_samples) >= target_points:
                                            break

                # 清理内存
                del momenta, proposed_positions, proposed_momenta
                gc.collect()
                torch.cuda.empty_cache()

                # 每100次迭代显示进度
                if iteration % 100 == 0:
                    print(f"Iteration {iteration}: collected {len(collected_samples)} points")

        # 转换为NumPy数组
        samples = np.array(collected_samples[:target_points])

        print(f"Sampling completed. Collected {len(samples)} points after {iteration} iterations")
        return samples


def plot_3d_points(points, intensities, title="3D Point Cloud"):
    """绘制3D点云 - 固定坐标轴范围，不裁剪点"""
    print(points.shape)
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

    # 设置视角
    ax.view_init(elev=20, azim=-60)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(1, 64)
    ax.set_ylim(40, 120)
    ax.set_zlim(1, 64)
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

    # 2. 初始化RBF近似器
    print("Initializing RBF Approximator...")
    try:
        rbf_approx = RBFApproximator(original_data, epsilon=1.0, device=device)
    except Exception as e:
        print(f"Error initializing RBF Approximator: {e}")
        raise

    # 3. 初始化采样器
    print("Initializing HMC sampler...")
    try:
        sampler = HMCSampler(rbf_approx, l=30, step_size=0.008, burn_in_points=20)
    except Exception as e:
        print(f"Error initializing sampler: {e}")
        raise

    # 4. 执行采样
    print("Starting sampling process...")
    try:
        samples = sampler.sample(original_data, target_points=1024, chain=32)
        print(f"Total sampled points: {len(samples)}")
    except Exception as e:
        print(f"Error during sampling: {e}")
        # samples = np.array([])
        raise

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

    # 5. 直接从原始数据获取采样点强度
    print("Retrieving actual sample intensities from original data...")
    if len(samples) > 0:
        try:
            # 创建原始数据的坐标-强度映射字典
            coord_intensity_map = {}
            for point in original_data:
                x, y, z, i = point
                key = (int(x), int(y), int(z))
                coord_intensity_map[key] = i

            # 查找实际强度值
            sample_intensities = np.zeros(len(samples))
            for i, pt in enumerate(samples):
                key = tuple(pt)
                sample_intensities[i] = coord_intensity_map.get(key, 0.0)

            print(f"Actual intensities min: {np.min(sample_intensities):.4f}, max: {np.max(sample_intensities):.4f}")
        except Exception as e:
            print(f"Error retrieving actual intensities: {e}")
            sample_intensities = np.zeros(len(samples))
    else:
        sample_intensities = np.array([])
        print("Warning: No samples to evaluate.")

    print('Points intensities:', sample_intensities[:5])

    # 计算并打印总时间
    total_time = time.time() - start_time
    print(f"Process completed in {total_time:.2f} seconds")

    # 6. 可视化
    print("Visualizing results...")
    try:
        plot_3d_points(samples, sample_intensities, "Sampled Points")
    except Exception as e:
        print(f"Error during visualization: {e}")

    # 7. 保存结果
    if len(samples) > 0:
        # 将强度值转为列向量
        si_r = sample_intensities.reshape(-1, 1)
        # 合并坐标和强度
        points4 = np.hstack((samples, si_r))
        output_file = 'D1.txt'
        np.savetxt(output_file, points4, delimiter=',', fmt='%d')
        print(f"Saved results to {output_file}")


if __name__ == '__main__':
    input_file = 'D:\\PYproject\\SPAD\\HMC\\2025-04-30_18-51-28_Delay-0_Width-200-11-13.txt'
    main(input_file)
