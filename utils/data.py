import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path

"""
数据处理相关工具函数和数据集类
包括：
1. 点云数据集类PointCloudDataset，支持grid和ds两种模式
2. 自定义collate函数，处理变长标签
3. 创建DataLoader的函数create_dataloaders


"""

# ============================================
# 设置随机种子
# ============================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# ============================================
# 辅助函数
# ============================================
def downsample(points: np.ndarray, num: int) -> np.ndarray:
    """
    降采样函数（占位符）

    Args:
        points: 原始点云，形状为 (N, 4)
        num: 目标点数

    Returns:
        降采样后的点云，形状为 (num, 4)
    """
    # 这里只是简单的均匀采样，实际使用时需要替换为您的降采样算法
    if len(points) <= num:
        return points
    indices = np.random.choice(len(points), num, replace=False)
    return points[indices]


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    自定义collate函数，处理变长的标签

    Args:
        batch: 批次数据列表，每个元素是(data, labels)元组

    Returns:
        data_batch: 堆叠后的数据张量
        targets_dict: 包含以下键的字典:
            - bbox_targets: 3D边界框坐标，形状为 (batch_size, max_num_objects, 6)
            - cls_targets: 类别标签，形状为 (batch_size, max_num_objects)
            - mask: 有效目标掩码，形状为 (batch_size, max_num_objects)
    """
    # 分离数据和标签
    data_list = []
    labels_list = []

    for data, labels in batch:
        data_list.append(data)
        labels_list.append(labels)

    # 计算批次中最大目标数
    num_objects_per_sample = [len(labels) for labels in labels_list]
    max_num_objects = max(num_objects_per_sample)
    batch_size = len(batch)

    # 使用torch.stack堆叠数据
    data_batch = torch.stack(data_list, dim=0)

    # 初始化输出张量，用-1填充
    # 注意：原始标签形状是 (M, 7)，其中7是[xmin, xmax, ymin, ymax, zmin, zmax, class]
    all_targets = torch.full((batch_size, max_num_objects, 7), -1.0, dtype=torch.float32)

    # 填充数据
    for i, labels in enumerate(labels_list):
        num_objects = len(labels)
        if num_objects > 0:
            all_targets[i, :num_objects, :] = labels

    # 分离边界框和类别
    # 前6个是边界框坐标，第7个是类别
    bbox_targets = all_targets[:, :, :6]  # (batch_size, max_num_objects, 6)
    cls_targets = all_targets[:, :, 6]    # (batch_size, max_num_objects)

    # 创建掩码：cls_targets中不为-1的位置就是真实目标
    mask = (cls_targets != -1).float()  # 转换为float类型，形状: (batch_size, max_num_objects)

    # 创建目标字典
    targets_dict = {
        'bbox_targets': bbox_targets,
        'cls_targets': cls_targets,
        'mask': mask
    }

    return data_batch, targets_dict


class PointCloudDataset(Dataset):
    """
    点云数据集类

    支持两种模式：
    1. grid模式：将点云体素化为64x64x200的网格
    2. ds模式：使用降采样函数将点云降采样到固定点数

    数据组织方式：
    - 点云文件：单个文件夹中的所有.txt文件（xyzi格式）
    - 标签文件：与点云文件同名的.json文件
    """

    def __init__(self,
                 data_dir: str,
                 mode: str = 'grid',
                 num_points: int = 1024,
                 grid_size: Tuple[int, int, int] = (64, 64, 200)):
        """
        初始化数据集

        Args:
            data_dir: 数据目录路径，包含点云文件(.txt)和对应的标签文件(.json)
            mode: 数据处理模式，'grid'或'ds'
            num_points: ds模式下目标点数
            grid_size: grid模式下网格大小 (height, width, depth)
        """
        self.data_dir = data_dir
        self.mode = mode
        self.num_points = num_points
        self.grid_size = grid_size

        # 验证模式参数
        assert mode in ['grid', 'ds'], f"模式必须是'grid'或'ds'，当前为: {mode}"

        # 获取所有点云文件
        self.pc_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.txt')])

        # 检测标签文件与点云文件数量对应关系
        json_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])

        # 检查点云文件和标签文件数量是否一致
        if len(self.pc_files) != len(json_files):
            # 找到缺失的标签文件或点云文件
            pc_base_names = {f.replace('.txt', '') for f in self.pc_files}
            json_base_names = {f.replace('.json', '') for f in json_files}

            missing_json = pc_base_names - json_base_names
            missing_pc = json_base_names - pc_base_names

            error_msg = "点云文件与标签文件数量不匹配！\n"
            if missing_json:
                error_msg += f"缺少以下点云文件对应的标签文件: {sorted(missing_json)}\n"
            if missing_pc:
                error_msg += f"缺少以下标签文件对应的点云文件: {sorted(missing_pc)}\n"

            print(error_msg)
            raise ValueError("点云文件与标签文件数量不匹配")

        # 获取所有标签并建立类别映射
        self.labels = []  # 每个样本的标签列表，每个标签是一个(M, 7)的数组
        self.all_annotations = []  # 存储原始注释信息
        self.unique_classes = set()

        # 处理每个点云文件的标签
        for pc_file in self.pc_files:
            # 对应的标签文件
            json_file = pc_file.replace('.txt', '.json')
            json_path = os.path.join(data_dir, json_file)

            # 读取标签文件
            with open(json_path, 'r') as f:
                annotation_data = json.load(f)

            # 存储原始注释
            self.all_annotations.append(annotation_data)

            # 提取所有目标的标签
            sample_labels = []
            annotations = annotation_data.get('annotations', [])

            for ann in annotations:
                # 提取边界框信息
                x_range = ann['x_range']
                y_range = ann['y_range']
                z_range = ann['z_range']
                label = ann['label']

                # 收集唯一类别
                self.unique_classes.add(label)

                # 将边界框信息存储为[xmin, xmax, ymin, ymax, zmin, zmax, class]
                # 注意：class暂时用字符串，后面会映射为数字
                bbox_info = [
                    float(x_range[0]), float(x_range[1]),
                    float(y_range[0]), float(y_range[1]),
                    float(z_range[0]), float(z_range[1]),
                    label  # 字符串，后面会映射
                ]
                sample_labels.append(bbox_info)

            self.labels.append(sample_labels)

        # 对唯一类别进行排序并创建映射
        self.unique_classes = sorted(list(self.unique_classes))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        # 将标签中的类别字符串映射为数字
        for i, sample_labels in enumerate(self.labels):
            for j, bbox_info in enumerate(sample_labels):
                class_str = bbox_info[6]
                class_idx = self.class_to_idx[class_str]
                sample_labels[j][6] = float(class_idx)

        print(f"数据集初始化完成:")
        print(f"  点云文件数量: {len(self.pc_files)}")
        print(f"  标签文件数量: {len(json_files)}")
        print(f"  类别数量: {len(self.unique_classes)}")
        print(f"  类别映射: {self.class_to_idx}")
        print(f"  处理模式: {mode}")
        if mode == 'grid':
            print(f"  网格大小: {grid_size}")
        else:
            print(f"  目标点数: {num_points}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.pc_files)

    def _load_point_cloud(self, file_path: str) -> np.ndarray:
        """
        加载点云数据

        Args:
            file_path: 点云文件路径

        Returns:
            点云数据，形状为 (N, 4)，列顺序为 [x, y, z, intensity]
        """
        # 加载点云数据（假设格式为空格或逗号分隔）
        try:
            data = np.loadtxt(file_path, delimiter=',')
        except:
            # 如果逗号分隔失败，尝试空格分隔
            data = np.loadtxt(file_path)

        # 确保数据有4列（xyzi）
        if data.shape[1] != 4:
            raise ValueError(f"点云数据应该有4列(xyzi)，但只有{data.shape[1]}列")

        return data

    def _points_to_grid(self, points: np.ndarray) -> torch.Tensor:
        """
        将点云转换为网格表示

        Args:
            points: 点云数据，形状为 (N, 4)，列顺序为 [x, y, z, i]

        Returns:
            网格数据，形状为 (1, H, W, D)
        """
        # 创建零网格
        H, W, D = self.grid_size
        grid = np.zeros((H, W, D), dtype=np.float32)

        # 将强度值放入网格
        for point in points:
            x, y, z, intensity = point
            grid[int(x)-1, int(y)-1, int(z)-1] = intensity  # 转换为0-based索引
        # 64*64*200的tensor
        # 转换为torch张量并添加通道维度
        grid_tensor = torch.from_numpy(grid).unsqueeze(0)
        print('grid_tensor.shape', grid_tensor.shape)
        return grid_tensor

    def _points_to_downsampled(self, points: np.ndarray) -> torch.Tensor:
        """
        将点云降采样并转换为标准格式

        Args:
            points: 点云数据，形状为 (N, 4)，列顺序为 [x, y, z, i]

        Returns:
            降采样后的点云，形状为 (4, num_points)
        """
        # 使用降采样函数
        downsampled = downsample(points, self.num_points)

        # 转置为 (4, num_points)
        downsampled = downsampled.T  # 形状从 (num_points, 4) 变为 (4, num_points)

        # 转换为torch张量
        return torch.from_numpy(downsampled.astype(np.float32))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个数据样本

        Args:
            idx: 数据索引

        Returns:
            (data, label): 数据和标签的元组
        """
        # 获取点云文件路径
        pc_file = self.pc_files[idx]
        pc_path = os.path.join(self.data_dir, pc_file)

        # 加载点云数据
        points = self._load_point_cloud(pc_path)

        # 根据模式处理点云数据
        if self.mode == 'grid':
            data = self._points_to_grid(points)
        else:  # mode == 'ds'
            data = self._points_to_downsampled(points)

        # 获取标签并转换为张量
        sample_labels = self.labels[idx]

        if len(sample_labels) > 0:
            # 将标签列表转换为numpy数组
            labels_array = np.array(sample_labels, dtype=np.float32)
        else:
            # 如果没有目标，创建空的标签数组
            labels_array = np.zeros((0, 7), dtype=np.float32)

        # 转换为torch张量
        label_tensor = torch.from_numpy(labels_array)

        return data, label_tensor


def create_dataloaders(data_dir: str,
                       mode: str = 'grid',
                       batch_size: int = 32,
                       test_size: float = 0.2,
                       val_size: float = 0.2,
                       num_points: int = 1024,
                       grid_size: Tuple[int, int, int] = (64, 64, 200),
                       seed: int = SEED) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    创建训练、验证和测试集的DataLoader

    Args:
        data_dir: 数据目录路径
        mode: 数据处理模式，'grid'或'ds'
        batch_size: 批量大小
        test_size: 测试集比例
        val_size: 验证集比例（占训练集的比例）
        num_points: ds模式下目标点数
        grid_size: grid模式下网格大小
        seed: 随机种子

    Returns:
        train_loader, val_loader, test_loader, class_to_idx
    """

    def worker_init_fn(worker_id: int) -> None:
        """DataLoader worker初始化函数，设置随机种子"""
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    # 创建完整数据集
    full_dataset = PointCloudDataset(
        data_dir=data_dir,
        mode=mode,
        num_points=num_points,
        grid_size=grid_size
    )

    # 获取所有样本的索引
    indices = list(range(len(full_dataset)))

    # 获取有标签的样本索引（所有样本都应该有标签，但可能有空标签）
    labeled_indices = [i for i, labels in enumerate(full_dataset.labels) if labels]

    # 检查是否有无标签的样本
    if len(labeled_indices) != len(indices):
        print(f"警告: 有 {len(indices) - len(labeled_indices)} 个样本没有标签或标签为空")

    if not labeled_indices:
        raise ValueError("数据集中没有找到有效的标签")

    # 检查是否有足够的样本进行分割
    if len(labeled_indices) < 10:
        print(f"警告: 样本数量较少 ({len(labeled_indices)})，使用简单分割")
        # 使用简单分割
        train_indices = labeled_indices[:int(len(labeled_indices) * 0.6)]
        val_indices = labeled_indices[int(len(labeled_indices) * 0.6):int(len(labeled_indices) * 0.8)]
        test_indices = labeled_indices[int(len(labeled_indices) * 0.8):]
    else:
        # 第一次划分：分离测试集
        train_val_indices, test_indices = train_test_split(
            labeled_indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True
        )

        # 第二次划分：分离验证集
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size,
            random_state=seed,
            shuffle=True
        )

    print(f"\n数据集分割:")
    print(f"  训练集样本数: {len(train_indices)}")
    print(f"  验证集样本数: {len(val_indices)}")
    print(f"  测试集样本数: {len(test_indices)}")

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # 创建DataLoader，使用自定义的collate_fn函数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, full_dataset.class_to_idx


def get_data_shape(mode: str) -> Tuple[int, ...]:
    """
    获取数据形状

    Args:
        mode: 数据处理模式

    Returns:
        数据形状元组
    """
    if mode == 'grid':
        return (1, 64, 64, 200)  # (channel, height, width, depth)
    else:  # mode == 'ds'
        return (4, 1024)  # (features, num_points)


# ============================================
# Classification Data Pipeline (for train/test scripts)
# ============================================
def _is_point_file(file_name: str) -> bool:
    return file_name.lower().endswith((".txt", ".npy"))


def _collect_point_files(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, file_names in os.walk(folder):
        for file_name in file_names:
            if _is_point_file(file_name):
                files.append(os.path.join(root, file_name))
    files.sort()
    return files


def load_point_cloud_auto(file_path: str) -> np.ndarray:
    """
    加载单个点云文件，支持 txt/npy。
    返回形状统一为 (N, 4)，列顺序为 (x, y, z, i)。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        try:
            data = np.loadtxt(file_path, delimiter=',')
        except Exception:
            data = np.loadtxt(file_path)
    elif ext == '.npy':
        data = np.load(file_path)
    else:
        raise ValueError(f"Unsupported point cloud file format: {ext}")

    arr = np.asarray(data)

    if arr.ndim == 1:
        if arr.size == 0:
            return np.zeros((0, 4), dtype=np.float32)
        if arr.size % 4 != 0:
            raise ValueError(f"Invalid 1D point cloud size in {file_path}: {arr.size}")
        arr = arr.reshape(-1, 4)
    elif arr.ndim == 2:
        if arr.shape[1] == 4:
            pass
        elif arr.shape[0] == 4:
            arr = arr.T
        else:
            raise ValueError(f"Point cloud shape must be (N,4) or (4,N), got {arr.shape} in {file_path}")
    else:
        raise ValueError(f"Point cloud dimensions must be 1D/2D, got {arr.ndim}D in {file_path}")

    return arr.astype(np.float32, copy=False)


def fix_point_count(points: np.ndarray, num_points: int, seed: Optional[int] = None) -> np.ndarray:
    """将点云统一到固定点数 (num_points, 4)。"""
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError(f"points shape must be (N,4), got {points.shape}")

    n = points.shape[0]
    if n == num_points:
        return points

    rng = np.random.default_rng(seed)
    if n > num_points:
        indices = rng.choice(n, size=num_points, replace=False)
        return points[indices]

    if n == 0:
        return np.zeros((num_points, 4), dtype=np.float32)

    pad_indices = rng.choice(n, size=(num_points - n), replace=True)
    out = np.concatenate([points, points[pad_indices]], axis=0)
    rng.shuffle(out)
    return out


def discover_spad_classification_samples(data_root: str) -> Tuple[List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]]]:
    """
    自动扫描 SPAD 数据目录。

    规则:
    1) 若顶层数据集目录下存在子目录，则子目录名视为类别名（例如 2025-04-30-dpc）。
    2) 若顶层数据集目录下只有点云文件（无子目录），则视为无标签数据（例如 0917-pc）。

    Returns:
        labeled_samples, unlabeled_samples
        每个样本结构: {"path": str, "label": Optional[str], "dataset": str}
    """
    root = Path(data_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")

    labeled_samples: List[Dict[str, Optional[str]]] = []
    unlabeled_samples: List[Dict[str, Optional[str]]] = []

    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        child_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

        if child_dirs:
            # 子目录名即类别
            for class_dir in child_dirs:
                point_files = _collect_point_files(str(class_dir))
                for path in point_files:
                    labeled_samples.append({
                        "path": path,
                        "label": class_dir.name,
                        "dataset": dataset_dir.name,
                    })
        else:
            # 无子目录，视为无标签数据
            point_files = _collect_point_files(str(dataset_dir))
            for path in point_files:
                unlabeled_samples.append({
                    "path": path,
                    "label": None,
                    "dataset": dataset_dir.name,
                })

    return labeled_samples, unlabeled_samples


def build_class_mapping(samples: List[Dict[str, Optional[str]]]) -> Dict[str, int]:
    labels = sorted({sample["label"] for sample in samples if sample["label"] is not None})
    return {label: idx for idx, label in enumerate(labels)}


def _stratify_or_none(labels: List[str]) -> Optional[List[str]]:
    unique_labels = sorted(set(labels))
    if len(unique_labels) <= 1:
        return None

    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    if min(counts.values()) < 2:
        return None
    return labels


def split_labeled_samples(
    labeled_samples: List[Dict[str, Optional[str]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]]]:
    """将有标签样本划分为 train/val/test。"""
    if not labeled_samples:
        raise ValueError("No labeled samples found for splitting.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("train/val/test ratios must be positive.")

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    if len(labeled_samples) < 3:
        raise ValueError("Need at least 3 labeled samples to create train/val/test splits.")

    labels = [str(sample["label"]) for sample in labeled_samples]
    indices = np.arange(len(labeled_samples))

    try:
        stratify_1 = _stratify_or_none(labels)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_1,
        )

        val_in_train_val = val_ratio / (train_ratio + val_ratio)
        train_val_labels = [labels[i] for i in train_val_idx]
        stratify_2 = _stratify_or_none(train_val_labels)

        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_in_train_val,
            random_state=seed,
            shuffle=True,
            stratify=stratify_2,
        )
    except Exception:
        # 小样本或分层失败时，使用随机回退
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(round(n_total * train_ratio)))
        n_val = max(1, int(round(n_total * val_ratio)))
        n_test = n_total - n_train - n_val

        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1

        train_idx = shuffled[:n_train]
        val_idx = shuffled[n_train:n_train + n_val]
        test_idx = shuffled[n_train + n_val:]

    train_samples = [labeled_samples[int(i)] for i in train_idx]
    val_samples = [labeled_samples[int(i)] for i in val_idx]
    test_samples = [labeled_samples[int(i)] for i in test_idx]

    return train_samples, val_samples, test_samples


class SPADClassificationDataset(Dataset):
    """用于点云分类任务的数据集，返回固定形状点云 (N,4) 与分类标签。"""

    def __init__(
        self,
        samples: List[Dict[str, Optional[str]]],
        class_to_idx: Dict[str, int],
        num_points: int = 1024,
        seed: int = SEED,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.num_points = num_points
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[idx]
        points = load_point_cloud_auto(sample["path"])
        points = fix_point_count(points, self.num_points, seed=self.seed + idx)

        label = sample["label"]
        label_idx = -1 if label is None else self.class_to_idx[str(label)]

        points_tensor = torch.from_numpy(points.astype(np.float32))  # (N,4)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return points_tensor, label_tensor, str(sample["path"])


def create_spad_classification_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_points: int = 1024,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader], Dict]:
    """
    创建 SPAD 分类任务的 DataLoader。

    Returns:
        train_loader, val_loader, test_loader, unlabeled_loader, meta
    """
    labeled_samples, unlabeled_samples = discover_spad_classification_samples(data_root)
    if not labeled_samples:
        raise ValueError(f"No labeled samples found in data root: {data_root}")

    class_to_idx = build_class_mapping(labeled_samples)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    train_samples, val_samples, test_samples = split_labeled_samples(
        labeled_samples=labeled_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_dataset = SPADClassificationDataset(train_samples, class_to_idx, num_points=num_points, seed=seed)
    val_dataset = SPADClassificationDataset(val_samples, class_to_idx, num_points=num_points, seed=seed)
    test_dataset = SPADClassificationDataset(test_samples, class_to_idx, num_points=num_points, seed=seed)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    unlabeled_loader: Optional[DataLoader] = None
    if unlabeled_samples:
        unlabeled_dataset = SPADClassificationDataset(unlabeled_samples, class_to_idx, num_points=num_points, seed=seed)
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    meta = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(class_to_idx),
        "num_labeled_samples": len(labeled_samples),
        "num_unlabeled_samples": len(unlabeled_samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
    }

    return train_loader, val_loader, test_loader, unlabeled_loader, meta


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 数据目录路径
    data_dir = r"E:\essay\硕士\研一\SPAD数据\20250430dataset\2025-04-30-dpc"

    # 测试grid模式
    print("=" * 60)
    print("测试grid模式:")
    print("=" * 60)

    try:
        train_loader_grid, val_loader_grid, test_loader_grid, class_to_idx_grid = create_dataloaders(
            data_dir=data_dir,
            mode='grid',
            batch_size=4,
            test_size=0.2,
            val_size=0.2,
            seed=SEED
        )

        # 打印类别映射
        print("\n类别映射:")
        for cls, idx in class_to_idx_grid.items():
            print(f"  {cls} -> {idx}")

        # 检查数据集大小
        print(f"\n数据集大小:")
        print(f"  训练集: {len(train_loader_grid.dataset)} 样本")
        print(f"  验证集: {len(val_loader_grid.dataset)} 样本")
        print(f"  测试集: {len(test_loader_grid.dataset)} 样本")

        # 检查一个批次的数据
        for batch_idx, (data_batch, targets_dict) in enumerate(train_loader_grid):
            print(f"\nBatch {batch_idx + 1} (grid模式):")
            print(f"  数据形状: {data_batch.shape}")  # 应为 (batch_size, 1, 64, 64, 200)
            print(f"  边界框目标形状: {targets_dict['bbox_targets'].shape}")  # (batch_size, max_num_objects, 6)
            print(f"  类别目标形状: {targets_dict['cls_targets'].shape}")  # (batch_size, max_num_objects)
            print(f"  掩码形状: {targets_dict['mask'].shape}")  # (batch_size, max_num_objects)
            print(f"  掩码数据类型: {targets_dict['mask'].dtype}")

            # 显示掩码信息
            mask = targets_dict['mask']
            print(f"  掩码值示例（第一个样本）: {mask[0].numpy()}")

            # 显示第一个样本的详细信息
            bbox_targets = targets_dict['bbox_targets'][0]
            cls_targets = targets_dict['cls_targets'][0]
            mask_sample = targets_dict['mask'][0]

            num_valid = int(mask_sample.sum().item())
            print(f"\n  第一个样本 - 有效目标数: {num_valid}")
            print(f"  总目标位置数: {len(mask_sample)}")

            for i in range(min(5, mask_sample.shape[0])):
                if mask_sample[i] > 0.5:  # 使用>0.5判断，因为mask是float
                    bbox = bbox_targets[i]
                    cls_idx = int(cls_targets[i].item())
                    cls_name = class_to_idx_grid.get(cls_idx, "未知")
                    print(f"    目标{i}: 类别={cls_name}({cls_idx}), "
                          f"边界框=[{bbox[0]:.1f}, {bbox[1]:.1f}] x "
                          f"[{bbox[2]:.1f}, {bbox[3]:.1f}] x "
                          f"[{bbox[4]:.1f}, {bbox[5]:.1f}]")
                else:
                    print(f"    目标{i}: 填充位置")

            # 检查填充值
            print(f"\n  填充值检查:")
            print(f"    bbox_targets填充值: {targets_dict['bbox_targets'][0, num_valid:, :] if num_valid < bbox_targets.shape[0] else '无填充'}")
            print(f"    cls_targets填充值: {targets_dict['cls_targets'][0, num_valid:] if num_valid < cls_targets.shape[0] else '无填充'}")

            # 只检查第一个批次
            if batch_idx == 0:
                break

    except Exception as e:
        print(f"创建grid模式DataLoader时出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试完成!")

