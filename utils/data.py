import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path
from utils.data_augment import augment_pytorch_batch

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
        将点云转换为标准格式（不再做降采样）

        Args:
            points: 点云数据，形状为 (N, 4)，列顺序为 [x, y, z, i]

        Returns:
            点云，形状为 (4, N)
        """
        # 数据已在上游完成下采样，这里只做转置，不再重采样。
        downsampled = points.T  # 形状从 (N, 4) 变为 (4, N)

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
ALPHABET_CLASSES: List[str] = [chr(ord("A") + i) for i in range(26)]
ALPHABET_CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(ALPHABET_CLASSES)}


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


def _infer_symbol_from_text(text: str) -> Optional[str]:
    for ch in str(text).upper():
        if "A" <= ch <= "Z":
            return ch
    return None


def load_point_cloud_auto(file_path: str) -> np.ndarray:
    """
    加载单个点云文件，支持 txt/npy。
    返回形状统一为 (N, 4)，列顺序为 (x, y, z, i)。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        try:
            data = np.loadtxt(file_path, delimiter=",")
        except Exception:
            data = np.loadtxt(file_path)
    elif ext == ".npy":
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


def discover_spad_classification_samples(data_root: str) -> Tuple[List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]]]:
    """
    自动扫描 SPAD 数据目录。

    Returns:
        labeled_samples, unlabeled_samples
        样本结构: {"path": str, "label": Optional[str], "dataset": str}
    """
    root = Path(data_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")

    labeled_samples: List[Dict[str, Optional[str]]] = []
    unlabeled_samples: List[Dict[str, Optional[str]]] = []

    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        child_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

        if child_dirs:
            for class_dir in child_dirs:
                inferred_label = _infer_symbol_from_text(class_dir.name)
                for path in _collect_point_files(str(class_dir)):
                    sample = {
                        "path": path,
                        "label": inferred_label,
                        "dataset": dataset_dir.name,
                    }
                    if inferred_label is None:
                        unlabeled_samples.append(sample)
                    else:
                        labeled_samples.append(sample)
        else:
            for path in _collect_point_files(str(dataset_dir)):
                inferred_label = _infer_symbol_from_text(Path(path).stem)
                sample = {
                    "path": path,
                    "label": inferred_label,
                    "dataset": dataset_dir.name,
                }
                if inferred_label is None:
                    unlabeled_samples.append(sample)
                else:
                    labeled_samples.append(sample)

    root_files = [p for p in root.iterdir() if p.is_file() and _is_point_file(p.name)]
    for file_path in sorted(root_files):
        inferred_label = _infer_symbol_from_text(file_path.stem)
        sample = {
            "path": str(file_path),
            "label": inferred_label,
            "dataset": root.name,
        }
        if inferred_label is None:
            unlabeled_samples.append(sample)
        else:
            labeled_samples.append(sample)

    return labeled_samples, unlabeled_samples


def build_class_mapping(_: List[Dict[str, Optional[str]]]) -> Dict[str, int]:
    return dict(ALPHABET_CLASS_TO_IDX)


def split_samples_deterministic(
    samples: List[Dict[str, Optional[str]]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]], List[Dict[str, Optional[str]]]]:
    if len(samples) < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("train/val/test ratios must be positive.")

    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    n_total = len(samples)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = n_total - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Not enough samples to split into train/val/test.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_samples = [samples[int(i)] for i in train_idx]
    val_samples = [samples[int(i)] for i in val_idx]
    test_samples = [samples[int(i)] for i in test_idx]
    return train_samples, val_samples, test_samples


def _symbol_from_augmented_position(meta: Dict) -> str:
    tx = int(meta.get("target_x_range", [20, 35])[0])
    ty = int(meta.get("target_y_range", [5, 25])[0])
    tz = int(meta.get("target_z_range", [80, 84])[0])

    tx = int(np.clip(tx, 1, 50))
    ty = int(np.clip(ty, 1, 45))
    tz = int(np.clip(tz, 60, 110))

    idx = ((tx - 1) * 7 + (ty - 1) * 3 + (tz - 60)) % 26
    return ALPHABET_CLASSES[idx]


class SPADClassificationDataset(Dataset):
    """分类数据集：输入点云 (N,4)，标签固定为 A-Z 共 26 类。"""

    def __init__(
        self,
        samples: List[Dict[str, Optional[str]]],
        class_to_idx: Dict[str, int],
        num_points: Optional[int] = None,
        seed: int = SEED,
        apply_augment: bool = True,
        label_mode: str = "generated",
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.num_points = num_points
        self.seed = int(seed)
        self.apply_augment = apply_augment
        self.label_mode = label_mode

        if self.label_mode not in {"generated", "raw"}:
            raise ValueError(f"label_mode must be 'generated' or 'raw', got: {self.label_mode}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        sample = self.samples[idx]
        points = load_point_cloud_auto(str(sample["path"]))

        if self.num_points is not None and self.num_points > 0 and points.shape[0] != self.num_points:
            raise ValueError(
                f"Point count mismatch for {sample['path']}: got N={points.shape[0]}, expected N={self.num_points}. "
                "Downsampling is disabled by design. Please ensure preprocessed data has fixed N."
            )

        points = points.astype(np.float32, copy=False)

        if self.apply_augment:
            sample_seed = self.seed + idx * 9973
            points_tensor = torch.from_numpy(points).unsqueeze(0)
            aug_points, aug_meta_list = augment_pytorch_batch(points_tensor, label_class=None, seed=sample_seed)
            points = aug_points.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
            aug_meta = aug_meta_list[0] if aug_meta_list is not None else {}
        else:
            aug_meta = {
                "target_shift": [0, 0, 0],
                "fog_shift_z": 0,
                "target_x_range": [20, 35],
                "target_y_range": [5, 25],
                "target_z_range": [80, 84],
                "fog_z_range": [35, 64],
                "fog_ahead_gap_bins": 16,
            }

        raw_label = sample.get("label")
        raw_symbol = _infer_symbol_from_text(raw_label) if raw_label is not None else None
        generated_symbol = _symbol_from_augmented_position(aug_meta)

        if self.label_mode == "raw" and raw_symbol in self.class_to_idx:
            symbol = str(raw_symbol)
        else:
            symbol = generated_symbol

        label_idx = self.class_to_idx[symbol]

        points_tensor = torch.from_numpy(points)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        sample_meta = {
            "path": str(sample["path"]),
            "dataset": str(sample.get("dataset", "")),
            "sym": symbol,
            "target_x_new": list(aug_meta.get("target_x_range", [20, 35])),
            "target_y_new": list(aug_meta.get("target_y_range", [5, 25])),
            "target_z_new": list(aug_meta.get("target_z_range", [80, 84])),
            "target_shift": list(aug_meta.get("target_shift", [0, 0, 0])),
        }
        return points_tensor, label_tensor, sample_meta


def create_spad_classification_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_points: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    seed: int = SEED,
    augment_train: bool = True,
    augment_eval: bool = True,
    label_mode: str = "generated",
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DataLoader], Dict]:
    """
    创建 SPAD 分类任务 DataLoader。

    标签规则：
    - 默认 label_mode='generated'，根据增强后的 target_x_new/target_y_new/target_z_new 生成 A-Z 类别。
    - label_mode='raw' 时，优先使用样本可解析的原始字母标签，否则回退到 generated。
    """
    labeled_samples, unlabeled_samples = discover_spad_classification_samples(data_root)
    all_samples = labeled_samples + unlabeled_samples

    if len(all_samples) < 3:
        raise ValueError(f"Need at least 3 samples, got {len(all_samples)} from: {data_root}")

    class_to_idx = build_class_mapping(all_samples)
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    train_samples, val_samples, test_samples = split_samples_deterministic(
        samples=all_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_dataset = SPADClassificationDataset(
        train_samples,
        class_to_idx,
        num_points=num_points,
        seed=seed + 11,
        apply_augment=augment_train,
        label_mode=label_mode,
    )
    val_dataset = SPADClassificationDataset(
        val_samples,
        class_to_idx,
        num_points=num_points,
        seed=seed + 29,
        apply_augment=augment_eval,
        label_mode=label_mode,
    )
    test_dataset = SPADClassificationDataset(
        test_samples,
        class_to_idx,
        num_points=num_points,
        seed=seed + 47,
        apply_augment=augment_eval,
        label_mode=label_mode,
    )

    def _seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed + 101)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(seed + 202)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=eval_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=eval_generator,
    )

    meta = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(class_to_idx),
        "num_labeled_samples": len(labeled_samples),
        "num_unlabeled_samples": len(unlabeled_samples),
        "num_total_samples": len(all_samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "num_test_samples": len(test_samples),
        "seed": seed,
        "label_mode": label_mode,
        "augmentation_location": "dataset",
        "seed_controls_dataloader_and_augmentation": True,
    }

    return train_loader, val_loader, test_loader, None, meta

