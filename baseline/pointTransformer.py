"""
Point Transformer V1 for Object Classification + 3D BBox

GitHub:  https://github.com/Pointcept/Pointcept
Local:   D:\essay\3d目标检测复现仓库\Pointcept-main

Pointcept 官方实现复现 (Xiaoyang Wu)
- 5 阶段编码器 + 全局池化 + 分类/框回归头
- 使用 PointTransformerLayer (向量注意力) + TransitionDown (FPS+kNN) + Bottleneck
- 输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])

Reference:
@inproceedings{zhao2021point,
  title={Point transformer},
  author={Zhao, Hengshuang and Jiang, Li and Jia, Jiaya and Torr, Philip HS and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={16259--16268},
  year={2021}
}
"""

import os
import sys
# 确保项目根目录在 path 中, 以便 import utils
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.pointnet_utils import (
    LayerNorm1d, farthest_point_sample, index_points, knn_point, square_distance
)


# ============================================================================
# Point Transformer Layer (ICCV 2021 向量注意力)
# ============================================================================

class PointTransformerLayer(nn.Module):
    """
    Point Transformer 层:
    x'_i = sum_j softmax(gamma(phi(x_i)-psi(x_j)+delta_ij)) * (alpha(x_j)+delta_ij)

    遵循 Pointcept 官方实现:
    - in_planes, out_planes, share_planes=8, nsample=16
    - 3D 位置编码 (仅 xyz)
    """
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        """
        Args:
            pxo: [p, x, o] = [(N, 3), (N, C), (B,)]
            变长点云格式: N = 总点数, B = batch size
        Returns:
            x: (N, C) 更新后的特征
        """
        p, x, o = pxo  # (N, 3), (N, C), (B,)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)

        # kNN 搜索并在自身位置上分组
        idx = knn_point_batched(self.nsample, p, o, p, o)
        x_k_grouped = group_points(x_k, idx)       # (N, nsample, C')
        x_v_grouped = group_points(x_v, idx)       # (N, nsample, C_out)

        # 相对位置编码 (3D xyz)
        p_grouped = group_points(p, idx)            # (N, nsample, 3)
        p_r = p_grouped - p.unsqueeze(1)            # 相对位置
        p_r = self.linear_p(p_r)                     # (N, nsample, C_out)

        # 注意力权重
        r_qk = x_k_grouped - x_q.unsqueeze(1) + einops_reduce_sum(
            p_r, self.mid_planes
        )  # (N, nsample, C')
        w = self.linear_w(r_qk)                     # (N, nsample, C_out//share)
        w = self.softmax(w)                         # 在 nsample 维度上 softmax

        # 加权聚合 (分组向量注意力)
        x_v_plus_p = x_v_grouped + p_r              # (N, nsample, C_out)
        x_v_plus_p = x_v_plus_p.reshape(
            -1, self.nsample, self.share_planes, self.out_planes // self.share_planes
        )  # (N, nsample, share, C'//share)
        w = w.unsqueeze(-1)                         # (N, nsample, C'//share, 1)
        x = torch.sum(x_v_plus_p * w, dim=1)        # (N, share, C'//share)
        x = x.reshape(-1, self.out_planes)          # (N, C_out)
        return x


def einops_reduce_sum(x, mid_planes):
    """替代 einops.reduce(x, 'n ns (i j) -> n ns j', reduction='sum', j=mid_planes)"""
    N, nsample, C = x.shape
    assert C % mid_planes == 0
    x = x.reshape(N, nsample, C // mid_planes, mid_planes)
    return x.sum(dim=2)


def knn_point_batched(k, xyz, offset, new_xyz, new_offset):
    """
    变长点云上的 kNN 搜索。
    xyz: (N, 3), offset: (B,)
    new_xyz: (M, 3), new_offset: (B,)
    Returns: idx (M, k)
    """
    device = xyz.device
    M = new_xyz.shape[0]
    idx = torch.zeros(M, k, dtype=torch.long, device=device)
    for i in range(len(offset)):
        s_i = 0 if i == 0 else offset[i - 1].item()
        e_i = offset[i].item()
        s_new = 0 if i == 0 else new_offset[i - 1].item()
        e_new = new_offset[i].item()
        if e_i - s_i <= 0 or e_new - s_new <= 0:
            continue
        # 计算距离
        dist = torch.cdist(new_xyz[s_new:e_new], xyz[s_i:e_i])  # (n_new, n)
        # 取 k 个最近邻
        if k > dist.shape[1]:
            k_actual = dist.shape[1]
            topk_idx = dist.topk(k=k_actual, dim=-1, largest=False)[1]
            # pad
            pad = topk_idx[:, :1].repeat(1, k - k_actual)
            idx[s_new:e_new] = torch.cat([topk_idx, pad], dim=1) + s_i
        else:
            idx[s_new:e_new] = dist.topk(k=k, dim=-1, largest=False)[1] + s_i
    return idx


def group_points(feat, idx):
    """
    根据索引分组特征。
    feat: (N, C), idx: (N, K)
    Returns: (N, K, C)
    """
    N, K = idx.shape
    C = feat.shape[1]
    # flatten idx: (N*K) → gather along dim=0 → reshape back
    idx_flat = idx.reshape(-1)  # (N*K,)
    idx_flat = idx_flat.unsqueeze(-1).expand(-1, C)  # (N*K, C)
    grouped = feat.gather(0, idx_flat)  # (N*K, C)
    return grouped.reshape(N, K, C)


# ============================================================================
# Transition Down (FPS + kNN 分组 + MLP + 最大池化)
# ============================================================================

class TransitionDown(nn.Module):
    """
    Point Transformer 下采样模块:
    - FPS 采样关键点
    - kNN 分组邻居
    - MLP + 最大池化
    """
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            # 下采样: 需要 FPS + kNN 分组
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """
        Args:
            pxo: [p, x, o] = [(N, 3), (N, C), (B,)]
        Returns:
            [p_new, x_new, o_new]
        """
        p, x, o = pxo  # (N, 3), (N, C), (B,)
        if self.stride != 1:
            # 计算新的 offset (每个样本下采样到 N/stride)
            n_o_list = []
            for i in range(len(o)):
                if i == 0:
                    n_points = o[0].item() // self.stride
                else:
                    n_points = (o[i].item() - o[i - 1].item()) // self.stride
                n_points = max(n_points, 1)  # 至少保留 1 个点
                prev = 0 if i == 0 else n_o_list[-1]
                n_o_list.append(prev + n_points)
            n_o = torch.tensor(n_o_list, dtype=torch.long, device=o.device)

            # FPS 采样
            fps_idx_list = []
            for i in range(len(o)):
                s_i = 0 if i == 0 else o[i - 1].item()
                e_i = o[i].item()
                s_new = 0 if i == 0 else n_o[i - 1].item()
                e_new = n_o[i].item()
                npoint = e_new - s_new
                if npoint <= 0 or e_i - s_i <= 0:
                    continue
                batch_xyz = p[s_i:e_i].unsqueeze(0)  # (1, N, 3)
                fps_idx = farthest_point_sample(batch_xyz, npoint)
                fps_idx_list.append(fps_idx[0] + s_i)

            if len(fps_idx_list) > 0:
                all_fps_idx = torch.cat(fps_idx_list)
            else:
                all_fps_idx = torch.arange(0, dtype=torch.long, device=o.device)

            n_p = p[all_fps_idx.long(), :]  # (M, 3)

            # kNN 分组
            idx = knn_point_batched(self.nsample, p, o, n_p, n_o)
            # 分组特征: concat(xyz, feat)
            x_cat = torch.cat([p, x], dim=-1)  # (N, 3+C)
            grouped = group_points(x_cat, idx)  # (M, nsample, 3+C)
            x = self.relu(
                self.bn(self.linear(grouped).transpose(1, 2).contiguous())
            )  # (M, C_out, nsample)
            x = self.pool(x).squeeze(-1)  # (M, C_out)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (N, C)
        return [p, x, o]


# ============================================================================
# Bottleneck (Point Transformer 残差块)
# ============================================================================

class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (N, 3), (N, C), (B,)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


# ============================================================================
# Point Transformer Classification + BBox
# ============================================================================

class PointTransformerCls(nn.Module):
    """
    Point Transformer V1 分类模型 (Pointcept 架构)

    5 阶段编码器:
    - stride = [1, 4, 4, 4, 4] → 下采样率: 1x, 4x, 16x, 64x, 256x
    - nsample = [8, 16, 16, 16, 16]
    - planes = [32, 64, 128, 256, 512]

    输入格式: 使用 data_dict {"coord": (N,3), "feat": (N,C), "offset": (B,)}
    输出: 分类 logits + 3D bbox 预测
    """
    def __init__(self, block, blocks, in_channels=6, num_classes=40, box_dim=6):
        super().__init__()
        self.in_channels = in_channels
        self.in_planes = in_channels
        planes = [32, 64, 128, 256, 512]
        share_planes = 8
        stride = [1, 4, 4, 4, 4]
        nsample = [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes,
                                    stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes,
                                    stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes,
                                    stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes,
                                    stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes,
                                    stride=stride[4], nsample=nsample[4])  # N/256

        self.cls_head = nn.Sequential(
            nn.Linear(planes[4], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

        self.box_head = nn.Sequential(
            nn.Linear(planes[4], 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, box_dim),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _batched_forward(self, p, x, o):
        """执行 5 阶段编码器前向"""
        p1, x1, o1 = self.enc1([p, x, o])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        return x5, o5

    def forward(self, data_dict):
        """
        Args:
            data_dict: {
                "coord": (N, 3), "feat": (N, C), "offset": (B,)
            }
        Returns:
            logits: (B, num_classes)
            box_pred: (B, box_dim)
        """
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        # 如果 in_channels==3, 只使用坐标; 否则拼接坐标+特征
        x0 = p0 if self.in_channels == 3 else torch.cat((p0, x0), 1)

        x5, o5 = self._batched_forward(p0, x0, o0)

        # 全局平均池化 (按样本)
        x_list = []
        for i in range(o5.shape[0]):
            s_i = 0 if i == 0 else o5[i - 1].item()
            e_i = o5[i].item()
            cnt = e_i - s_i
            x_b = x5[s_i:e_i, :].sum(0, True) / cnt
            x_list.append(x_b)
        x = torch.cat(x_list, 0)  # (B, C)

        logits = self.cls_head(x)
        box_pred = self.box_head(x)
        return logits, box_pred


# ============================================================================
# 具体模型变体 (适配 SPAD 训练管道)
# ============================================================================

class PointTransformerClassification(PointTransformerCls):
    """
    PointTransformerClassification — 适配 SPAD 训练管道

    输入: (B, N, 4) xyzi 点云
    输出: (logits [B, num_classes], box_pred [B, 6])
    """
    def __init__(self, num_classes=26, block_config=(1, 1, 1, 1, 1), in_channels=4, box_dim=6):
        super().__init__(
            Bottleneck, list(block_config),
            in_channels=in_channels, num_classes=num_classes, box_dim=box_dim
        )

    @staticmethod
    def _normalize_input_points(x):
        """统一输入格式为 (B, N, 4)"""
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")
        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported shape {tuple(x.shape)}")
        if points.shape[-1] == 3:
            pad_i = torch.zeros(points.shape[0], points.shape[1], 1,
                                dtype=points.dtype, device=points.device)
            points = torch.cat([points, pad_i], dim=-1)
        return points

    def forward(self, x):
        """
        Args:
            x: (B, N, 4) xyzi 点云
        Returns:
            logits: (B, num_classes)
            box_pred: (B, 6)
        """
        x = self._normalize_input_points(x)
        B, N, C_in = x.shape

        # 构建 data_dict
        coord = x[:, :, :3].reshape(B * N, 3).contiguous()
        feat = x[:, :, 3:].reshape(B * N, -1).contiguous()
        offset = torch.arange(N, (B + 1) * N, step=N, dtype=torch.long, device=x.device)

        data_dict = {"coord": coord, "feat": feat, "offset": offset}
        return super().forward(data_dict)


# ============================================================================
# 不同大小的模型变体
# ============================================================================

def point_transformer_cls26(num_classes=26, **kwargs):
    """PointTransformer-Cls26: blocks=[1,1,1,1,1]"""
    return PointTransformerClassification(num_classes=num_classes, block_config=(1, 1, 1, 1, 1), **kwargs)


def point_transformer_cls38(num_classes=26, **kwargs):
    """PointTransformer-Cls38: blocks=[1,2,2,2,2]"""
    return PointTransformerClassification(num_classes=num_classes, block_config=(1, 2, 2, 2, 2), **kwargs)


def point_transformer_cls50(num_classes=26, **kwargs):
    """PointTransformer-Cls50: blocks=[1,2,3,5,2]"""
    return PointTransformerClassification(num_classes=num_classes, block_config=(1, 2, 3, 5, 2), **kwargs)


# ============================================================================
# 快速测试 + GPU 显存测试
# ============================================================================

def _quick_test():
    """形状验证。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing PointTransformer V1 on {device}")

    model = PointTransformerClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Box:    {tuple(box_pred.shape)}")
    print("✓ PointTransformer V1 works correctly")


def _gpu_memory_test():
    """GPU 显存压力测试 (逐 batch size 扫查)。"""
    import gc
    if not torch.cuda.is_available():
        print("无 CUDA，跳过 GPU 显存测试。")
        return

    print("\n=== GPU 显存测试 ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        props = torch.cuda.get_device_properties(0)
        total_mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        if total_mem:
            print(f"总显存: {total_mem / 1024**3:.1f} GB")
    except Exception:
        pass
    print()

    N = 1024
    for bs in [4, 8, 16, 32]:
        try:
            m = PointTransformerClassification(num_classes=26).cuda()
            pts = torch.randn(bs, N, 4).cuda()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            m.train()
            o = m(pts)
            loss = o[0].sum() + o[1].sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  B={bs:2d}: peak {peak:6.0f} MB")
            del m, pts, o, loss
            torch.cuda.empty_cache()
            gc.collect()
        except torch.cuda.OutOfMemoryError:
            print(f"  B={bs:2d}: OOM!")
            torch.cuda.empty_cache()
            gc.collect()
            break


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()