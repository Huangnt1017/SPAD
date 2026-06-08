"""
DGCNN: Dynamic Graph CNN for Learning on Point Clouds

GitHub:  https://github.com/WangYueFt/dgcnn
Local:   D:\essay\3d目标检测复现仓库\dgcnn-master (baseline 参考)

@article{wang2019dynamic,
  title={Dynamic graph cnn for learning on point clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E and Bronstein, Michael M and Solomon, Justin M},
  journal={ACM Transactions on Graphics (tog)},
  volume={38},
  number={5},
  pages={1--12},
  year={2019},
  publisher={Acm New York, NY, USA}
}
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.heads import build_standard_cls_head, build_standard_box_head


def knn(x, k):
    """
    x: [B, C, N]
    return idx: [B, N, k]
    """
    # pairwise distance: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)  # [B, N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)        # [B, 1, N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # negative squared distance
    idx = pairwise_distance.topk(k=k, dim=-1)[1]       # largest -> nearest due to negative distance
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construct edge features for EdgeConv.
    x: [B, C, N]
    return: [B, 2C, N, k], where feature = concat(x_j - x_i, x_i)
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)  # [B, N, k]

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)  # flatten global index

    x_t = x.transpose(2, 1).contiguous()               # [B, N, C]
    feature = x_t.view(B * N, C)[idx, :]               # [B*N*k, C]
    feature = feature.view(B, N, k, C)                 # [B, N, k, C]
    x_i = x_t.view(B, N, 1, C).repeat(1, 1, k, 1)      # [B, N, k, C]

    feature = torch.cat((feature - x_i, x_i), dim=3)   # [B, N, k, 2C]
    return feature.permute(0, 3, 1, 2).contiguous()    # [B, 2C, N, k]


class DGCNNCls(nn.Module):
    """
    DGCNN for point cloud classification.
    Input:  x [B, N, 4]
    Output:
        logits [B, num_classes]
        box_pred [B, 3] -> 中心点 [cx, cy, cz] (center-only 回归)
    """
    def __init__(self, num_classes=26, k=20, emb_dims=1024, dropout=0.3):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims

        # backbone 卷积层
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 多尺度特征聚合: 64+64+128+256=512 → emb_dims
        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        # 全局池化: adaptive_max + adaptive_avg → (B, emb_dims*2)
        # pooled_dim = emb_dims * 2 = 2048

        # 统一分类头: 3 层 MLP (2048 → 256 → 128 → num_classes)
        self.cls_head = build_standard_cls_head(
            pooled_dim=emb_dims * 2,
            num_classes=num_classes,
            dropout=dropout,
        )

        # 统一中心点回归头: 3 层 MLP (2048 → 256 → 128 → 3)
        self.box_head = build_standard_box_head(
            pooled_dim=emb_dims * 2,
            box_dim=3,
            dropout=dropout,
        )

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 4:
            raise ValueError(f"DGCNNCls expects input shape (B, N, 4), got {tuple(x.shape)}")

        x = x.transpose(1, 2).contiguous()  # [B, 4, N]
        B = x.size(0)

        x = get_graph_feature(x, k=self.k)   # [B, 8, N, k]
        x = self.conv1(x)
        x1 = x.max(dim=-1)[0]                # [B, 64, N]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1)[0]                # [B, 64, N]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1)[0]                # [B, 128, N]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1)[0]                # [B, 256, N]

        x = torch.cat((x1, x2, x3, x4), dim=1)  # [B, 512, N]
        x = self.conv5(x)                        # [B, emb_dims, N]

        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        # (B, 2048) 全局特征
        f_pooled = torch.cat((x1, x2), dim=1)

        # 统一头直接处理池化特征
        logits = self.cls_head(f_pooled)
        box_pred = self.box_head(f_pooled)
        return logits, box_pred


def _quick_shape_test():
    """快速形状验证 + GPU 显存压力测试。"""
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_model = DGCNNCls(num_classes=26).to(device)
    pts = torch.randn(4, 1024, 4, device=device)
    cls_logits, box_pred = cls_model(pts)
    print("DGCNNCls logits:", cls_logits.shape)  # [4, 26]
    print("DGCNNCls box_pred:", box_pred.shape)  # [4, 3]

    # ══════════════════════════════════════════════
    # GPU 显存测试
    # ══════════════════════════════════════════════
    print("\n=== GPU 显存测试 ===")
    if torch.cuda.is_available():
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
                m = DGCNNCls(num_classes=26).cuda()
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
    else:
        print("无 CUDA，跳过。")


if __name__ == "__main__":
    _quick_shape_test()
