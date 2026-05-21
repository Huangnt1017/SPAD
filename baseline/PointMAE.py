"""
Point-MAE: Masked Autoencoders for Point Cloud Self-Supervised Learning

GitHub:  https://github.com/Pang-Yatian/Point-MAE
Local:   D:\\essay\\3d目标检测复现仓库\\Point-MAE-main

完整复现 Point-MAE 分类微调模型 (PointTransformer, finetune 变体, 按 SPAD 双头改造):
- PatchGroup: FPS + KNN 分组构建局部 patches  (utils.transformer_blocks)
- PatchEncoder: Mini-PointNet 编码 patches → patch tokens  (utils.transformer_blocks)
- TransformerEncoder: 标准 Transformer + cls_token + 位置编码  (utils.transformer_blocks)
- 分类头: cls_token + 全局最大池化拼接 → BN → MLP (Point-MAE 官方 finetune 设计)
- 输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])

Reference:
@inproceedings{pang2022masked,
  title={Masked autoencoders for point cloud self-supervised learning},
  author={Pang, Yatian and Wang, Wenxiao and Tay, Francis EH and Liu, Wei and Tian, Yonghong and Yuan, Li},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
"""

import os
import sys

import torch
import torch.nn as nn

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.transformer_blocks import (
    PatchGroup,
    PatchEncoder,
    TransformerEncoder,
    trunc_normal_,
)


# ============================================================================
# Point-MAE 分类微调模型
# ============================================================================

class PointMAEClassification(nn.Module):
    """Point-MAE 分类 + 3D BBox 模型 (适配 SPAD 训练管道)。

    对应官方 models/Point_MAE.py 中的 PointTransformer (finetune model)。
    与 Point-BERT 类似, 但分类头更宽 (双层 + BatchNorm)。

    架构:
    1. PatchGroup: FPS + KNN 分组构建局部 patches
    2. PatchEncoder: Mini-PointNet 编码为 patch tokens
    3. cls_token + 位置编码
    4. TransformerEncoder: 标准 Transformer
    5. 分类头: cls_token + 全局最大池化拼接 → BN → MLP

    输入: (B, N, 4) xyzi → 输出: (logits [B, C], box_pred [B, 6])
    """
    def __init__(self, num_classes: int = 26, trans_dim: int = 384,
                 depth: int = 6, num_heads: int = 6, drop_path_rate: float = 0.1,
                 group_size: int = 32, num_group: int = 64,
                 encoder_dims: int = 256, **kwargs):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims

        self.group_divider = PatchGroup(num_group=num_group, group_size=group_size)
        self.encoder = PatchEncoder(encoder_channel=encoder_dims)

        # 维度缩减: encoder_dims → trans_dim
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        # 位置编码 (MLP: 3 → 128 → trans_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        self.blocks = TransformerEncoder(
            embed_dim=trans_dim, depth=depth,
            drop_path_rate=drop_path_rate, num_heads=num_heads,
        )

        self.norm = nn.LayerNorm(trans_dim)

        # 分类头 (Point-MAE 官方 finetune 设计: 双层 MLP + BN + Dropout)
        self.cls_head = nn.Sequential(
            nn.Linear(trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        # BBox 回归头
        self.box_head = nn.Sequential(
            nn.Linear(trans_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 6),
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _normalize_input_points(x):
        """统一输入为 (B, N, 4) xyzi。"""
        if x.ndim != 3:
            raise ValueError(f"PointMAEClassification expects 3D input, got {tuple(x.shape)}")
        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
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
        pts = x[:, :, :3].contiguous()

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(x, pos)
        x = self.norm(x)

        cls_feat = x[:, 0]
        max_feat = x[:, 1:].max(dim=1)[0]
        concat_f = torch.cat([cls_feat, max_feat], dim=-1)

        logits = self.cls_head(concat_f)
        box_pred = self.box_head(concat_f)
        return logits, box_pred


# ============================================================================
# GPU 显存测试 (SKILL 规范)
# ============================================================================

def _gpu_memory_test():
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
            m = PointMAEClassification(num_classes=26).cuda()
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


def _quick_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing Point-MAE on {device}")
    model = PointMAEClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, box_pred = model(pts)
    print(f"Input:  {tuple(pts.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Box:    {tuple(box_pred.shape)}")
    print("OK Point-MAE works correctly")


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()
