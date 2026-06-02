"""
Point-MAE: Masked Autoencoders for Point Cloud Self-Supervised Learning

GitHub:  https://github.com/Pang-Yatian/Point-MAE
Local:   D:\\essay\\3d目标检测复现仓库\\Point-MAE-main

严格对齐官方 ``models/Point_MAE.py`` 中的 ``PointTransformer`` (finetune model)
并改造为 SPAD 双头 (logits + 中心点) 输出。

官方 finetune 模型设计要点 (与 SPAD 实现对应关系):
- ``Group`` (FPS + KNN)                  → utils.transformer_blocks.PatchGroup
- ``Encoder`` (Mini-PointNet)            → utils.transformer_blocks.PatchEncoder
- 自家 ``Mlp/Attention/Block``           → utils.transformer_blocks.TransformerEncoder
- ``encoder_dims == trans_dim`` 因此官方**没有任何投影层**, patch tokens 直接送入 transformer
- ``cls_token`` + ``cls_pos`` 与 patch tokens / pos_embed 拼接后送入 transformer
- 分类头: ``concat(cls_feat, max_pool_feat)`` → Linear-BN-ReLU-Drop ×2 → Linear

SPAD 适配 (与其他 baseline 统一):
- 输入: (B, N, 4) xyzi, 内部只取 xyz 三通道 (与官方一致)
- 输出: tuple ``(logits [B, num_classes], center_pred [B, 3])``
  - SPAD 项目已重构 bbox 为只回归中心 (固定半宽重建框), 因此 box_head 输出 3 维。

默认超参对齐 ``cfgs/finetune_modelnet.yaml`` 的 PointTransformer 配置:
trans_dim=384, depth=12, num_heads=6, encoder_dims=384, group_size=32,
num_group=64, drop_path_rate=0.1。

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

# 项目根目录入 sys.path 后再 import utils, 避免脚本式调用时找不到 utils。
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.transformer_blocks import (
    PatchGroup,
    PatchEncoder,
    TransformerEncoder,
    trunc_normal_,
)
from utils.heads import build_standard_cls_head, build_standard_box_head


# ============================================================================
# Point-MAE 分类微调模型 (官方 PointTransformer finetune 变体)
# ============================================================================

class PointMAEClassification(nn.Module):
    """Point-MAE 分类 + 3D 中心点回归模型 (SPAD 双头适配)。

    对应官方 ``models/Point_MAE.py`` 的 ``PointTransformer`` (finetune model)。

    架构 (严格对齐官方):
        1. ``PatchGroup``: FPS + KNN 分组得到 (B, G, K, 3) patches 与 (B, G, 3) centers
        2. ``PatchEncoder``: Mini-PointNet 编码为 (B, G, trans_dim) patch tokens
           (官方 ``encoder_dims == trans_dim``, 无任何投影层)
        3. 拼接 ``cls_token``、加上由 centers 经 MLP 映射得到的 pos_embed (含 cls_pos)
        4. ``TransformerEncoder``: 12 层 ViT 块 (block 内 ``x = x + pos`` 相加再做 attention)
        5. ``LayerNorm`` → ``concat(cls_feat, max_pool_patch_feat)`` 得到 (B, 2*trans_dim)
        6. 分类头: Linear-BN-ReLU-Drop ×2 → Linear(num_classes)
        7. SPAD 中心点回归头: Linear-BN-LeakyReLU-Drop → Linear(3)

    Args:
        num_classes: 分类类别数, 对应官方 ``cls_dim``。
        trans_dim: transformer hidden 维度, 与 ``encoder_dims`` 必须一致。
        depth: transformer 层数, 官方 finetune 配置为 12。
        num_heads: 多头注意力头数。
        drop_path_rate: 随机深度的最大比例 (各层线性插值)。
        group_size: 每个局部 patch 的 KNN 邻居数 K。
        num_group: FPS 采样的 patch 中心数 G。
        encoder_dims: PatchEncoder 的输出通道, 官方要求与 ``trans_dim`` 相等。

    输入: (B, N, 4) xyzi (或可被 ``_normalize_input_points`` 重排的形态)
    输出: tuple
        - logits: (B, num_classes)
        - center_pred: (B, 3) — SPAD 中心点回归 (固定半宽重建 6D bbox)
    """

    def __init__(self, num_classes: int = 26, trans_dim: int = 384,
                 depth: int = 12, num_heads: int = 6, drop_path_rate: float = 0.1,
                 group_size: int = 32, num_group: int = 64,
                 encoder_dims: int = 384, dropout: float = 0.3, **kwargs):
        super().__init__()
        # 官方约束: encoder_dims 必须等于 trans_dim, 否则 patch tokens 维度与 transformer 不匹配。
        if encoder_dims != trans_dim:
            raise ValueError(
                f"Point-MAE 官方实现要求 encoder_dims == trans_dim, "
                f"got encoder_dims={encoder_dims}, trans_dim={trans_dim}"
            )

        self.trans_dim = trans_dim
        self.depth = depth
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims

        # 1) Patch 分组 + Mini-PointNet 编码 (官方 Group / Encoder)
        self.group_divider = PatchGroup(num_group=num_group, group_size=group_size)
        self.encoder = PatchEncoder(encoder_channel=encoder_dims)

        # 2) cls_token / cls_pos: 参数初始化与官方一致 (zeros + randn, 再截断正态覆盖)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))

        # 3) 位置编码 MLP: 3 (中心 xyz) → 128 → trans_dim
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        # 4) Transformer 编码器: depth 层标准 ViT Block, 内部以 x+pos 相加再 attn
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim, depth=depth,
            drop_path_rate=drop_path_rate, num_heads=num_heads,
        )

        self.norm = nn.LayerNorm(trans_dim)

        # 5) 统一分类头: pooled_dim = trans_dim * 2 (cls_token + max_pool 拼接)
        pooled_dim = trans_dim * 2
        self.cls_head = build_standard_cls_head(pooled_dim, num_classes, dropout=dropout)

        # 6) 统一中心点回归头: 直接回归 [B, 3] 中心坐标
        self.box_head = build_standard_box_head(pooled_dim, dropout=dropout)

        # 与官方一致: 仅对 cls_token / cls_pos 用截断正态初始化, 其余靠 _init_weights。
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """与官方 ``_init_weights`` 一致: Linear/Conv1d 截断正态 0.02, LayerNorm = (1, 0)。"""
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
    def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
        """把 (B, N, 3/4) 或 (B, 3/4, N) 统一成 (B, N, 4) xyzi。

        - 若末维是 3 或 4, 视为 (B, N, C);
        - 若第 1 维是 3 或 4, 视为 (B, C, N) 并转置;
        - 缺失 intensity 通道 (C=3) 时, 末尾补零成 xyzi。
        """
        if x.ndim != 3:
            raise ValueError(f"PointMAEClassification expects 3D input, got {tuple(x.shape)}")
        if x.shape[-1] in (3, 4):
            points = x
        elif x.shape[1] in (3, 4):
            # (B, C, N) → (B, N, C)
            points = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
        if points.shape[-1] == 3:
            pad_i = torch.zeros(points.shape[0], points.shape[1], 1,
                                dtype=points.dtype, device=points.device)
            # (B, N, 3) → (B, N, 4): 末尾拼接 intensity=0 占位通道
            points = torch.cat([points, pad_i], dim=-1)
        return points

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, 4) xyzi 点云 (兼容 (B, 3/4, N) / (B, N, 3) 输入)

        Returns:
            logits: (B, num_classes)
            center_pred: (B, 3) — 归一化到样本坐标系的物体中心
        """
        x = self._normalize_input_points(x)
        # 官方 finetune 模型仅消费 xyz, intensity 通道丢弃。
        pts = x[:, :, :3].contiguous()

        # 1) 分组: (B, N, 3) → (B, G, K, 3) patches + (B, G, 3) centers
        neighborhood, center = self.group_divider(pts)
        # 2) Mini-PointNet 编码: (B, G, K, 3) → (B, G, trans_dim) (官方 encoder_dims==trans_dim)
        group_input_tokens = self.encoder(neighborhood)

        # 3) cls_token / cls_pos 扩展到 batch 维: (1, 1, C) → (B, 1, C)
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # 位置编码: (B, G, 3) → (B, G, trans_dim)
        pos = self.pos_embed(center)

        # 4) 拼接 cls 与 patch tokens / pos: (B, 1+G, trans_dim)
        tokens = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos_full = torch.cat((cls_pos, pos), dim=1)

        # 5) Transformer + 最终 LayerNorm
        tokens = self.blocks(tokens, pos_full)
        tokens = self.norm(tokens)

        # 6) 全局特征聚合: 官方做法是 cls_token 与 patch tokens 的 max-pool 拼接
        #    (B, 1+G, C) → cls_feat (B, C) + max_feat (B, C) → concat_f (B, 2C)
        cls_feat = tokens[:, 0]
        max_feat = tokens[:, 1:].max(dim=1)[0]
        concat_f = torch.cat([cls_feat, max_feat], dim=-1)

        logits = self.cls_head(concat_f)
        center_pred = self.box_head(concat_f)
        return logits, center_pred


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

    n_points = 1024
    for bs in [4, 8, 16, 32]:
        try:
            model = PointMAEClassification(num_classes=26).cuda()
            pts = torch.randn(bs, n_points, 4).cuda()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            out = model(pts)
            # 临时 loss: 同时覆盖 logits 与 center_pred 两个输出分支
            loss = out[0].sum() + out[1].sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  B={bs:2d}: peak {peak:6.0f} MB")
            del model, pts, out, loss
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
    logits, center_pred = model(pts)
    print(f"Input:       {tuple(pts.shape)}")
    print(f"Logits:      {tuple(logits.shape)}")
    print(f"CenterPred:  {tuple(center_pred.shape)}")
    assert logits.shape == (2, 26)
    assert center_pred.shape == (2, 3)
    print("OK Point-MAE works correctly")


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()
