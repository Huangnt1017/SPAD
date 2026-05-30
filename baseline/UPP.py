"""
UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis (ICCV 2025)

GitHub:  https://github.com/zhoujiahuan1991/ICCV2025-UPP
Local:   D:\\essay\\3d目标检测复现仓库\\ICCV2025-UPP

复现说明 — UPP 是一个 PEFT (参数高效微调) 框架, 把"去噪"和"补全"两件事
统一为 **point-level prompting**, 套在冻结的 Point-MAE 骨干外面。这里按 SPAD 双头
(logits + 中心点) 改造, 同时保留官方架构的核心三条路径:

    1) 校正路径 (rectify):  对噪声点云用 RectifyPrompter 预测每点修正向量 + 噪声分数,
       依分数过滤掉最噪的 5% (官方默认), 剩余 95% 点送入下游。
    2) 补全路径 (completion): 用 shape_pred → coarse_pred 预测稀疏中心点,
       经 MAE decoder 生成密集点, FPS 合并回原云, 缓解残缺。
    3) 下游分类路径 (downstream): 经 RectifyPrompter + completion 处理后的点云,
       通过 Block 内的 ``downstream_adapter`` + ``downstream_prompts`` 进入 transformer,
       与 cls_token 一起聚合, 经 cls_head_finetune → logits。

关键模块 (与官方对照):
- ``UPPBlock``               ← models/Point_MAE_pretask_dev.py::Block (含 3 套 adapter / prompt 槽位)
- ``UPPTransformerEncoder``  ← TransformerEncoder (按 path 动态截断 block 深度)
- ``UPPTransformerDecoder``  ← TransformerDecoder (补全路径用)
- ``Adapter``                ← Adapter (PEFT 瓶颈, LN → Linear↓ → GELU → Linear↑)
- ``RectifyPrompter``        ← RectifyPrompter (PointNet SA + 2× FP + score_head)
- ``PositionalEmbedding``    ← Fourier 频率编码 (NeRF 风格)
- ``upp_pooling``            ← pooling (level-2 中心特征聚合, 官方代码中漏定义, 这里按
                              language semantics 补成 max-pool + BN 变换)

SPAD 适配点:
- KNN 替换: 官方 ``knn_cuda.KNN`` → ``utils.pointnet_utils.knn_point`` (纯 PyTorch)
- 损失替换: 官方 CD/EMD/Pytorch3D → SPAD 用普通监督, 故 forward 不返回重建 loss
- 双头输出: 在分类特征基础上额外接 ``box_head`` 输出 3 维中心点
- 输入: (B, N, 4) xyzi, 内部只取 xyz (与 Point-MAE 一致)

Reference:
@inproceedings{ai2025upp,
  title={UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis},
  author={Ai, Zixiang and Cui, Zhenyu and Peng, Yuxin and Zhou, Jiahuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.transformer_blocks import (
    PatchGroup,
    PatchEncoder,
    DropPath,
    Attention,
    Mlp,
    trunc_normal_,
)
from utils.pointnet_utils import (
    square_distance,
    knn_point,
    farthest_point_sample_fast,
    index_points_fast,
)


# ============================================================================
# Fourier 位置编码 — NeRF 风格 (对应官方 PositionalEmbedding)
# ============================================================================

class PositionalEmbedding(nn.Module):
    """NeRF 风格 Fourier 编码: x → (x, sin(2^k x), cos(2^k x), ...)。

    输出维度 = in_features * (2 * N_freqs + 1)。
    """

    def __init__(self, n_freqs: int = 4, logscale: bool = True):
        super().__init__()
        self.n_freqs = n_freqs
        if logscale:
            # 2^0, 2^1, ..., 2^(n-1)
            freq_bands = 2.0 ** torch.linspace(0, n_freqs - 1, n_freqs)
        else:
            freq_bands = torch.linspace(1.0, 2.0 ** (n_freqs - 1), n_freqs)
        # 注册为 buffer, 自动随模型 to(device)
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., f) — 末维任意, 通常是 3 (xyz)。
        Returns:
            out: (..., f * (2 * N_freqs + 1))
        """
        # 原始 x + 每个频率下的 sin / cos
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
        return torch.cat(out, dim=-1)


# ============================================================================
# Adapter — PEFT 瓶颈层 (LN → 降维 → GELU → 升维, scale 0.7)
# ============================================================================

class Adapter(nn.Module):
    """官方 Adapter, 对应 models/Point_MAE_pretask_dev.py::Adapter。

    Args:
        embed_dims: 输入/输出维度 (与 transformer hidden 一致)。
        reduction_dims: 瓶颈中间维度 (官方默认 32)。
        drop_rate_adapter: 瓶颈中的 dropout 概率。
    """

    def __init__(self, embed_dims: int, reduction_dims: int = 32,
                 drop_rate_adapter: float = 0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.reduction_dims = reduction_dims
        self.layer_norm = nn.LayerNorm(embed_dims)
        self.ln1 = nn.Linear(embed_dims, reduction_dims)
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(drop_rate_adapter)
        self.ln2 = nn.Linear(reduction_dims, embed_dims)
        # 与官方一致: kaiming_uniform_ + 小常数 bias
        for m in (self.ln1, self.ln2):
            nn.init.kaiming_uniform_(m.weight, a=5.0 ** 0.5)
            nn.init.normal_(m.bias, std=1e-6)
        self.scale = 0.7  # 官方写死的常数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        out = self.ln1(x)
        out = self.activate(out)
        out = self.dropout(out)
        out = self.ln2(out)
        return out * self.scale


# ============================================================================
# upp_pooling — level-2 中心特征聚合 (官方漏定义, 这里按 propagate 语义补)
# ============================================================================

def upp_pooling(grouped_feat: torch.Tensor, transform: nn.Module) -> torch.Tensor:
    """对邻域内特征做 max-pool 并经 BN 变换。

    Args:
        grouped_feat: (B, G, K, C) — 每个中心的 K 邻居特征。
        transform: 通常是 BatchNorm1d, 用于变换池化后的特征。
    Returns:
        pooled: (B, G, C) — 池化后的中心特征 (经 BN 整形)。
    """
    # (B, G, K, C) → max-pool over K → (B, G, C)
    pooled, _ = grouped_feat.max(dim=2)
    bs, g, c = pooled.shape
    # BN1d 期望 (B, C, *), 因此 (B, G, C) → (B, C, G) → BN → (B, G, C)
    pooled = transform(pooled.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    return pooled


# ============================================================================
# 邻域特征插值 — 对应官方 propagate / PointNetFeaturePropagation
# ============================================================================

def feature_propagate(xyz1: torch.Tensor, xyz2: torch.Tensor,
                      points1: Optional[torch.Tensor], points2: torch.Tensor,
                      de_neighbors: int = 6, dist_e: float = 1e-8,
                      residual_weight: float = 0.3) -> torch.Tensor:
    """从 (xyz2, points2) 反距离加权插值到 xyz1, 与可选的 points1 残差融合。

    对应官方 propagate (xyz1: B,N,3; xyz2: B,S,3; points1: B,N,C; points2: B,S,C)。

    Args:
        xyz1: (B, N, 3) — 查询点坐标 (上层 / 细)。
        xyz2: (B, S, 3) — 已知点坐标 (下层 / 粗)。
        points1: (B, N, C) or None — 上层已知特征, 与插值结果以 residual_weight 相加。
        points2: (B, S, C) — 下层特征 (插值源)。
        de_neighbors: 取距离最近的 K 个 xyz2 做 IDW。
        dist_e: 防 0 除小常数 (注意官方对补全用 1e-4, 对 propagate 用 1e-8)。
        residual_weight: 插值结果叠回 points1 的权重 (官方对 propagate 用 0.3)。
    Returns:
        new_points: (B, N, C) — 上层位置上的融合特征。
    """
    # (B, N, S) 完整 pairwise 距离
    dists = square_distance(xyz1, xyz2)
    # 取最近 K 个邻居
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[..., :de_neighbors], idx[..., :de_neighbors]

    # IDW 权重: 倒数距离归一化, dist_e 防 0 除
    dist_recip = 1.0 / (dists + dist_e)
    norm = dist_recip.sum(dim=-1, keepdim=True)
    weight = (dist_recip / norm).unsqueeze(-1)  # (B, N, K, 1)

    # (B, N, K, C) 邻居特征 × 权重 → (B, N, C)
    interpolated = (index_points_fast(points2, idx) * weight).sum(dim=2)
    if points1 is None:
        return interpolated
    # 与官方 propagate 一致: 上层特征 + 0.3 × 插值
    return points1 + residual_weight * interpolated


# ============================================================================
# UPPBlock — 含 3 套 adapter / prompt 槽位的 Transformer 块
# ============================================================================

class UPPBlock(nn.Module):
    """单层 transformer block, 三套 PEFT 槽位 (rectify / pretask / downstream)。

    与官方对照: Point_MAE_pretask_dev.py::Block。
    forward 必须接收 ``path`` ∈ {'rectify', 'pretask', 'downstream'} 来选择槽位。

    Args:
        dim: hidden 维度。
        num_heads: MHA 头数。
        mlp_ratio: FFN 隐藏维度倍率。
        drop_path: 随机深度概率。
        block_idx: 本块在 stack 中的位置 (0-based), 用来与 ``*_depth`` 比较决定是否启用 prompt。
        prompter_config: 含 ``rectify_*`` / ``pretask_*`` / ``downstream_*`` 等键, 见官方
            cfgs/unify_modelnet_cls.yaml 的 prompter_config 节。
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None,
                 drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0,
                 block_idx: int = 0,
                 prompter_config: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.block_idx = block_idx
        prompter_config = prompter_config or {}

        # —— 标准 ViT 块组件 ——
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=drop)

        # level-2 propagation 时用到的 BN1d (对池化后的中心特征做归一化)
        self.bnorm = nn.BatchNorm1d(dim)

        # —— 三套 PEFT 槽位 (按 block_idx 与配置中的 *_depth 决定是否实例化) ——
        # rectify 路径
        self.rectify_adapter = None
        self.rectify_prompts = None
        if prompter_config.get("rectify_adapter", False) \
                and block_idx < prompter_config.get("rectify_depth", 0):
            self.rectify_adapter = Adapter(embed_dims=dim, reduction_dims=32, drop_rate_adapter=0.1)
        if prompter_config.get("rectify_prompts", False) \
                and block_idx < prompter_config.get("rectify_prompts_depth", 0):
            self.rectify_prompts = nn.Parameter(
                torch.zeros(prompter_config.get("rectify_prompts_num", 3), dim)
            )
            nn.init.xavier_uniform_(self.rectify_prompts)

        # pretask (补全) 路径
        self.pretask_adapter = None
        self.pretask_prompts = None
        if prompter_config.get("pretask_adapter", False) \
                and block_idx < prompter_config.get("pretask_depth", 0):
            self.pretask_adapter = Adapter(embed_dims=dim, reduction_dims=32, drop_rate_adapter=0.1)
        if prompter_config.get("pretask_prompts", False) \
                and block_idx < prompter_config.get("pretask_prompts_depth", 0):
            self.pretask_prompts = nn.Parameter(
                torch.zeros(prompter_config.get("pretask_prompts_num", 3), dim)
            )
            nn.init.xavier_uniform_(self.pretask_prompts)

        # downstream (分类) 路径
        self.downstream_adapter = None
        self.downstream_prompts = None
        if prompter_config.get("downstream_adapter", False):
            self.downstream_adapter = Adapter(embed_dims=dim, reduction_dims=32, drop_rate_adapter=0.1)
        if prompter_config.get("downstream_prompts", False) \
                and block_idx < prompter_config.get("downstream_prompts_depth", 0):
            self.downstream_prompts = nn.Parameter(
                torch.zeros(prompter_config.get("downstream_prompts_num", 10), dim)
            )
            nn.init.xavier_uniform_(self.downstream_prompts)

    def _select_prompts(self, path: str) -> Optional[torch.Tensor]:
        if path == "rectify":
            return self.rectify_prompts
        if path == "pretask":
            return self.pretask_prompts
        if path == "downstream":
            return self.downstream_prompts
        return None

    def _select_adapter(self, path: str) -> Optional[Adapter]:
        if path == "rectify":
            return self.rectify_adapter
        if path == "pretask":
            return self.pretask_adapter
        if path == "downstream":
            return self.downstream_adapter
        return None

    def forward(self, x: torch.Tensor, path: str = "downstream",
                classification: bool = False,
                propagation: Optional[dict] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — 输入 token 序列。
            path: 选择本次走哪条 PEFT 路径的 prompt / adapter 槽位。
            classification: True 时第 0 个 token 是 cls_token, prompt 要插在它后面。
            propagation: 可选, 启用 level-2 prompt propagation 时传入字典:
                {center1: (B,G,3), center2: (B,G2,3), level1_idx: 见下, level2_idx: 见下}
                propagation 的 idx 形状为 (B*G2, K1) (level1) 与 (B,G2) (level2),
                与 SPAD utils.PatchGroup 的 (B,G,K) 不直接兼容, 因此本复现里通过
                `_run_propagation` 重新组织; 调用方按下方约定传 raw idx。
        Returns:
            x: (B, T, C) — 块输出 (prompt token 已剥离, 仅余原始 token + cls)。
        """
        # —— Step 1: 选择并拼接 prompt token (放在 cls_token 后或最前) ——
        prompts = self._select_prompts(path)
        prompt_tokens = None
        if prompts is not None:
            prompt_tokens = prompts.unsqueeze(0).expand(x.shape[0], -1, -1)
            if classification:
                # 顺序: cls, [prompts], 其它 patch tokens
                x = torch.cat([x[:, 0:1], prompt_tokens, x[:, 1:]], dim=1)
            else:
                x = torch.cat([prompt_tokens, x], dim=1)

        # —— Step 2: 标准 ViT 块 (Attn + 残差) ——
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # (官方在 attn 残差之后插了一个可选 adapter1, 默认未启用; 这里省略以避免冗余空模块。)

        # —— Step 3: FFN + 残差 ——
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # —— Step 4: prompt-aware 多层次特征传播 (可选, 在剥离 prompt 之前完成) ——
        if prompt_tokens is not None and propagation is not None:
            x = self._run_propagation(x, classification=classification, propagation=propagation)

        # —— Step 5: 剥离 prompt token, 恢复成 cls + patch tokens 序列 ——
        if prompt_tokens is not None:
            tokens_num = prompt_tokens.shape[1]
            if classification:
                # cls 在最前, 紧跟着是 prompts; tokens_num+1 之后是 patch tokens
                x = torch.cat([x[:, 0:1], x[:, tokens_num + 1:]], dim=1)
            else:
                x = x[:, tokens_num:]

        # —— Step 6: PEFT adapter 残差 (本路径对应那条) ——
        adapter = self._select_adapter(path)
        if adapter is not None:
            x = x + adapter(x)

        return x

    def _run_propagation(self, x: torch.Tensor, classification: bool,
                         propagation: dict) -> torch.Tensor:
        """level-2 token 聚合 + 反距离插值, 复现官方 prompt_propagation_after 分支。

        propagation 必须包含:
            center1: (B, G, 3) — patch 中心 (与 x 的 patch tokens 一一对应)。
            center2: (B, G2, 3) — 在 center1 之上再 FPS 得到的二级中心。
            level1_idx: (B, G2, K1) — 每个二级中心在 center1 上的 KNN 邻居索引 (0-based)。
            level2_idx: (B, G2) — center2 在 center1 中的 FPS 采样索引 (0-based)。

        说明: 这里用纯 PyTorch ``torch.gather`` 替代官方 ``reshape + idx_base`` 的把戏,
        语义等价但更直观。
        """
        B = x.shape[0]
        # 拆出 cls / prompts / patch tokens
        if classification:
            cls_x = x[:, 0:1]
            rest = x[:, 1:]
        else:
            cls_x = None
            rest = x

        # rest 形如: [prompts (P), patch_tokens (G)]; level-2 propagation 只动 patch_tokens
        n_prompts = rest.shape[1] - propagation["center1"].shape[1]
        prompt_tokens = rest[:, :n_prompts]
        patch_tokens = rest[:, n_prompts:]  # (B, G, C)

        level1_idx = propagation["level1_idx"]  # (B, G2, K1)
        level2_idx = propagation["level2_idx"]  # (B, G2)
        center1 = propagation["center1"]        # (B, G, 3)
        center2 = propagation["center2"]        # (B, G2, 3)

        # 取 K1 个邻居特征: patch_tokens (B,G,C) gather → (B, G2, K1, C)
        x_neighborhoods = index_points_fast(patch_tokens, level1_idx)
        # 取 G2 个二级中心自身的特征: (B, G2, C)
        x_centers = index_points_fast(patch_tokens, level2_idx)

        # 对每个二级中心邻域内做 max-pool + BN, 再与中心自身按 0.3 残差融合 (对应官方写法)
        pooled_centers = upp_pooling(x_neighborhoods, transform=self.bnorm) + 0.3 * x_centers

        # 从二级中心特征反插值回一级中心 (patch_tokens), 用 patch_tokens 自身做 0.3 残差
        new_patch_tokens = feature_propagate(
            xyz1=center1, xyz2=center2,
            points1=patch_tokens, points2=pooled_centers,
            de_neighbors=8, dist_e=1e-3, residual_weight=0.3,
        )

        # 重新拼回: cls (可选) + prompts + 新 patch_tokens
        if classification:
            return torch.cat([cls_x, prompt_tokens, new_patch_tokens], dim=1)
        return torch.cat([prompt_tokens, new_patch_tokens], dim=1)


# ============================================================================
# UPPTransformerEncoder — 支持按 path 动态截断深度
# ============================================================================

class UPPTransformerEncoder(nn.Module):
    """堆叠 UPPBlock, 在 forward 时按 path 决定走前 K 个 block。

    对应官方 TransformerEncoder (models/Point_MAE_pretask_dev.py)。
    rectify_depth / pretask_depth / downstream_depth 配置见 cfgs/unify_modelnet_cls.yaml。
    """

    def __init__(self, embed_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0, drop_path_rate: float = 0.1,
                 prompter_config: Optional[dict] = None):
        super().__init__()
        self.depth = depth
        self.prompter_config = prompter_config or {}
        # 每层独立的随机深度概率 (线性增长, 与 Point-MAE 官方一致)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            UPPBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], block_idx=i,
                prompter_config=self.prompter_config,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                path: str = "downstream", classification: bool = False,
                propagation: Optional[dict] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — token 序列 (含 cls_token 和 patch tokens)。
            pos: (B, T, C) — 对应的位置编码, 与 x 形状一致, 每层加 x = x + pos。
            path: 走哪条 PEFT 路径。
            classification: 是否第 0 个 token 是 cls_token。
            propagation: 可选 prompt propagation 字典 (见 UPPBlock._run_propagation)。
        """
        # 按 path 决定本次实际走多少 block (官方机制: rectify 浅, pretask 中, downstream 全)
        if path == "rectify":
            n = min(self.prompter_config.get("rectify_depth", self.depth), self.depth)
        elif path == "pretask":
            n = min(self.prompter_config.get("pretask_depth", self.depth), self.depth)
        else:
            n = min(self.prompter_config.get("downstream_depth", self.depth), self.depth)

        for idx in range(n):
            x = self.blocks[idx](x + pos, path=path,
                                 classification=classification,
                                 propagation=propagation)
        return x


# ============================================================================
# UPPTransformerDecoder — 补全路径专用 (与 Point-MAE 解码器相同, 加 pretask adapter)
# ============================================================================

class UPPTransformerDecoder(nn.Module):
    """对应官方 TransformerDecoder, 用于补全路径的 mask token 重建。

    Args:
        embed_dim / depth / num_heads / mlp_ratio: 标准 transformer 配置。
        drop_path_rate: 随机深度。
        prompter_config: 透传以启用 pretask_adapter。
    """

    def __init__(self, embed_dim: int = 384, depth: int = 4, num_heads: int = 6,
                 mlp_ratio: float = 4.0, drop_path_rate: float = 0.1,
                 prompter_config: Optional[dict] = None):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            UPPBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop_path=dpr[i], block_idx=i,
                prompter_config=prompter_config or {},
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, pos: torch.Tensor,
                return_token_num: int, path: str = "pretask") -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — token 序列 (前段是 visible, 末尾是 mask token)。
            pos: 同上, 每层加上。
            return_token_num: 只返回末尾的 N 个 token (mask token), 与官方一致。
        """
        for block in self.blocks:
            x = block(x + pos, path=path, classification=False)
        x = self.norm(x[:, -return_token_num:])
        return x


# ============================================================================
# RectifyPrompter — PointNet SA + 2× FP + Fourier 编码 + score_head
# ============================================================================

class _PointNetSAForUPP(nn.Module):
    """PointNet++ Set Abstraction 简化版, 输入 (xyz, point_feat) → (new_xyz, new_feat)。

    与官方 PointNetSetAbstraction 一致, 区别: 用 PatchGroup (FPS+KNN) 取邻域。
    """

    def __init__(self, num_group: int, group_size: int,
                 in_channel: int, mlp_dims: list):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        # PatchGroup 给出归一化邻域 + 中心; 这里只需要 idx, 因此自己再做一遍 FPS+KNN。
        self.group_divider = PatchGroup(num_group=num_group, group_size=group_size)

        # SharedMLP (Conv2d) 处理 (B, C, K, G) 形状
        layers = []
        last = in_channel
        for out in mlp_dims:
            layers += [nn.Conv2d(last, out, 1), nn.BatchNorm2d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)
        self.out_channel = mlp_dims[-1]

    def forward(self, xyz: torch.Tensor, point_feat: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)
            point_feat: (B, N, C) — 每点已有特征 (官方传入 transformer 输出特征)。
        Returns:
            new_xyz: (B, G, 3)
            new_feat: (B, G, out_channel)
        """
        B, N, _ = xyz.shape
        # FPS 中心 + KNN 索引 (复用 SPAD 工具)
        center_idx = farthest_point_sample_fast(xyz, self.num_group)   # (B, G)
        center = index_points_fast(xyz, center_idx)                    # (B, G, 3)
        idx = knn_point(self.group_size, xyz, center)                   # (B, G, K)
        # 邻域特征: (B, N, C) gather → (B, G, K, C)
        grouped = index_points_fast(point_feat, idx)
        # SharedMLP 期望 (B, C, K, G) → (B, out, K, G)
        grouped = grouped.permute(0, 3, 2, 1).contiguous()
        feat = self.mlp(grouped)
        # max-pool over K → (B, out, G) → (B, G, out)
        feat = feat.max(dim=2)[0].permute(0, 2, 1).contiguous()
        return center, feat


class _PointNetFPForUPP(nn.Module):
    """PointNet++ Feature Propagation 简化版, 反距离插值 + Conv1d 序列。

    对应官方 PointNetFeaturePropagation。
    """

    def __init__(self, in_channel: int, mlp_dims: list, interpolate_neighbors: int = 16):
        super().__init__()
        self.interpolate_neighbors = interpolate_neighbors
        layers = []
        last = in_channel
        for out in mlp_dims:
            layers += [nn.Conv1d(last, out, 1), nn.BatchNorm1d(out), nn.ReLU(inplace=True)]
            last = out
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: Optional[torch.Tensor], points2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz1: (B, N, 3) — 待插值目标位置 (上层 / 细)。
            xyz2: (B, S, 3) — 已知位置 (下层 / 粗)。
            points1: (B, N, C1) or None — 上层已有特征, 与插值结果 cat (与官方一致)。
            points2: (B, S, C2) — 下层特征 (插值源)。
        Returns:
            new: (B, N, mlp_dims[-1]) — 融合 + 经 Conv1d MLP 后的上层特征。
        """
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        # 对单一参考点的极端情况直接复制
        if S == 1:
            interpolated = points2.expand(B, N, -1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists = dists[..., :self.interpolate_neighbors]
            idx = idx[..., :self.interpolate_neighbors]
            dist_recip = 1.0 / (dists + 1e-4)  # 注意: 官方在 FP 里用 1e-4, 与 propagate 不同
            norm = dist_recip.sum(dim=-1, keepdim=True)
            weight = (dist_recip / norm).unsqueeze(-1)  # (B, N, K, 1)
            interpolated = (index_points_fast(points2, idx) * weight).sum(dim=2)

        if points1 is not None:
            new = torch.cat([points1, interpolated], dim=-1)
        else:
            new = interpolated

        # (B, N, C) → (B, C, N) → MLP → (B, N, out)
        new = self.mlp(new.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return new


class RectifyPrompter(nn.Module):
    """对应官方 RectifyPrompter, 预测每点 3 维修正向量。

    架构:
        1) PointNet SA: (xyz=center1, feat=center1_feature) → (center2, center2_feature)
        2) FP 2: 把 center2_feature 反插值回 center1 → center1_feature'
        3) FP 1: 用原始点 xyz 上的 Fourier 位置编码 + center1_feature' 反插值到所有点
        4) score_head: 32→64→3, 输出每点修正向量 (即 rectify direction × magnitude)

    Args:
        in_channels: 输入点特征维度 (3, 即 xyz)。
        out_channels: 修正向量维度 (3)。
        hidden_dim: transformer 给出的中心特征维度 (=trans_dim, 默认 384)。
        embedding_level: Fourier 编码频段数 N_freqs (默认 4)。
        num_group / group_size: SA 模块用的二级 FPS / KNN 配置。
        top_center_dim: SA 出口通道 (官方 12)。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 hidden_dim: int = 384, embedding_level: int = 4,
                 num_group: int = 32, group_size: int = 16,
                 top_center_dim: int = 12):
        super().__init__()
        self.position_embedding = PositionalEmbedding(embedding_level)
        self.abstraction = _PointNetSAForUPP(
            num_group=num_group, group_size=group_size,
            in_channel=hidden_dim, mlp_dims=[64, 32, top_center_dim],
        )
        # FP2: 二级 → 一级 (无 points1)
        self.propagation2 = _PointNetFPForUPP(
            in_channel=top_center_dim, mlp_dims=[64, 32],
        )
        # FP1: 一级 → 所有点 (有 points1 = Fourier 位置编码)
        # Fourier 编码后维度 = in_channels * (2*embedding_level + 1)
        fp1_in = in_channels * (2 * embedding_level + 1) + 32
        self.propagation1 = _PointNetFPForUPP(
            in_channel=fp1_in, mlp_dims=[32, 32],
        )

        self.score_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, out_channels),
        )
        self.score_factor = 1.0
        for layer in self.score_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=5.0 ** 0.5)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor, center1: torch.Tensor,
                center1_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) — 所有原始点 xyz (用于做 Fourier 编码)。
            center1: (B, G1, 3) — 一级中心坐标 (vis_center)。
            center1_feature: (B, G1, hidden_dim) — 一级中心特征 (transformer 输出)。
        Returns:
            noise_score: (B, N, 3) — 每点 3 维修正向量 (即 pred_vector)。
        """
        # 二级中心特征
        center2, center2_feature = self.abstraction(center1, center1_feature)
        # FP2: center2 反插值到 center1
        center1_feature_new = self.propagation2(center1, center2, None, center2_feature)
        # FP1: 用 Fourier 编码的原始点位置 + 一级特征 反插值到所有点
        x_fourier = self.position_embedding(x)
        feature = self.propagation1(x, center1, x_fourier, center1_feature_new)
        # 每点 3 维修正向量
        noise_score = self.score_head(feature) * self.score_factor
        return noise_score


# ============================================================================
# UPPClassification — SPAD 双头入口
# ============================================================================

class UPPClassification(nn.Module):
    """UPP 分类 + 中心点回归模型 (SPAD 适配)。

    完整路径 (按官方 Point_MAE_unify.forward 翻译, 三段串联):
        1. denoise=True: RectifyPrompter 预测每点修正向量, 修正 + 过滤掉最噪 5% 点
        2. completion_prompt=True (可选): 经 pretask 路径 + MAE decoder 重建稀疏点云后
           FPS 合并回去 (SPAD 默认关闭以节省显存; 噪声单光子数据更需要的是去噪)
        3. 下游分类: downstream 路径 + cls_head + box_head 给出 (logits, center)

    Args 与 cfgs/unify_modelnet_cls.yaml 对齐:
        num_classes: 分类类别数 (SPAD: 26)。
        trans_dim: 384。 depth: 12。 num_heads: 6。 encoder_dims: 必须 == trans_dim。
        group_size / num_group: patch 配置, 默认 32 / 64。
        vis_num: 可见中心数 = num_group * (1 - mask_ratio), 默认 mask_ratio=0.5 → 32。
        drop_path_rate: 0.1。
        prompter_config: 三套 PEFT 配置字典 (key 名与官方 yaml 一致)。
        enable_completion: 是否启用补全路径 (默认 False, 仅当输入显著残缺时打开)。
        enable_denoise: 是否启用去噪路径 (默认 True, 这是 UPP 在 SPAD 上的主要价值)。
        gather_idx: 与官方 yaml 一致, level-2 propagation 是否走 gather 路径 (默认 False)。

    输入: (B, N, 4) xyzi (或可被规范化的 (B, 3/4, N) / (B, N, 3))。
    输出: tuple
        - logits: (B, num_classes)
        - center_pred: (B, 3) — SPAD 中心点回归输出。
    """

    DEFAULT_PROMPTER_CONFIG = {
        "rectify_adapter": True,
        "rectify_prompts": True,
        "rectify_prompts_num": 3,
        "rectify_prompts_depth": 3,
        "rectify_depth": 3,
        "pretask_adapter": True,
        "pretask_prompts": True,
        "pretask_prompts_num": 3,
        "pretask_prompts_depth": 6,
        "pretask_depth": 6,
        "downstream_adapter": True,
        "downstream_prompts": True,
        "downstream_prompts_num": 10,
        "downstream_prompts_depth": 6,
        "downstream_depth": 12,
    }

    def __init__(self, num_classes: int = 26, trans_dim: int = 384,
                 depth: int = 12, num_heads: int = 6, drop_path_rate: float = 0.1,
                 group_size: int = 32, num_group: int = 64,
                 encoder_dims: int = 384, mask_ratio: float = 0.5,
                 vis_short: int = 16,
                 prompter_config: Optional[dict] = None,
                 enable_denoise: bool = True,
                 enable_completion: bool = False,
                 noise_keep_ratio: float = 0.95,
                 noise_step: float = 0.2,
                 **kwargs):
        super().__init__()
        if encoder_dims != trans_dim:
            raise ValueError(
                f"UPP 沿用 Point-MAE 的 encoder_dims == trans_dim 约束, "
                f"got encoder_dims={encoder_dims}, trans_dim={trans_dim}"
            )

        self.trans_dim = trans_dim
        self.depth = depth
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims
        self.mask_ratio = mask_ratio
        self.vis_num = num_group - int(mask_ratio * num_group)  # 可见中心数, 默认 32
        self.vis_short = vis_short
        self.enable_denoise = enable_denoise
        self.enable_completion = enable_completion
        self.noise_keep_ratio = noise_keep_ratio
        self.noise_step = noise_step

        cfg = dict(self.DEFAULT_PROMPTER_CONFIG)
        if prompter_config:
            cfg.update(prompter_config)
        self.prompter_config = cfg

        # —— Patch 分组器 (主分类路径 num_group×group_size, 校正/补全用 vis_num×16) ——
        self.group_divider = PatchGroup(num_group=num_group, group_size=group_size)
        self.vis_grouper = PatchGroup(num_group=self.vis_num, group_size=16)
        # 二级中心采样器 (主分类前用来构建 level-2 propagation)
        self.level2_grouper = PatchGroup(num_group=num_group // 2, group_size=8)

        # —— Mini-PointNet patch encoder + 位置编码 MLP ——
        self.encoder = PatchEncoder(encoder_channel=encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        # —— UPP TransformerEncoder (含三套 PEFT) ——
        self.blocks = UPPTransformerEncoder(
            embed_dim=trans_dim, depth=depth, num_heads=num_heads,
            drop_path_rate=drop_path_rate, prompter_config=self.prompter_config,
        )
        self.norm = nn.LayerNorm(trans_dim)

        # —— Rectification 路径专属: RectifyPrompter ——
        self.rectify_prompter = RectifyPrompter(
            in_channels=3, out_channels=3,
            hidden_dim=trans_dim, embedding_level=4,
            num_group=32, group_size=16, top_center_dim=12,
        )

        # —— Completion 路径专属: shape_pred → coarse_pred → predict_token_generator
        #    → MAE decoder → dense_pred (官方完整流程) ——
        # shape feature: trans_dim → trans_dim/2 → vis_short, 然后 reshape 成
        # (B, vis_short * vis_num) 喂给 coarse_pred 预测稀疏中心。
        self.shape_pred = nn.Sequential(
            nn.Linear(trans_dim, trans_dim // 2),
            nn.GELU(),
            nn.Linear(trans_dim // 2, vis_short),
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(vis_short * self.vis_num, trans_dim),
            nn.GELU(),
            nn.Linear(trans_dim, 3 * (num_group - self.vis_num)),
        )
        self.predict_token_generator = nn.Sequential(
            nn.Linear(trans_dim, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )
        self.mae_decoder = UPPTransformerDecoder(
            embed_dim=trans_dim, depth=4, num_heads=num_heads,
            drop_path_rate=drop_path_rate, prompter_config=self.prompter_config,
        )
        # dense_pred: 每个 mask token 还原 group_size 个点 (B*M, group_size, 3)
        self.dense_pred = nn.Conv1d(trans_dim, 3 * group_size, 1)

        # —— 分类头 cls_token / cls_pos / cls_head_finetune (与 Point-MAE 一致) ——
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
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

        # —— SPAD 中心点回归头 (官方无此分支) ——
        self.box_head = nn.Sequential(
            nn.Linear(trans_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

        # —— 初始化 ——
        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        for layer in self.cls_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=5.0 ** 0.5)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
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

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _normalize_input_points(x: torch.Tensor) -> torch.Tensor:
        """统一 (B, N, 3/4) / (B, 3/4, N) → (B, N, 4) xyzi。"""
        if x.ndim != 3:
            raise ValueError(f"UPPClassification expects 3D input, got {tuple(x.shape)}")
        if x.shape[-1] in (3, 4):
            pts = x
        elif x.shape[1] in (3, 4):
            # (B, C, N) → (B, N, C)
            pts = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")
        if pts.shape[-1] == 3:
            pad_i = torch.zeros(pts.shape[0], pts.shape[1], 1,
                                dtype=pts.dtype, device=pts.device)
            # (B, N, 3) → (B, N, 4) 末尾补 intensity=0
            pts = torch.cat([pts, pad_i], dim=-1)
        return pts

    def _build_level2_propagation(self, center: torch.Tensor) -> dict:
        """从 patch 中心 center (B, G, 3) 再做一次 FPS+KNN 得到二级中心信息。

        返回字典与 UPPBlock._run_propagation 期望的格式对齐。
        """
        # level2_grouper 给出归一化邻域 (不用) + center2 (B, G2, 3); 再用 knn_point 拿邻域索引。
        # 注意: PatchGroup 内部已做了 FPS + KNN, 这里复用以得到 center2;
        # 同时显式调用 farthest_point_sample / knn_point 拿原始索引 (因为 PatchGroup 不返回 idx)。
        B, G, _ = center.shape
        n_group2 = self.num_group // 2

        # level-2 FPS: G → G2
        level2_idx = farthest_point_sample_fast(center, n_group2)        # (B, G2)
        center2 = index_points_fast(center, level2_idx)                  # (B, G2, 3)
        # level-1 KNN: center2 → 在 center 中找 K1=8 邻居
        level1_idx = knn_point(8, center, center2)                        # (B, G2, K1)

        return {
            "center1": center,
            "center2": center2,
            "level1_idx": level1_idx,
            "level2_idx": level2_idx,
        }

    # ------------------------------------------------------------------ paths

    def _rectify(self, pts: torch.Tensor) -> torch.Tensor:
        """去噪路径: 预测修正向量 + 过滤最噪 5% 点, 返回更新后的点云。

        Args:
            pts: (B, N, 3)。
        Returns:
            new_pts: (B, M, 3), M = round(N * noise_keep_ratio)。
        """
        # vis 中心: vis_num × 16 邻域 (官方默认 32 × 16)
        vis_neighborhood, vis_center = self.vis_grouper(pts)
        # Mini-PointNet 编码每个 vis_neighborhood → (B, vis_num, trans_dim)
        vis_tokens = self.encoder(vis_neighborhood)
        # 位置编码 + 走 rectify 路径 (前 rectify_depth 层)
        pos = self.pos_embed(vis_center)
        vis_tokens = self.blocks(
            vis_tokens, pos,
            path="rectify", classification=False,
        )
        # 用 RectifyPrompter 预测每点修正向量
        pred_vector = self.rectify_prompter(pts, vis_center, vis_tokens)
        # 噪声分数 = 修正向量模长 (越大越像噪声)
        noise_score = pred_vector.norm(dim=-1)  # (B, N)
        noise_idx = torch.argsort(noise_score, dim=1, descending=True)  # (B, N) 从噪到干净
        # 修正: pts ← pts + step × vector (官方 step=0.2)
        pts = pts + self.noise_step * pred_vector
        # 过滤: 保留排序末尾的 noise_keep_ratio 比例 (即丢掉最噪的 1 - ratio)
        n_points = pts.shape[1]
        keep_n = int(n_points * self.noise_keep_ratio)
        keep_idx = noise_idx[:, -keep_n:].unsqueeze(-1).expand(-1, -1, 3)
        # (B, N, 3) gather → (B, keep_n, 3)
        new_pts = torch.gather(pts, 1, keep_idx)
        return new_pts

    def _complete(self, pts: torch.Tensor) -> torch.Tensor:
        """补全路径: 预测稀疏中心 → MAE decode → 密集点 → FPS 合并。

        Args:
            pts: (B, N, 3)。
        Returns:
            new_pts: (B, N, 3) — 合并后再 FPS 到 N 个点 (与官方一致)。
        """
        B, N, _ = pts.shape
        # 同样取 vis_num × 16 邻域
        vis_neighborhood, vis_center = self.vis_grouper(pts)
        vis_tokens = self.encoder(vis_neighborhood)
        pos = self.pos_embed(vis_center)
        # 走 pretask 路径 (前 pretask_depth 层)
        x_vis = self.blocks(
            vis_tokens, pos,
            path="pretask", classification=False,
        )
        x_vis = self.norm(x_vis)

        # shape feature: (B, vis_num, trans_dim) → (B, vis_num, vis_short) → flatten (B, vis_num*vis_short)
        shape_feat = self.shape_pred(x_vis).reshape(B, self.vis_short * self.vis_num)
        n_mask = self.num_group - self.vis_num  # 补全的中心数 (与 mask_ratio*num_group 一致)
        # 粗预测稀疏中心: (B, vis_num*vis_short) → (B, n_mask, 3)
        predict_center = self.coarse_pred(shape_feat).reshape(B, n_mask, 3)
        # 由 vis token 生成 predict token (作为后续 propagate 的下层特征源)
        predict_token = self.predict_token_generator(x_vis)  # (B, vis_num, trans_dim)
        # 位置编码 + 拼接 cls 之前的 vis + mask 部分
        pos_vis = self.decoder_pos_embed(vis_center)            # (B, vis_num, trans_dim)
        pos_mask = self.decoder_pos_embed(predict_center)       # (B, n_mask, trans_dim)
        # mask_token 初始全零 + 反距离插值, 把 vis 处的 predict_token 信息传播到 mask 位置
        mask_token = self.mask_token.expand(B, n_mask, -1)
        mask_token = feature_propagate(
            xyz1=predict_center, xyz2=vis_center,
            points1=mask_token, points2=predict_token,
            de_neighbors=6, dist_e=1e-8, residual_weight=0.3,
        )
        # 拼接送解码器, return_token_num=n_mask 让其只返回 mask 段
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_vis, pos_mask], dim=1)
        x_rec = self.mae_decoder(x_full, pos_full, return_token_num=n_mask, path="pretask")

        # dense_pred: 每个 mask token 还原 group_size 个相对偏移 → 加回 predict_center
        # x_rec: (B, n_mask, trans_dim) → (B, trans_dim, n_mask) → Conv1d → (B, 3*K, n_mask)
        # → (B, n_mask, 3*K) → (B, n_mask, K, 3)
        relative = self.dense_pred(x_rec.transpose(1, 2)).transpose(1, 2)
        relative = relative.reshape(B, n_mask, -1, 3)
        # (B, n_mask, K, 3) + (B, n_mask, 1, 3) → (B, n_mask, K, 3) → (B, n_mask*K, 3)
        rebuild = (relative + predict_center.unsqueeze(2)).reshape(B, -1, 3)

        # 取重建点的前 N/4 (FPS), 与原 pts cat 后再 FPS 到 N (官方流程)
        sample_n = N // 4
        sample_idx = farthest_point_sample_fast(rebuild, sample_n)        # (B, N/4)
        sampled = index_points_fast(rebuild, sample_idx)                  # (B, N/4, 3)
        merged = torch.cat([pts, sampled], dim=1)                         # (B, N + N/4, 3)
        if merged.shape[1] > N:
            final_idx = farthest_point_sample_fast(merged, N)
            merged = index_points_fast(merged, final_idx)
        return merged

    def _classify(self, pts: torch.Tensor) -> torch.Tensor:
        """下游分类路径: 主 patch + cls_token + downstream prompts/adapter → 全局特征。

        Returns:
            concat_f: (B, 2 * trans_dim) — [cls_feat, max_pool_feat], 后续接 cls_head/box_head。
        """
        # 主 patch 分组 (与 Point-MAE 一致)
        neighborhood, center = self.group_divider(pts)
        group_tokens = self.encoder(neighborhood)                 # (B, G, trans_dim)

        # cls_token + 位置编码拼接
        cls_tokens = self.cls_token.expand(group_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_tokens.size(0), -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat([cls_tokens, group_tokens], dim=1)
        pos_full = torch.cat([cls_pos, pos], dim=1)

        # level-2 propagation 信息 (官方 prompt_propagation_after=True)
        propagation = self._build_level2_propagation(center)

        # 走 downstream 路径, classification=True 让 prompt 插在 cls 后
        x = self.blocks(
            x, pos_full,
            path="downstream", classification=True,
            propagation=propagation,
        )
        x = self.norm(x)
        # 全局聚合: cls_token + patch tokens 的 max pool
        cls_feat = x[:, 0]
        max_feat = x[:, 1:].max(dim=1)[0]
        return torch.cat([cls_feat, max_feat], dim=-1)

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, 4) xyzi (或可被规范化的其它形态)。

        Returns:
            logits: (B, num_classes)
            center_pred: (B, 3) — SPAD 中心点回归 (固定半宽重建 6D bbox)。
        """
        x = self._normalize_input_points(x)
        pts = x[:, :, :3].contiguous()

        # 1) 去噪
        if self.enable_denoise:
            pts = self._rectify(pts)

        # 2) 补全 (可选; SPAD 数据通常不需要, 默认关闭以节省显存)
        if self.enable_completion:
            pts = self._complete(pts)

        # 3) 下游分类 + 中心点回归
        concat_f = self._classify(pts)
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
        total_mem = getattr(props, "total_memory", getattr(props, "total_mem", 0))
        if total_mem:
            print(f"总显存: {total_mem / 1024 ** 3:.1f} GB")
    except Exception:
        pass
    print()

    n_points = 1024
    for bs in [4, 8, 16, 32]:
        try:
            model = UPPClassification(num_classes=26).cuda()
            pts = torch.randn(bs, n_points, 4).cuda()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            model.train()
            out = model(pts)
            # 临时 loss 同时覆盖两个分支
            loss = out[0].sum() + out[1].sum()
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
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
    print(f"Testing UPP on {device}")
    model = UPPClassification(num_classes=26).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    logits, center_pred = model(pts)
    print(f"Input:       {tuple(pts.shape)}")
    print(f"Logits:      {tuple(logits.shape)}")
    print(f"CenterPred:  {tuple(center_pred.shape)}")
    assert logits.shape == (2, 26)
    assert center_pred.shape == (2, 3)
    print("OK UPP works correctly (denoise on, completion off)")

    print("\n--- 同时启用 denoise + completion ---")
    model2 = UPPClassification(num_classes=26, enable_completion=True).to(device)
    logits2, center_pred2 = model2(pts)
    print(f"Logits (+completion): {tuple(logits2.shape)}")
    print(f"CenterPred (+completion): {tuple(center_pred2.shape)}")
    assert logits2.shape == (2, 26)
    assert center_pred2.shape == (2, 3)
    print("OK UPP with completion works")


if __name__ == "__main__":
    _quick_test()
    _gpu_memory_test()
