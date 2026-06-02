"""SPT: Spiking Point Transformer for Point Cloud Classification + 3D BBox

GitHub:  https://github.com/PeppaWu/SPT
Local:   D:\essay\3d目标检测复现仓库\SPT-main\SPT-main

Strictly reproduces https://github.com/PeppaWu/SPT incorporating spiking nodes,
while appending dual-head regression parameters.

当前本地 SPT 模型结构默认对齐原版 Hengshuang.yaml:
    Hengshuang.yaml:
        nblocks=4, blocks=[1, 1, 1, 1, 1], transformer_dim=512,
        timestep=2, use_encoder=True, num_samples=512,
        spike_mode="lif", nneighbor=16。
    本项目训练默认保持上述结构，仅额外默认开启:
        amp=True, tf32=True。
    本项目任务适配:
        输入为 SPAD xyzi 点云 (B, N, 4)，类别数来自数据集；
        输出从原版单分类头扩展为 {"logits", "box_pred"} 双头，
        其中 box_pred 为 3D bbox 中心点回归。
    必要实现差异:
        原版 Q-SDE 只维护 xyz 剩余点集，后续 gather 仍从原始 x 取点；
        本项目输入为 xyzi，因此 queue_SDE 在剩余 xyzi 点集上同步采样与移除，
        避免 xyz 索引误用于原始 xyzi 点集。

@inproceedings{wu2025spiking,
  title={Spiking point transformer for point cloud classification},
  author={Wu, Peixi and Chai, Bosong and Li, Hebei and Zheng, Menghua and Peng, Yansong and Wang, Zeyu and Nie, Xuan and Zhang, Yueyi and Sun, Xiaoyan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={20},
  pages={21563--21571},
  year={2025}
}
"""

from __future__ import annotations

import time
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pointnet_utils import (
    square_distance,
    index_points,
    index_points_fast,
    farthest_point_sample_fast,
    build_spike_node,
    PointNetSetAbstraction,
)

class globals:
    MID_TIME = 0.0

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, timestep, spike_mode, use_encoder, use_moe_lif=True) -> None:
        super().__init__()
        input_spike = (
            build_spike_node(timestep, ['lif', 'elif', 'plif', 'if'], d_points)
            if spike_mode is not None and use_moe_lif
            else build_spike_node(timestep, spike_mode)
            if spike_mode is not None
            else nn.Identity()
        )
        self.fc1 = nn.Sequential(
            input_spike,
            nn.Conv1d(d_points, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.fc2 = nn.Sequential(
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
            nn.Conv1d(d_model, d_points, 1), 
            nn.BatchNorm1d(d_points)
        )
        self.fc_delta = nn.Sequential(
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            nn.Conv2d(3, d_model, 1),
            nn.BatchNorm2d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.ReLU(),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.fc_gamma = nn.Sequential(
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.ReLU(),
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
        )

        self.w_qs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.w_ks = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.w_vs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), 
            nn.BatchNorm1d(d_model),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else  nn.Identity(),
        )
        self.k = k
        self.use_encoder = use_encoder
        
    def forward(self, xyz, features):
        T = xyz.shape[0]
        loc = xyz[0] if not self.use_encoder else xyz
        dists = square_distance(loc, loc)
        k_neighbors = min(self.k, dists.shape[-1])
        knn_idx = dists.topk(k=k_neighbors, dim=-1, largest=False, sorted=False)[1]
        knn_xyz = index_points(loc, knn_idx)
        knn_idx = knn_idx.repeat(T, 1, 1, 1).flatten(0,1) \
                if not self.use_encoder else \
                knn_idx.flatten(0,1)

        features = features.flatten(0,1).permute(0,2,1).contiguous()
        pre = features

        x = self.fc1(features)

        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        k = index_points(k.permute(0,2,1), knn_idx).permute(0,3,1,2).contiguous()
        v = index_points(v.permute(0,2,1), knn_idx).permute(0,3,1,2).contiguous()
        
        pos_enc = self.fc_delta((xyz[:, :, :, None] - (knn_xyz.repeat(T, 1, 1, 1 ,1) \
                                                       if not self.use_encoder else knn_xyz)).flatten(0,1).permute(0,3,1,2).contiguous()) 

        attn = self.fc_gamma(q[:, :, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(1)), dim=-1)  
        
        res = torch.einsum('bcnm,bcnm->bcn', attn, v + pos_enc)
        res = self.fc2(res) + pre
        res = res.permute(0,2,1).reshape(T, xyz.shape[1], xyz.shape[2], -1)
        return res, attn


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels, timestep, spike_mode, use_encoder):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], timestep, spike_mode, use_encoder, group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        npoints = getattr(cfg, 'num_point', 1024)
        if hasattr(cfg, 'model'):
            nblocks = getattr(cfg.model, 'nblocks', 4)
            nneighbor = getattr(cfg.model, 'nneighbor', 16)
            blocks = getattr(cfg.model, 'blocks', [1, 1, 1, 1, 1])
            num_samples = getattr(cfg.model, 'num_samples', 512)
            spike_mode = getattr(cfg.model, 'spike_mode', 'lif')
            timestep = getattr(cfg.model, 'timestep', 2)
            use_encoder = getattr(cfg.model, 'use_encoder', True)
            transformer_dim = getattr(cfg.model, 'transformer_dim', 512)
            use_moe_lif = getattr(cfg.model, 'use_moe_lif', True)
        else:
            nblocks, nneighbor, blocks = 4, 16, [1, 1, 1, 1, 1]
            num_samples, spike_mode, timestep, use_encoder, transformer_dim = 512, 'lif', 2, True, 512
            use_moe_lif = True
        
        d_points = getattr(cfg, 'input_dim', 4)  

        assert len(blocks) == nblocks+1, "Block mismatches"

        self.fc1 = nn.Sequential(
            nn.Conv1d(d_points, 32, 1),
            nn.BatchNorm1d(32),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Conv1d(32, 32, 1), 
            nn.BatchNorm1d(32),
        )

        transblock = lambda channel: TransformerBlock(channel, transformer_dim, nneighbor, timestep, spike_mode, use_encoder, use_moe_lif)
        self.transformer1 = nn.ModuleList(transblock(32) for _ in range(blocks[0]))

        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel], timestep, spike_mode, use_encoder))
            for _ in range(blocks[i + 1]):
                self.transformers.append(transblock(channel))

        self.nblocks = nblocks
        self.blocks = blocks
    
    def forward(self, x):
        T, B, N, C = x.shape
        xyz = x[..., :3]
        x = self.fc1(x.flatten(0, 1).permute(0, 2, 1).contiguous())
        x = x.view(T, B, -1, N).permute(0, 1, 3, 2).contiguous()
        points = self.transformer1[0](xyz, x)[0]

        xyz_and_feats = [(xyz, points)]
        id = 0 
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            for _ in range(self.blocks[i + 1]):                
                points = self.transformers[id](xyz, points)[0]
                id += 1
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class PointTransformerCls(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = Backbone(cfg)
        
        npoints = getattr(cfg, 'num_point', 1024)
        if hasattr(cfg, 'model'):
            nblocks = getattr(cfg.model, 'nblocks', 4)
            spike_mode = getattr(cfg.model, 'spike_mode', 'lif')
            timestep = getattr(cfg.model, 'timestep', 2)
            use_encoder = getattr(cfg.model, 'use_encoder', True)
            num_samples = getattr(cfg.model, 'num_samples', 512)
        else:
            nblocks, spike_mode, timestep, use_encoder, num_samples = 4, 'lif', 2, True, 512

        num_classes = getattr(cfg, 'num_classes', 26)  # From the SPAD runner
        
        self.fc2_cls = nn.Sequential(
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            nn.Conv1d(32 * 2 ** nblocks, 256, 1),
            nn.BatchNorm1d(256),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Conv1d(64, num_classes, 1),
        )
        
        self.fc2_box = nn.Sequential(
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.Identity(),
            nn.Conv1d(32 * 2 ** nblocks, 256, 1),
            nn.BatchNorm1d(256),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Conv1d(256, 64, 1),
            nn.BatchNorm1d(64),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Conv1d(64, 3, 1),
        )

        self.nblocks = nblocks
        self.T = timestep if spike_mode is not None else 1
        self.spike_mode  = spike_mode
        self.use_encoder = use_encoder
        self.num_samples = max(npoints//self.T, num_samples)

    def queue_SDE(self, x):
        def sample_points(points, sample_count):
            """采样当前剩余点；采全量时直接返回顺序索引，避免 FPS 重复索引。"""
            batch_size, num_remaining, channels = points.shape
            sample_count = min(int(sample_count), num_remaining)
            if sample_count <= 0:
                empty_idx = torch.empty(batch_size, 0, dtype=torch.long, device=points.device)
                return points[:, :0, :], empty_idx

            if sample_count == num_remaining:
                sample_idx = torch.arange(num_remaining, device=points.device).view(1, -1).expand(batch_size, -1)
            else:
                # 原版使用 CUDA FPS；这里优先走 pointnet2_ops，失败时回退到纯 PyTorch FPS。
                sample_idx = farthest_point_sample_fast(points[..., :3].contiguous(), sample_count)
            return index_points_fast(points, sample_idx), sample_idx.long()

        def remove_selected(points, selected_idx):
            """按目标数量移除本轮采样点，避免重复 FPS 索引造成各 batch 剩余长度不同。"""
            if selected_idx.numel() == 0:
                return points
            batch_size, num_remaining, channels = points.shape
            remove_count = min(int(selected_idx.shape[1]), num_remaining)
            target_remaining = num_remaining - remove_count
            if target_remaining <= 0:
                return points[:, :0, :]

            selected_idx = selected_idx.long().clamp(0, num_remaining - 1)
            mask = torch.ones(batch_size, num_remaining, dtype=torch.bool, device=points.device)
            batch_idx = torch.arange(batch_size, device=points.device).unsqueeze(1).expand_as(selected_idx)
            mask[batch_idx, selected_idx] = False

            order = torch.arange(num_remaining, device=points.device).view(1, -1).expand(batch_size, -1)
            keep_order = torch.where(mask, order, order + num_remaining)
            keep_idx = keep_order.argsort(dim=1)[:, :target_remaining]
            return index_points_fast(points, keep_idx)
        
        B, N, C = x.shape
        npoint = min(self.num_samples, N)
        res = max((N - npoint)//(self.T-1), 0) if self.T != 1 else 0

        onion = x.new_empty(self.T, B, npoint, C)
        remaining = x
        sampled, fps_idx = sample_points(remaining, npoint)
        onion[0] = sampled
        if self.T > 1:
            remaining = remove_selected(remaining, fps_idx)

        for i in range(1, self.T):
            take_count = min(res, remaining.shape[1])
            if take_count <= 0:
                onion[i] = onion[i-1]
            else:
                # 原版 Q-SDE 的队列思想：保留上一帧后段，末尾补入剩余点中的新 FPS 点。
                sampled, fps_idx = sample_points(remaining, take_count)
                onion[i, :, :npoint - take_count] = onion[i - 1][:, take_count:]
                onion[i, :, npoint - take_count:] = sampled
                if i + 1 < self.T:
                    remaining = remove_selected(remaining, fps_idx)
        return onion

    def forward(self, x) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            if x.shape[-1] not in [3, 4] and x.shape[1] in [3, 4]:
                x = x.transpose(1, 2).contiguous()
            
            if x.shape[-1] == 3:
                x = torch.cat([x, torch.zeros((x.shape[0], x.shape[1], 1), dtype=x.dtype, device=x.device)], dim=-1)
                
        assert len(x.shape) < 4, "shape of inputs is invalid"
        st = time.time()
        if self.spike_mode is not None:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) \
                if not self.use_encoder else \
                self.queue_SDE(x)
        else:
            x = x.unsqueeze(0)
        end = time.time()
        globals.MID_TIME = end - st

        points, _ = self.backbone(x)

        points = points.mean(2) if len(points.shape) == 4 else points.mean(1)
        points = points.unsqueeze(-1)
        
        res_cls = self.fc2_cls(points.flatten(0,1))
        res_cls = res_cls.view(self.T, -1, *res_cls.shape[1:]).mean(0).squeeze(-1)
        
        res_box = self.fc2_box(points.flatten(0,1))
        res_box = res_box.view(self.T, -1, *res_box.shape[1:]).mean(0).squeeze(-1)
        
        return {"logits": res_cls, "box_pred": res_box}

SPTNet = PointTransformerCls


# ═══════════════════════════════════════════════════════
# 验证 + GPU 显存测试
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import gc
    from types import SimpleNamespace

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== SPT 验证 (device={device}) ===")

    # 创建简易配置
    cfg = SimpleNamespace(
        num_point=1024,
        input_dim=4,
        num_classes=26,
        model=SimpleNamespace(
            nblocks=4,
            nneighbor=16,
            blocks=[1, 1, 1, 1, 1],
            num_samples=512,
            spike_mode="lif",
            timestep=2,
            use_encoder=True,
            transformer_dim=512,
            use_moe_lif=True,
        ),
    )

    model = PointTransformerCls(cfg).to(device)
    pts = torch.randn(2, 1024, 4, device=device)
    out = model(pts)
    print(f"logits: {out['logits'].shape}, box_pred: {out['box_pred'].shape}")

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
                m = PointTransformerCls(cfg).cuda()
                pts = torch.randn(bs, N, 4).cuda()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                m.train()
                o = m(pts)
                loss = o['logits'].sum() + o['box_pred'].sum()
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
