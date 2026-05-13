"""SPT backbone for SPAD point cloud classification and 3D box regression.
Strictly reproduces https://github.com/PeppaWu/SPT incorporating spiking nodes,
while appending dual-head regression parameters.
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
    farthest_point_sample,
    build_spike_node,
    PointNetSetAbstraction,
)

class globals:
    MID_TIME = 0.0

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, timestep, spike_mode, use_encoder) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(
            build_spike_node(timestep, ['lif', 'elif', 'plif', 'if'], d_points) if spike_mode is not None else  nn.Identity(),
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
        knn_idx = dists.argsort()[:, :, :self.k] \
                if not self.use_encoder else \
                dists.argsort()[:, :, :, :self.k]
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
            nblocks = getattr(cfg.model, 'nblocks', 3)
            nneighbor = getattr(cfg.model, 'nneighbor', 16)
            blocks = getattr(cfg.model, 'blocks', [1, 1, 1, 1])
            num_samples = getattr(cfg.model, 'num_samples', 256)
            spike_mode = getattr(cfg.model, 'spike_mode', 'lif')
            timestep = getattr(cfg.model, 'timestep', 4)
            use_encoder = getattr(cfg.model, 'use_encoder', False)
            transformer_dim = getattr(cfg.model, 'transformer_dim', 64)
        else:
            nblocks, nneighbor, blocks = 3, 16, [1, 1, 1, 1]
            num_samples, spike_mode, timestep, use_encoder, transformer_dim = 256, 'lif', 4, False, 64
        
        d_points = getattr(cfg, 'input_dim', 4)  

        assert len(blocks) == nblocks+1, "Block mismatches"

        self.fc1 = nn.Sequential(
            nn.Conv1d(d_points, 32, 1),
            nn.BatchNorm1d(32),
            build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU(),
            nn.Conv1d(32, 32, 1), 
            nn.BatchNorm1d(32),
        )

        transblock = lambda channel: TransformerBlock(channel, transformer_dim, nneighbor, timestep, spike_mode, use_encoder)
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
            nblocks = getattr(cfg.model, 'nblocks', 3)
            spike_mode = getattr(cfg.model, 'spike_mode', 'lif')
            timestep = getattr(cfg.model, 'timestep', 4)
            use_encoder = getattr(cfg.model, 'use_encoder', False)
            num_samples = getattr(cfg.model, 'num_samples', 256)
        else:
            nblocks, spike_mode, timestep, use_encoder, num_samples = 3, 'lif', 4, False, 256

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
            nn.Conv1d(64, 6, 1),
        )
        
        self.nblocks = nblocks
        self.T = timestep
        self.spike_mode  = spike_mode
        self.use_encoder = use_encoder
        self.num_samples = max(npoints//self.T, num_samples)

    def queue_SDE(self, x):
        def queue_mask(loc, fps_idx):
            B = loc.shape[0]
            mask = torch.ones_like(loc, dtype=torch.bool)
            mask[torch.arange(B).unsqueeze(1), fps_idx] = False    
            loc = loc[mask].view(B, -1, 3)
            return loc
        
        B, N, C = x.shape
        loc = x[...,:3]
        npoint = self.num_samples
        res = (N - npoint)//(self.T-1) if self.T != 1 else 0

        onion = torch.zeros(self.T, B, npoint, C).to(x.device)
        fps_idx = farthest_point_sample(loc, npoint)
        onion[0] = index_points(x, fps_idx)
        loc = queue_mask(loc, fps_idx)    

        for i in range(1, self.T):
            if loc.shape[1] == 0: onion[i] = onion[i-1]
            else:
                fps_idx = farthest_point_sample(loc, res)   
                onion[i, :, :npoint - res] = onion[i - 1][:, res:]
                onion[i, :, npoint - res:] = index_points(x, fps_idx)
                loc = queue_mask(loc, fps_idx)
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
