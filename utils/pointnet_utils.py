"""PointNet utilities — consolidated geometric operations for all baselines.

Central hub for shared point cloud geometric functions:
- square_distance, index_points, farthest_point_sample, knn_point, query_ball_point
- fps (alias), fps_points, knn_point_with_break_tie, grouping, interpolation
- farthest_point_sample_varlen, knn_point_varlen, knn_query_and_group
- LayerNorm1d, offset/batch utils, off_diagonal
- PointNetSetAbstraction, sample_and_group
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode, MultiStepEIFNode,
        MultiStepParametricLIFNode, MultiStepIFNode
    )
except ImportError:
    import logging
    logging.warning("Please install spikingjelly: pip install spikingjelly")


# ============================================================================
# 基础几何辅助函数
# ============================================================================

def square_distance(src, dst):
    """欧氏距离平方: src[B,N,C] dst[B,M,C] → [B,N,M]"""
    if src.dim() == 4 and dst.dim() == 4:
        return torch.sum((src[:, :, :, None] - dst[:, :, None]) ** 2, dim=-1)
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """根据索引收集点: points[B,N,C] idx[B,S]或[B,S,K] → [B,S,C]或[B,S,K,C]"""
    idx = idx.to(torch.int64)
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], raw_size[1], -1) if len(raw_size) == 4 else idx.reshape(raw_size[0], -1)
    idx_size = list(idx.shape)
    idx_size.append(points.size(-1))
    res = torch.gather(points, 2 if len(raw_size) == 4 else 1, idx[..., None].expand(*idx_size))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """最远点采样 (FPS): xyz[B,N,3] → centroids[B,npoint] (索引)"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


# Alias for compatibility
fps = farthest_point_sample


def fps_points(xyz, npoint):
    """FPS 采样并直接返回坐标: xyz[B,N,3] → [B,npoint,3]"""
    idx = farthest_point_sample(xyz, npoint)
    return index_points(xyz, idx)


def knn_point(k, xyz, new_xyz):
    """kNN 搜索: xyz[B,N,C] new_xyz[B,S,C] → idx[B,S,k]"""
    dist = square_distance(new_xyz, xyz)
    _, idx = dist.topk(k, dim=-1, largest=False, sorted=False)
    return idx


def knn_point_with_break_tie(k, xyz, new_xyz):
    """带微小噪声的 kNN，用于避免平局情况。"""
    dist = square_distance(new_xyz, xyz)
    noise = torch.rand_like(dist) * 1e-8
    idx = (dist + noise).topk(k=k, dim=-1, largest=False)[1]
    return idx


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball Query: 以 new_xyz 为中心、半径 radius 内的近邻点索引。"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def grouping(feat, idx):
    """根据索引分组: feat[N,C] idx[N,K] → [N,K,C] (纯 PyTorch gather)"""
    N, K = idx.shape
    C = feat.shape[1]
    idx_flat = idx.reshape(-1).unsqueeze(-1).expand(-1, C)
    grouped = feat.gather(0, idx_flat)
    return grouped.reshape(N, K, C)


def grouping_with_xyz(xyz, feat, idx):
    """分组并拼接坐标: xyz[N,3] feat[N,C] idx[N,K] → pos[N,K,3], grouped_feat[N,K,C]"""
    grouped_xyz = grouping(xyz, idx)
    grouped_feat = grouping(feat, idx)
    return grouped_xyz, grouped_feat


def interpolation(unknown_xyz, known_xyz, known_feat, n_unknown, n_known, k=3):
    """反距离加权插值: 用于上采样点特征。"""
    dist = square_distance(unknown_xyz, known_xyz)
    dist, idx = dist.topk(k=k, dim=-1, largest=False)
    dist_recip = 1.0 / (dist + 1e-8)
    norm = torch.sum(dist_recip, dim=-1, keepdim=True)
    weight = dist_recip / norm
    interpolated_feat = torch.sum(
        index_points(known_feat, idx) * weight.unsqueeze(-1), dim=2
    )
    return interpolated_feat


# ============================================================================
# 变长点云版本 (使用 offset 约定)
# ============================================================================

def farthest_point_sample_varlen(xyz, offset, new_offset):
    """变长点云的 FPS。"""
    device = xyz.device
    npoints_new = new_offset[-1].item()
    new_xyz_idx = torch.zeros(npoints_new, dtype=torch.long, device=device)
    for i in range(len(offset)):
        s_i = 0 if i == 0 else offset[i - 1].item()
        e_i = offset[i].item()
        s_new = 0 if i == 0 else new_offset[i - 1].item()
        e_new = new_offset[i].item()
        npoint = e_new - s_new
        if npoint <= 0 or e_i - s_i <= 0:
            continue
        batch_xyz = xyz[s_i:e_i, :].unsqueeze(0)
        idx = farthest_point_sample(batch_xyz, npoint)
        new_xyz_idx[s_new:e_new] = idx[0] + s_i
    return new_xyz_idx


def knn_point_varlen(k, xyz, offset, new_xyz, new_offset):
    """变长点云的 kNN。"""
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
        batch_xyz = xyz[s_i:e_i, :].unsqueeze(0)
        batch_new_xyz = new_xyz[s_new:e_new, :].unsqueeze(0)
        batch_idx = knn_point(k, batch_xyz, batch_new_xyz)
        idx[s_new:e_new, :] = batch_idx[0]
    return idx


def knn_query_and_group(feat, xyz, offset, new_xyz=None, new_offset=None,
                        nsample=16, idx=None, with_xyz=True):
    """Pointcept 风格的 knn_query_and_group。"""
    if new_xyz is None:
        new_xyz = xyz
        new_offset = offset
    if idx is None:
        idx = knn_point_varlen(nsample, xyz, offset, new_xyz, new_offset)
    if with_xyz:
        grouped_xyz = new_xyz.unsqueeze(1).expand(-1, nsample, -1)
        grouped_feat = feat[idx.long(), :]
        return torch.cat([grouped_xyz, grouped_feat], dim=-1)
    else:
        return feat[idx.long(), :]


# ============================================================================
# LayerNorm1d (Point Transformer V1 使用)
# ============================================================================

class LayerNorm1d(nn.Module):
    """对点云特征进行 LayerNorm (包装 BatchNorm1d)。"""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.norm = nn.BatchNorm1d(normalized_shape, eps=eps, affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        elif x.dim() == 2:
            return self.norm(x.t()).t()
        else:
            raise NotImplementedError(f"LayerNorm1d: unsupported dim {x.dim()}")


# ============================================================================
# Batch / Offset 转换
# ============================================================================

@torch.no_grad()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.no_grad()
def bincount2offset(bincount):
    return torch.cumsum(bincount, dim=0)


@torch.no_grad()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()


def off_diagonal(x):
    """返回方阵非对角元素的扁平视图。"""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# ============================================================================
# Sample & Group (SPT 使用)
# ============================================================================

def sample_and_group(npoint, radius, nsample, xyz, points, use_encoder, returnfps=False, knn=False):
    T, B, N, C = xyz.shape
    S = npoint
    loc = xyz[0] if not use_encoder else xyz.flatten(0, 1)

    fps_idx = farthest_point_sample(loc, npoint)

    torch.cuda.empty_cache()
    new_xyz = index_points(loc, fps_idx)
    new_xyz = new_xyz.repeat(T, 1, 1, 1) if not use_encoder else new_xyz.view(T, B, -1, C)

    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)
        idx = dists.argsort()[:, :, :, :nsample]
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(T, B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    T, B, N, C = xyz.shape
    new_xyz = torch.zeros(T, B, 1, C).to(device)
    grouped_xyz = xyz.view(T, B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(T, B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


# ============================================================================
# Spike-related code (SPT model)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from spikingjelly.clock_driven.neuron import (
        MultiStepLIFNode, MultiStepEIFNode,
        MultiStepParametricLIFNode, MultiStepIFNode
    )
except ImportError:
    import logging
    logging.warning("Please install spikingjelly: pip install spikingjelly")


class HeavisideSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        sigmoid_grad = torch.sigmoid(input) * (1 - torch.sigmoid(input))
        return grad_output * sigmoid_grad


class HeavisideParametricSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, v_threshold):
        ctx.save_for_backward(input)
        ctx.k = k
        ctx.v_threshold = v_threshold
        return (input >= v_threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        k = ctx.k
        v_threshold = ctx.v_threshold
        sigmoid_grad = k * torch.sigmoid(k * (input - v_threshold)) * (1 - torch.sigmoid(k * (input - v_threshold)))
        return grad_output * sigmoid_grad, None, None


heaviside_sigmoid = HeavisideSigmoid.apply
heaviside_parametric_sigmoid = HeavisideParametricSigmoid.apply


def square_distance(src, dst):
    if src.dim() == 4 and dst.dim() == 4:
        return torch.sum((src[:, :, :, None] - dst[:, :, None]) ** 2, dim=-1)
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    idx = idx.to(torch.int64)
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], raw_size[1], -1) if len(raw_size) == 4 else idx.reshape(raw_size[0], -1)
    idx_size = list(idx.shape)
    idx_size.append(points.size(-1))
    res = torch.gather(points, 2 if len(raw_size) == 4 else 1, idx[..., None].expand(*idx_size))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, use_encoder, returnfps=False, knn=False):
    T, B, N, C = xyz.shape
    S = npoint
    loc = xyz[0] if not use_encoder else xyz.flatten(0, 1)

    fps_idx = farthest_point_sample(loc, npoint)

    torch.cuda.empty_cache()
    new_xyz = index_points(loc, fps_idx)
    new_xyz = new_xyz.repeat(T, 1, 1, 1) if not use_encoder else new_xyz.view(T, B, -1, C)

    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)
        idx = dists.argsort()[:, :, :, :nsample]
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(T, B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    T, B, N, C = xyz.shape
    new_xyz = torch.zeros(T, B, 1, C).to(device)
    grouped_xyz = xyz.view(T, B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(T, B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class MoELIFNode(nn.Module):
    def __init__(self, timestep, spike_mode, input_dim, tau=2.0, v_threshold=0.2):
        super().__init__()
        self.spike_mode = spike_mode
        self.num_experts = len(spike_mode)
        self.tau = tau
        self.v_threshold = v_threshold
        self.k = 4
        self.T = timestep
        self.v_th = 0.2

        self.experts = nn.ModuleList([])
        self.gate = nn.Conv1d(input_dim * self.T, self.num_experts, 1)
        for i in range(self.num_experts):
            if spike_mode[i] == "lif":
                self.experts.append(MultiStepLIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy"))
            elif spike_mode[i] == "elif":
                self.experts.append(MultiStepEIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy"))
            elif spike_mode[i] == "plif":
                self.experts.append(MultiStepParametricLIFNode(timestep=timestep, v_threshold=v_threshold, init_tau=tau, detach_reset=True, backend="cupy"))
            elif spike_mode[i] == "if":
                self.experts.append(MultiStepIFNode(timestep=timestep, v_threshold=v_threshold, detach_reset=True, backend="cupy"))

    def forward(self, x):
        if self.training:
            z = x.view(-1, self.T, *x.shape[1:]).flatten(1, 2)
            gate = F.softmax(self.gate(z), dim=-2).repeat(self.T, 1, 1, 1)
            _ = torch.stack([expert(x) for expert in self.experts], dim=1)
            expert_outputs = torch.stack([expert.v_seq.flatten(0, 1) for expert in self.experts], dim=1).view(self.T, -1, self.num_experts, *x.shape[1:])
            expert_outputs[expert_outputs == 0.0] = self.v_threshold
            output = torch.sum(gate.unsqueeze(3) * expert_outputs, dim=2)
            output = heaviside_parametric_sigmoid(output, self.k, self.v_th)
        else:
            z = x.view(-1, self.T, *x.shape[1:]).flatten(1, 2)
            gate = self.gate(z)
            topk_values, topk_indices = torch.topk(gate, 2, dim=-2)

            gate = torch.zeros_like(gate)
            gate.scatter_(-2, topk_indices, topk_values)
            gate_masked = gate.clone()
            gate_masked[gate == 0] = float("-inf")
            gate = F.softmax(gate_masked, dim=-2).repeat(self.T, 1, 1, 1)

            _ = torch.stack([expert(x) for expert in self.experts], dim=1)
            expert_outputs = torch.stack([expert.v_seq.flatten(0, 1) for expert in self.experts], dim=1).view(self.T, -1, self.num_experts, *x.shape[1:])
            expert_outputs[expert_outputs == 0.0] = self.v_threshold
            output = torch.sum(gate.unsqueeze(3) * expert_outputs, dim=2)
            output = heaviside_parametric_sigmoid(output, self.k, self.v_th)
        return output.flatten(0, 1)


def build_spike_node(timestep, spike_mode, input_dim=None, tau=2.0, v_threshold=0.5):
    if spike_mode == "lif":
        proj_lif = MultiStepLIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "elif":
        proj_lif = MultiStepEIFNode(timestep=timestep, v_threshold=v_threshold, tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "plif":
        proj_lif = MultiStepParametricLIFNode(timestep=timestep, v_threshold=v_threshold, init_tau=tau, detach_reset=True, backend="cupy")
    elif spike_mode == "if":
        proj_lif = MultiStepIFNode(timestep=timestep, v_threshold=v_threshold, detach_reset=True, backend="cupy")
    elif isinstance(spike_mode, list) and input_dim is not None:
        proj_lif = MoELIFNode(timestep, spike_mode, input_dim, v_threshold=v_threshold)
    else:
        raise ValueError(f"Unsupported spike_mode: {spike_mode}")
    return proj_lif


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, timestep, spike_mode, use_encoder, group_all, knn=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.use_encoder = use_encoder
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp_lifs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            self.mlp_lifs.append(build_spike_node(timestep, spike_mode) if spike_mode is not None else nn.ReLU())
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, self.use_encoder, knn=self.knn)

        T, B, N, M, _ = new_points.shape
        new_points = new_points.permute(0, 1, 4, 3, 2).flatten(0, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = bn(conv(self.mlp_lifs[i](new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        new_points = new_points.reshape(T, B, N, -1)
        return new_xyz, new_points
