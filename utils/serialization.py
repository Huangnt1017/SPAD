"""
Point Cloud Serialization Utils

从 Pointcept 仓库复现的序列化工具:
- Z-order (Morton) 编码/解码
- Hilbert 曲线编码/解码
- 序列化/反序列化函数

用于 Point Transformer V3 的序列化注意力机制。

Author: Adapted from Pointcept (Xiaoyang Wu, Peng-Shuai Wang)
"""

import torch


# ============================================================================
# Z-Order (Morton) 编码 — 基于 OCNN 实现
# ============================================================================

class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z


_key_lut = KeyLUT()


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    """Z-order (Morton) 编码"""
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    EX, EY, EZ = _key_lut.encode_lut(x.device)
    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key
    return key


def z_order_decode(code: torch.Tensor, depth: int = 16):
    """Z-order 解码"""
    DX, DY, DZ = _key_lut.decode_lut(code.device)
    x, y, z = torch.zeros_like(code), torch.zeros_like(code), torch.zeros_like(code)
    b = code >> 48
    code = code & ((1 << 48) - 1)
    n = (depth + 2) // 3
    for i in range(n):
        k = code >> (i * 9) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))
    return x, y, z, b


# ============================================================================
# Hilbert 曲线编码/解码
# ============================================================================

def right_shift(binary, k=1, axis=-1):
    """沿指定轴右移二进制数组"""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )
    return shifted


def binary2gray(binary, axis=-1):
    """二进制 → Gray 码"""
    shifted = right_shift(binary, axis=axis)
    return torch.logical_xor(binary, shifted)


def gray2binary(gray, axis=-1):
    """Gray 码 → 二进制"""
    shift = 2 ** (torch.Tensor([gray.shape[axis]]).log2().ceil().int() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = torch.div(shift, 2, rounding_mode="floor")
    return gray


def hilbert_encode(locs: torch.Tensor, num_dims: int = 3, num_bits: int = 16):
    """Hilbert 曲线编码"""
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8).to(locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(f"Last dim {orig_shape[-1]} != num_dims {num_dims}")

    if num_dims * num_bits > 63:
        raise ValueError(f"num_dims={num_dims} num_bits={num_bits} too many bits")

    locs_uint8 = locs.long().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = torch.logical_xor(
                gray[:, 0, bit + 1:], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]),
            )
            gray[:, dim, bit + 1:] = torch.logical_xor(
                gray[:, dim, bit + 1:], to_flip
            )
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))
    hh_bin = gray2binary(gray)
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).squeeze().type(torch.uint8)
    )
    hh_uint64 = hh_uint8.view(torch.int64).squeeze()
    return hh_uint64


def hilbert_decode(hilberts: torch.Tensor, num_dims: int = 3, num_bits: int = 16):
    """Hilbert 曲线解码"""
    bitpack_mask = 1 << torch.arange(0, 8).to(hilberts.device)
    hh_uint8 = hilberts.view(torch.uint8).reshape((-1, 8)).flip(-1)
    gray = (
        (hh_uint8.unsqueeze(-1).bitwise_and(bitpack_mask).ne(0))
        .byte()
        .flatten(-2, -1)
    )
    gray = gray2binary(gray)
    # 截取有效位
    gray = gray[..., -num_dims * num_bits:]

    for bit in range(num_bits - 1, -1, -1):
        for dim in range(num_dims - 1, -1, -1):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = torch.logical_xor(
                gray[:, 0, bit + 1:], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]),
            )
            gray[:, dim, bit + 1:] = torch.logical_xor(
                gray[:, dim, bit + 1:], to_flip
            )
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)

    # 将 Gray 码转回坐标
    locs_bits = gray[..., -num_bits * num_dims:]
    locs_uint8 = torch.zeros(
        hilberts.shape[0], 8 * num_dims, device=hilberts.device, dtype=torch.uint8
    )
    for i in range(num_bits):
        for d in range(num_dims):
            locs_uint8[:, d * 8 + (num_bits - 1 - i) // 8] |= (
                locs_bits[:, d * num_bits + i].byte() << ((num_bits - 1 - i) % 8)
            )
    locs = locs_uint8.view(torch.int64).reshape(-1, num_dims)
    return locs


# ============================================================================
# 通用编码/解码接口 (Pointcept 风格)
# ============================================================================

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    """对点云网格坐标进行序列化编码

    Args:
        grid_coord: (N, 3) int64 网格坐标
        batch: (N,) int64 batch 索引
        depth: 编码深度
        order: "z" | "z-trans" | "hilbert" | "hilbert-trans"

    Returns:
        code: (N,) int64 序列化编码
    """
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, num_dims=3, num_bits=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], num_dims=3, num_bits=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    """反序列化解码"""
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        x, y, z, _ = z_order_decode(code, depth=depth)
        grid_coord = torch.stack([x, y, z], dim=-1)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, num_dims=3, num_bits=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch
