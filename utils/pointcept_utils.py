"""
Pointcept 专用工具

唯一内容:
- 从 utils.pointnet_utils 导入通用几何函数和 LayerNorm1d
- 保留本模块仅用于保持向后兼容

所有基础几何函数已迁移至 utils.pointnet_utils。
"""

from utils.pointnet_utils import (
    LayerNorm1d,
    farthest_point_sample, index_points, knn_point, square_distance,
    knn_point_with_break_tie, grouping, grouping_with_xyz, interpolation,
    farthest_point_sample_varlen, knn_point_varlen, knn_query_and_group,
    off_diagonal,
    offset2bincount, bincount2offset, offset2batch, batch2offset,
)
