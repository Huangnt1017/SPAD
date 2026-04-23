import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Input:  x [B, 3, N]
    Output: logits [B, num_classes]
    """
    def __init__(self, num_classes=40, k=20, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
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

        self.conv5 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = get_graph_feature(x, k=self.k)   # [B, 6, N, k]
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
        x = torch.cat((x1, x2), dim=1)          # [B, emb_dims*2]

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DGCNNPartSeg(nn.Module):
    """
    DGCNN for part segmentation.
    Input:
      x         [B, 3, N]
      cls_label [B, num_categories] one-hot (e.g., 16 for ShapeNetPart)
    Output:
      seg_logits [B, num_parts, N]
    """
    def __init__(self, num_parts=50, num_categories=16, k=40, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.num_categories = num_categories

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(emb_dims)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(64 * 3, emb_dims, kernel_size=1, bias=False),
            self.bn6,
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(emb_dims + 64 * 3 + 64, 256, kernel_size=1, bias=False),
            self.bn7,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            self.bn8,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            self.bn9,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv10 = nn.Conv1d(128, num_parts, kernel_size=1, bias=True)

        self.cls_embed = nn.Sequential(
            nn.Conv1d(num_categories, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x, cls_label):
        B, _, N = x.size()

        x = get_graph_feature(x, k=self.k)    # [B, 6, N, k]
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1)[0]                 # [B, 64, N]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1)[0]                 # [B, 64, N]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1)[0]                 # [B, 64, N]

        x = torch.cat((x1, x2, x3), dim=1)    # [B, 192, N]
        x = self.conv6(x)                     # [B, emb_dims, N]
        x_global = F.adaptive_max_pool1d(x, 1)  # [B, emb_dims, 1]

        if cls_label.dim() == 2:
            cls_label = cls_label.unsqueeze(-1)  # [B, num_categories, 1]
        cls_feat = self.cls_embed(cls_label)     # [B, 64, 1]

        x_global = x_global.repeat(1, 1, N)      # [B, emb_dims, N]
        cls_feat = cls_feat.repeat(1, 1, N)      # [B, 64, N]

        x = torch.cat((x_global, x1, x2, x3, cls_feat), dim=1)  # [B, emb_dims+192+64, N]
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.conv10(x)                       # [B, num_parts, N]
        return x


def _quick_shape_test():
    """
    Minimal runtime sanity check.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cls_model = DGCNNCls(num_classes=40).to(device)
    pts = torch.randn(4, 3, 1024, device=device)
    cls_logits = cls_model(pts)
    print("DGCNNCls output:", cls_logits.shape)  # [4, 40]

    seg_model = DGCNNPartSeg(num_parts=50, num_categories=16).to(device)
    cls_onehot = torch.zeros(4, 16, device=device)
    cls_onehot[:, 0] = 1.0
    seg_logits = seg_model(pts, cls_onehot)
    print("DGCNNPartSeg output:", seg_logits.shape)  # [4, 50, 1024]


if __name__ == "__main__":
    _quick_shape_test()
