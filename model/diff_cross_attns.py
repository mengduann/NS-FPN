import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from SFS_MSDeformAttn.ops.modules import MSDeformAttn_for_sfs

def generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0):
    """
    Initialization of spiral-aware sampling pattern.

    parameters:
    - n_heads: number of attention heads
    - n_points: number of sampling points of each head
    - n_levels: number of feature levels, default=1
    - base_radius: initial radius of sampling point
    - radius_step: radial step between consecutive points of each head

    return:
    - grid: Tensor, [n_heads, n_levels, n_points, 2]
    """
    offsets = []
    for h in range(n_heads):
        head_offsets = []
        delta_theta = 2 * math.pi * h / n_heads  # initial angle of each head
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points + delta_theta
            r = base_radius + i * radius_step
            dx = r * math.cos(theta)
            dy = r * math.sin(theta)
            head_offsets.append([dx, dy])
        offsets.append(head_offsets)

    grid = torch.tensor(offsets, dtype=torch.float32)
    grid = grid.unsqueeze(1).repeat(1, n_levels, 1, 1)  # [n_heads, n_levels, n_points, 2]
    return grid

# SFS Module:
class SpiralAware_CrossDeformAttn2D(nn.Module):
    """
    Spiral-Aware MSDeformAttn.

    Inputs:
        - query_feat: [B, C, H1, W1], larger scale feature maps
        - key_feat:   [B, C, H2, W2], smaller scale feature maps
    Output:
        - out:   [B, C, H1, W1]
    """
    def __init__(self, dim, n_heads=8, n_points=4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points

        self.query_Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.key_Conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.shared_offsets_residual = nn.Parameter(torch.zeros(n_heads, n_points, 2))
        # generate uniform spiral-aware sampling pattern, and register it as buffer
        fixed_bias = generate_structured_grid(n_heads, n_points, n_levels=1, base_radius=1.0, radius_step=1.0)
        self.register_buffer("offset_base", fixed_bias.view(1, 1, n_heads, 1, n_points, 2))

        # LayerNorm on flattened features
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.out_norm = nn.LayerNorm(dim)

        self.attn = MSDeformAttn_for_sfs(
            d_model=dim,
            n_levels=1,
            n_heads=n_heads,
            n_points=n_points
        )

    def forward(self, query_feat: Tensor, key_feat: Tensor) -> Tensor:
        B, C, H1, W1 = query_feat.shape
        _, _, H2, W2 = key_feat.shape

        query_feat = self.query_Conv(query_feat)
        key_feat = self.key_Conv(key_feat)

        offsets_residual = self.shared_offsets_residual

        shared_offsets = self.offset_base.view(self.n_heads, 1, self.n_points, 2) + offsets_residual.view(self.n_heads, 1, self.n_points, 2)
        offsets = shared_offsets.view(1, 1, self.n_heads, 1, self.n_points, 2).expand(B, H1 * W1, -1, -1, -1, -1)

        # flatten & transpose to [B, HW, C]
        query = query_feat.flatten(2).transpose(1, 2)       # [B, H1*W1, C]
        kv = key_feat.flatten(2).transpose(1, 2)            # [B, H2*W2, C]
        query = self.query_norm(query)
        kv = self.key_norm(kv)

        spatial_shapes = torch.tensor([[H2, W2]], device=key_feat.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=key_feat.device, dtype=torch.long)

        # generate normalized reference points for each query position, both shapes: [H1, W1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / H1, 1 - 0.5 / H1, H1, device=query_feat.device),
            torch.linspace(0.5 / W1, 1 - 0.5 / W1, W1, device=query_feat.device),
            indexing='ij'
        )
        reference_points = torch.stack((grid_x, grid_y), -1)  # [H1, W1, 2]
        reference_points = reference_points.view(1, H1 * W1, 1, 2).repeat(B, 1, 1, 1)  # [B, H1*W1, 1, 2]

        # run deformable attention
        attn = self.attn(
            query=query,
            reference_points=reference_points,
            input_flatten=kv,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            sampling_offsets=offsets
        )  # [B, H1*W1, C]

        out = query + query * attn
        out = self.out_norm(out).transpose(1, 2).reshape(B, C, H1, W1)
        return out