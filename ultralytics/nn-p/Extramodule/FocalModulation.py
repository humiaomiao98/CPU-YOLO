import torch.nn as nn
from timm.models.layers import DropPath
import torch
import torch.nn.functional as F

__all__ = ['FocalModulation']


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FocalModulation(nn.Module):
    # def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
    #              use_postln_in_modulation=False, normalize_modulator=False):
    #     super().__init__()

    def __init__(self, dim, focal_window=3, focal_level=2, focal_factor=2, bias=True, proj_drop=0.,
                 use_postln_in_modulation=False, normalize_modulator=False):
        super().__init__()

        # 如果参数是通过列表传递的（如YAML配置）
        if isinstance(dim, list):
            dim = dim[0]
            if len(dim) > 1:
                focal_window = dim[1]
            if len(dim) > 2:
                focal_level = dim[2]

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        # 修改点1：调整输出通道数为 2*dim + focal_level + 1
        self.f_linear = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), kernel_size=1, bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              groups=dim, padding=kernel_size // 2, bias=False),
                    nn.GELU(),
                )
            )
            self.kernel_sizes.append(kernel_size)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        B, C, H, W = x.shape

        # 确保输出尺寸与输入相同
        x_out = ...  # 你的计算逻辑

        # 添加尺寸检查
        if x_out.shape[-2:] != (H, W):
            x_out = F.interpolate(x_out, size=(H, W), mode='bilinear', align_corners=False)

        return x_out

        # pre linear projection
        x_proj = self.f_linear(x)
        # 确保split的维度正确
        q, ctx, gates = torch.split(x_proj, [self.dim, self.dim, self.focal_level + 1], dim=1)

        # context aggregation
        ctx_all = 0.0
        for l in range(self.focal_level):
            ctx_conv = self.focal_layers[l](ctx)
            # 确保gates切片维度匹配
            gate_slice = gates[:, l:l + 1]
            if ctx_conv.shape[-2:] != gate_slice.shape[-2:]:
                gate_slice = F.interpolate(gate_slice, size=ctx_conv.shape[-2:], mode='bilinear', align_corners=False)
            ctx_all = ctx_all + ctx_conv * gate_slice

        # 全局上下文
        ctx_global = self.act(ctx.mean(dim=[2, 3], keepdim=True))
        gate_global = gates[:, self.focal_level:]
        if ctx_global.shape[-2:] != gate_global.shape[-2:]:
            gate_global = F.interpolate(gate_global, size=ctx_global.shape[-2:], mode='bilinear', align_corners=False)
        ctx_all = ctx_all + ctx_global * gate_global

        # 确保q和ctx_all的尺寸匹配
        if q.shape[-2:] != ctx_all.shape[-2:]:
            q = F.interpolate(q, size=ctx_all.shape[-2:], mode='bilinear', align_corners=False)

        # focal modulation
        ctx_all = self.h(ctx_all)
        if ctx_all.shape[-2:] != q.shape[-2:]:
            ctx_all = F.interpolate(ctx_all, size=q.shape[-2:], mode='bilinear', align_corners=False)

        x_out = q * ctx_all

        # post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

    # def forward(self, x):
    #     """
    #     Args:
    #         x: input features with shape of (B, C, H, W)
    #     """
    #     B, C, H, W = x.shape
    #
    #     # pre linear projection
    #     x_proj = self.f_linear(x)
    #     q, ctx, gates = torch.split(x_proj, [self.dim, self.dim, self.focal_level + 1], dim=1)
    #
    #     # context aggregation
    #     ctx_all = 0.0
    #     for l in range(self.focal_level):
    #         ctx_conv = self.focal_layers[l](ctx)
    #         # 修改点2：确保 gates 切片维度匹配
    #         gate_slice = gates[:, l:l+1]
    #         if ctx_conv.shape[-2:] != gate_slice.shape[-2:]:
    #             gate_slice = F.interpolate(gate_slice, size=ctx_conv.shape[-2:], mode='bilinear', align_corners=False)
    #         ctx_all = ctx_all + ctx_conv * gate_slice
    #
    #     ctx_global = self.act(ctx.mean(dim=[2, 3], keepdim=True))
    #     # 修改点3：确保全局上下文维度匹配
    #     gate_global = gates[:, self.focal_level:]
    #     if ctx_global.shape[-2:] != gate_global.shape[-2:]:
    #         gate_global = F.interpolate(gate_global, size=ctx_global.shape[-2:], mode='bilinear', align_corners=False)
    #     ctx_all = ctx_all + ctx_global * gate_global
    #
    #     # normalize context
    #     if self.normalize_modulator:
    #         ctx_all = ctx_all / (self.focal_level + 1)
    #
    #     # focal modulation
    #     x_out = q * self.h(ctx_all)
    #     if self.use_postln_in_modulation:
    #         x_out = self.ln(x_out)
    #
    #     # post linear projection
    #     x_out = self.proj(x_out)
    #     x_out = self.proj_drop(x_out)
    #     return x_out


class FocalModulationBlock(nn.Module):
    """ Focal Modulation Block.
    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 focal_level=2, focal_window=9, use_layerscale=False, layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_layerscale = use_layerscale

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim, focal_window=focal_window, focal_level=focal_level, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # FM
        x = self.modulation(x).permute(0, 2, 3, 1).view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x