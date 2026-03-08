import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse
from .diff_cross_attns import SpiralAware_CrossDeformAttn2D # SFS Module

class ConvModule(nn.Module):
    """Simplified ConvModule (Conv -> BN -> ReLU)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, norm=True, activation=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.ReLU(inplace=False))
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)

class ConvDWT(nn.Module):  # DWT: (B,C,H,W) -> (B,4C,H/2,W/2), no parameters are learnable
    def __init__(self, wave='haar', mode='zero'):
        super(ConvDWT, self).__init__()
        # one-level DWT
        self.dwt_forward = DWTForward(J=1, wave=wave, mode=mode)

    def forward(self, x):
        # input size: x (B, C, H, W)
        with torch.cuda.amp.autocast(enabled=False):
            if x.dtype != torch.float32:
                x = x.float()
            Yl, Yh = self.dwt_forward(x)
        b, c, h, w = x.shape
        # Yl (B, C, H/2, W/2) for low-frequency LL
        # List Yh for high-frequency from each level of DWT
        # Yh[0] (B, C, 3, H/2, W/2) for high-frequency LH,HL,HH

        Yh = Yh[0].transpose(1, 2).reshape(Yh[0].shape[0], -1, Yh[0].shape[3], Yh[0].shape[4])

        # output size: output (B, 4C, H/2, W/2)
        output = torch.cat((Yl, Yh), dim=1)
        output = F.interpolate(output, size=(h // 2, w // 2), mode='bilinear', align_corners=False)
        return output

class ConvIDWT(nn.Module):  # IDWT
    def __init__(self, wave='haar', mode='zero'):
        super(ConvIDWT, self).__init__()
        self.dwt_inverse = DWTInverse(wave=wave, mode=mode)

    def forward(self, low_freqs, high_freqs):
        # low_freqs: (B, C, H/2, W/2)
        # high_freqs: (B, 3C, H/2, W/2)
        B, C, H, W = low_freqs.shape

        high_freqs = high_freqs.reshape(B, C, 3, H, W)

        with torch.cuda.amp.autocast(enabled=False):
            reconstruction = self.dwt_inverse((low_freqs, [high_freqs.float()]))
        reconstruction = F.interpolate(reconstruction, size=(2 * H, 2 * W), mode='bilinear', align_corners=False)

        return reconstruction

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, bn_before_sigmoid=False):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.bn_before_sigmoid = bn_before_sigmoid
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        if bn_before_sigmoid:
            self.bn = nn.BatchNorm2d(1)
            self.bn.bias.data.fill_(0)
            self.bn.bias.requires_grad = False
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        if self.bn_before_sigmoid:
            x = self.bn(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class LearnableGaussianFilterBank(nn.Module):
    def __init__(self, kernel_size, num_filters, num_channels):
        super(LearnableGaussianFilterBank, self).__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.C = num_channels
        self.padding = kernel_size // 2  # Padding size to maintain input size

        # Create learnable parameters for sigmas
        self.sigmas = nn.ParameterList([nn.Parameter(torch.tensor([1.0])) for _ in range(num_filters)])

    def forward(self, x):
        # Apply Gaussian filters using convolution
        weights = [self._gaussian_kernel(self.kernel_size, sigma).repeat(self.C, 1, 1, 1) for sigma in self.sigmas]
        filtered_outputs = [F.conv2d(F.pad(x, (self.padding,self.padding,self.padding,self.padding), mode='replicate')
                                     , weight.to(x.device), groups=self.C) for weight in weights]
        return torch.cat(filtered_outputs, dim=1)

    def _gaussian_kernel(self, kernel_size, sigma):
        # Create a 2D Gaussian kernel with learnable sigma as tensors
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[:, :, i, j] = torch.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))

        return kernel / kernel.sum() # normalization

# LFP Module:
class wav_Enhance(nn.Module): # Low-frequency Guided Feature Purification
    def __init__(self, in_channels, wave='haar', mode='symmetric', with_gauss=True, gauss_gate=0.5):
        super(wav_Enhance, self).__init__()
        self.dwt = ConvDWT(wave=wave, mode=mode)
        self.idwt = ConvIDWT(wave=wave, mode=mode)
        self.with_gauss = with_gauss
        self.gauss_gate = gauss_gate

        self.attention = SpatialAttention()
        if self.with_gauss:
            self.gaussian_filter = LearnableGaussianFilterBank(kernel_size=3, num_filters=1, num_channels=3 * in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        dwt_out = self.dwt(x)  # (B, 4C, H/2, W/2)

        LL = dwt_out[:, :C, :, :]
        Yh = dwt_out[:, C:, :, :]

        # low-frequency guided modulation of high-frequency
        att = self.attention(LL)  # (B, 1, H/2, W/2)
        Yh = Yh * att

        if self.with_gauss: # Gaussian Filter for high-frequency
            Yh_blurred = self.gaussian_filter(Yh)
            mask = (Yh.abs() < self.gauss_gate).float()
            Yh = Yh * (1 - mask) + Yh_blurred * mask

        x_rec = self.idwt(LL, Yh) # (B, C, H, W)
        return x_rec

class NS_FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 upsample_mode='nearest',
                 use_wav_enhance=True, use_crossattn_topdown=True):
        super(NS_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.relu_before_extra_convs = relu_before_extra_convs
        self.upsample_mode = upsample_mode
        self.relu = nn.ReLU(inplace=False)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
        assert self.backbone_end_level <= self.num_ins
        self.add_extra_convs = add_extra_convs

        self.use_wav_enhance = use_wav_enhance
        self.use_crossattn_topdown = use_crossattn_topdown

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.wavenhance_list = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(in_channels[i], out_channels, 1)
            fpn_conv = ConvModule(out_channels, out_channels, 3, padding=1)

            if self.use_wav_enhance and i < 4:
                wavenhance = wav_Enhance(out_channels, wave='haar', mode='zero')
            else:
                wavenhance = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.wavenhance_list.append(wavenhance)

        if self.use_crossattn_topdown:
            self.crossattn_list = nn.ModuleList()
            for i in range(self.start_level, self.backbone_end_level - 1):
                crossattn = SpiralAware_CrossDeformAttn2D(dim=out_channels, n_heads=8, n_points=4)
                self.crossattn_list.append(crossattn)

        extra_levels = num_outs - (self.backbone_end_level - self.start_level)
        if self.add_extra_convs and extra_levels >= 1:
            self.extra_convs = nn.ModuleList()
            for i in range(extra_levels):
                in_c = self.in_channels[self.backbone_end_level - 1] if i == 0 else out_channels
                self.extra_convs.append(
                    ConvModule(in_c, out_channels, 3, stride=2, padding=1)
                )
        else:
            self.extra_convs = None

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = []
        for i in range(len(inputs) - self.start_level):
            lateral = self.lateral_convs[i](inputs[i + self.start_level])
            lateral = self.wavenhance_list[i](lateral)
            laterals.append(lateral)

        if self.use_crossattn_topdown:
            # top-down path with SFS modules
            for i in range(len(laterals) - 1, 0, -1):
                laterals[i - 1] = self.crossattn_list[i - 1](laterals[i - 1], laterals[i])
        else:
            # top-down path with general laterals
            for i in range(len(laterals) - 1, 0, -1):
                upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode=self.upsample_mode)
                laterals[i - 1] = laterals[i - 1] + upsampled

        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]

        # Add extra conv layers (if needed)
        if self.num_outs > len(outs):
            if self.extra_convs is None:
                for _ in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], kernel_size=1, stride=2))
            else:
                x = inputs[self.backbone_end_level - 1] if self.add_extra_convs else outs[-1]
                for conv in self.extra_convs:
                    if self.relu_before_extra_convs:
                        x = self.relu(x)
                    x = conv(x)
                    outs.append(x)

        return outs