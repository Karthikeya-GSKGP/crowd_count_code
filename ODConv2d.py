import torch
import torch.nn as nn
import torch.nn.functional as F


class ODConv2d(nn.Module):
    """
    Fast ODConv2d:
    Mixes K kernels per-sample via attention, then uses grouped-conv trick.
    """
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        reduction: float = 0.0625,
        kernel_num: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        assert in_planes % groups == 0, "in_planes must be divisible by groups"
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.use_bias = bias

        hidden = max(1, int(in_planes * reduction))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, kernel_num, 1, bias=False),
        )

        # (K, out, in/groups, kh, kw)
        self.weight = nn.Parameter(
            torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size) * 0.02
        )

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(kernel_num, out_planes))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # attention: (B,K)
        a = self.fc(self.avgpool(x)).view(B, self.kernel_num)
        a = torch.softmax(a, dim=1)

        # mixed_w: (B, out, Cin/groups, kh, kw)
        mixed_w = torch.einsum("bk,kocij->bocij", a, self.weight)

        if self.use_bias:
            mixed_b = torch.einsum("bk,ko->bo", a, self.bias)
        else:
            mixed_b = None

        # grouped conv trick
        xg = x.reshape(1, B * C, H, W)
        wg = mixed_w.reshape(
            B * self.out_planes,
            self.in_planes // self.groups,
            self.kernel_size,
            self.kernel_size
        )

        yg = F.conv2d(
            xg, wg, bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=B * self.groups
        )

        y = yg.reshape(B, self.out_planes, yg.shape[-2], yg.shape[-1])

        if mixed_b is not None:
            y = y + mixed_b.view(B, self.out_planes, 1, 1)

        return y