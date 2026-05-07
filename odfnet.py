"""
ODFNet Count-First + ASPP
=========================

Architecture:
    Input
      -> ConvNeXt backbone features
      -> P1 (192 ch) -> CCSM1 -> 16 ch
      -> P2 (384 ch) -> CCSM2 -> 32 ch
      -> P3 (768 ch) -> ASPP -> 128 ch -> CCSM3 -> 64 ch
      -> Upsample all to [out_size, out_size]
      -> Concat: 16 + 32 + 64 = 112
      -> Fusion: 112 -> 128
      -> Shape head: [B,1,H,W]
      -> Count head: [B,1]
      -> P = softmax(S over spatial)
      -> D = P * C

"""

import os
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Utils
def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdirp(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def atomic_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def imnet_normalize(img_t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    return (img_t - mean) / std


# BOX density helpers
def _map_x(x, W0, out_size):
    return (x / max(1.0, (W0 - 1))) * (out_size - 1)


def _map_y(y, H0, out_size):
    return (y / max(1.0, (H0 - 1))) * (out_size - 1)


def add_box_density(dmap: np.ndarray, x: float, y: float, w: float, h: float, W0: int, H0: int, out_size: int):
    """
    Add 1 count uniformly inside the mapped bounding box region.
    Final density sum should match number of boxes.
    """
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    ox1 = int(round(_map_x(x1, W0, out_size)))
    oy1 = int(round(_map_y(y1, H0, out_size)))
    ox2 = int(round(_map_x(x2, W0, out_size)))
    oy2 = int(round(_map_y(y2, H0, out_size)))

    ox1 = max(0, min(out_size - 1, ox1))
    oy1 = max(0, min(out_size - 1, oy1))
    ox2 = max(0, min(out_size - 1, ox2))
    oy2 = max(0, min(out_size - 1, oy2))

    if ox2 < ox1:
        ox1, ox2 = ox2, ox1
    if oy2 < oy1:
        oy1, oy2 = oy2, oy1

    if ox2 == ox1:
        ox2 = min(out_size - 1, ox1 + 1)
    if oy2 == oy1:
        oy2 = min(out_size - 1, oy1 + 1)

    area = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)
    if area <= 0:
        return

    dmap[oy1:oy2 + 1, ox1:ox2 + 1] += (1.0 / float(area))


# Dataset
class CocoBBoxDensityDataset(Dataset):
    """
    COCO-style detection dataset -> box-uniform density map.

    Folder expected:
        root/
          train/
            images/
            annotations.json
          val/
            images/
            annotations.json
    """

    def __init__(
        self,
        root: str,
        split: str,
        img_size: int = 640,
        out_size: int = 128,
        max_objs: Optional[int] = None,
        use_imagenet_norm: bool = True,
        tile_train: bool = False,
        tile_size: int = 640,
        blur_sigma: float = 0.0,
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.img_dir = self.root / split / "images"
        self.ann_path = self.root / split / "annotations.json"

        assert self.img_dir.exists(), f"Missing images dir: {self.img_dir}"
        assert self.ann_path.exists(), f"Missing annotations.json: {self.ann_path}"

        self.img_size = int(img_size)
        self.out_size = int(out_size)
        self.max_objs = max_objs
        self.use_imagenet_norm = use_imagenet_norm
        self.tile_train = bool(tile_train)
        self.tile_size = int(tile_size)
        self.blur_sigma = float(blur_sigma)

        with open(self.ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images: List[Dict] = coco.get("images", [])
        self.img_to_anns: Dict[int, List[Dict]] = {}

        for ann in coco.get("annotations", []):
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.items: List[Tuple[int, str]] = []
        for img in self.images:
            img_id = img["id"]
            fname = img["file_name"]
            if (self.img_dir / fname).exists():
                self.items.append((img_id, fname))

        assert len(self.items) > 0, f"No images found in {self.img_dir}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        img_id, fname = self.items[idx]
        img_path = self.img_dir / fname

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img.shape[:2]

        anns = self.img_to_anns.get(img_id, [])
        if self.max_objs is not None:
            anns = anns[:self.max_objs]

        if self.tile_train:
            cw = min(W0, self.tile_size)
            ch = min(H0, self.tile_size)
            x0 = random.randint(0, W0 - cw) if W0 > cw else 0
            y0 = random.randint(0, H0 - ch) if H0 > ch else 0
            x1, y1 = x0 + cw, y0 + ch

            img = img[y0:y1, x0:x1].copy()
            H0, W0 = img.shape[:2]

            kept = []
            for a in anns:
                bbox = a.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                cx = x + 0.5 * w
                cy = y + 0.5 * h

                if (x0 <= cx < x1) and (y0 <= cy < y1):
                    x = x - x0
                    y = y - y0
                    x2 = min(W0, x + w)
                    y2 = min(H0, y + h)
                    x = max(0.0, x)
                    y = max(0.0, y)
                    w = max(1.0, x2 - x)
                    h = max(1.0, y2 - y)
                    kept.append({"bbox": [x, y, w, h]})
            anns = kept

        dmap = np.zeros((self.out_size, self.out_size), dtype=np.float32)
        for a in anns:
            bbox = a.get("bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            add_box_density(dmap, x, y, w, h, W0, H0, self.out_size)

        gt_count = float(len(anns))

        s = float(dmap.sum())
        if gt_count > 0 and s > 1e-8:
            dmap *= (gt_count / s)

        if self.blur_sigma > 0:
            dmap = cv2.GaussianBlur(dmap, ksize=(0, 0), sigmaX=self.blur_sigma)
            s2 = float(dmap.sum())
            if gt_count > 0 and s2 > 1e-8:
                dmap *= (gt_count / s2)

        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        if self.use_imagenet_norm:
            img_t = imnet_normalize(img_t)

        den_t = torch.from_numpy(dmap).unsqueeze(0).float()
        cnt_t = torch.tensor([gt_count], dtype=torch.float32)

        return img_t, den_t, cnt_t, str(img_path)


# Attention blocks

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg(x))
        m = self.mlp(self.max(x))
        return x * self.sigmoid(a + m)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(a))


class ResidualCA_SA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention(7)

    def forward(self, x):
        y = self.ca(x)
        y = self.sa(y)
        return x + y


# ASPP
class ASPP(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, rates=(1, 6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList()
        for r in rates:
            if r == 1:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))

        self.global_pool_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.global_pool_bn = nn.BatchNorm2d(out_ch)
        self.global_pool_relu = nn.ReLU(inplace=True)

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[-2:]

        outs = [branch(x) for branch in self.branches]

        gp = F.adaptive_avg_pool2d(x, 1)
        gp = self.global_pool_conv(gp)
        gp = self.global_pool_bn(gp) if gp.shape[0] > 1 else gp
        gp = self.global_pool_relu(gp)
        gp = F.interpolate(gp, size=(H, W), mode="bilinear", align_corners=False)

        outs.append(gp)
        y = torch.cat(outs, dim=1)
        return self.project(y)


# CCSM
class CCSM(nn.Module):
    """
    CCSM block:
        1x1 projection
        depthwise 3x3
        ODConv 1x1
        ODConv 1x1
        CA + SA
    """
    def __init__(self, ODConv2d, in_ch: int, out_ch: int, hidden_ratio: int = 2):
        super().__init__()

        hidden = max(out_ch, in_ch * hidden_ratio)

        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.od1 = ODConv2d(hidden, hidden, kernel_size=1, reduction=0.0625, kernel_num=4)
        self.act1 = nn.ReLU(inplace=True)

        self.od2 = ODConv2d(hidden, out_ch, kernel_size=1, reduction=0.0625, kernel_num=4)
        self.act2 = nn.ReLU(inplace=True)

        self.attn = ResidualCA_SA(out_ch)

    def forward(self, x):
        x = self.pre(x)
        x = self.dw(x)
        x = self.act1(self.od1(x))
        x = self.act2(self.od2(x))
        x = self.attn(x)
        return x


# Backbone
def build_backbone(name: str = "convnext_tiny", pretrained: bool = True):
    import torchvision

    if name == "convnext_tiny":
        model = torchvision.models.convnext_tiny(
            weights=(torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        )
    elif name == "convnext_small":
        model = torchvision.models.convnext_small(
            weights=(torchvision.models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None)
        )
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    return model.features


# ODFNet Count-First + ASPP
class ODFNet(nn.Module):
    def __init__(self, ODConv2d, backbone_name="convnext_tiny", pretrained=True, out_size=128):
        super().__init__()
        self.out_size = int(out_size)
        self.backbone = build_backbone(backbone_name, pretrained=pretrained)

        self.ccsm1 = CCSM(ODConv2d, in_ch=192, out_ch=16)
        self.ccsm2 = CCSM(ODConv2d, in_ch=384, out_ch=32)
        self.aspp  = ASPP(in_ch=768, out_ch=128, rates=(1, 6, 12, 18))
        self.ccsm3 = CCSM(ODConv2d, in_ch=128, out_ch=64)

        self.fuse = nn.Sequential(
            nn.Conv2d(16 + 32 + 64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.shape_head = nn.Conv2d(128, 1, kernel_size=1)

        self.count_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Softplus()
        )

    def _extract_pools(self, x):
        out = x
        p1 = p2 = p3 = None

        for i, blk in enumerate(self.backbone):
            out = blk(out)
            if i == 2:
                p1 = out
            elif i == 4:
                p2 = out
            elif i == 6:
                p3 = out

        if p1 is None or p2 is None or p3 is None:
            raise RuntimeError("Could not extract ConvNeXt stages correctly.")

        return p1, p2, p3

    def forward(self, x):
        p1, p2, p3 = self._extract_pools(x)

        f1 = self.ccsm1(p1)
        f2 = self.ccsm2(p2)
        f3 = self.aspp(p3)
        f3 = self.ccsm3(f3)

        f1 = F.interpolate(f1, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        fcat = torch.cat([f1, f2, f3], dim=1)
        ffused = self.fuse(fcat)

        S = self.shape_head(ffused)
        C = self.count_head(ffused)

        B, _, H, W = S.shape
        P = torch.softmax(S.view(B, 1, -1), dim=-1).view(B, 1, H, W)
        D = P * C.view(B, 1, 1, 1)

        return D, C, P, S


# Loss / metrics
def kl_q_to_p(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    P = P.clamp_min(eps)
    Q = Q.clamp_min(eps)
    return (Q * (Q.log() - P.log())).sum(dim=(-2, -1)).mean()


@torch.no_grad()
def eval_mae_rmse(model, loader, device):
    model.eval()

    abs_errs = []
    sq_errs = []

    for img, den, cnt, _ in loader:
        img = img.to(device)
        cnt = cnt.to(device)

        _, C, _, _ = model(img)

        pred = C.view(-1)
        gt = cnt.view(-1)
        e = pred - gt

        abs_errs.append(e.abs().cpu().numpy())
        sq_errs.append((e ** 2).cpu().numpy())

    abs_err = float(np.concatenate(abs_errs).mean()) if abs_errs else 0.0
    rmse = float(math.sqrt(np.concatenate(sq_errs).mean())) if sq_errs else 0.0
    return abs_err, rmse


# Visualization
@torch.no_grad()
def save_epoch_viz(model, loader, device, save_path: str, max_samples: int = 4):
    import matplotlib.pyplot as plt

    model.eval()
    batch = next(iter(loader))
    img, gt_den, gt_cnt, _ = batch

    img = img.to(device)
    gt_den = gt_den.to(device)
    gt_cnt = gt_cnt.to(device)

    D, C, _, _ = model(img)
    n = min(max_samples, img.size(0))

    plt.figure(figsize=(14, 4 * n))
    for i in range(n):
        im = img[i].detach().cpu().permute(1, 2, 0).numpy()
        gden = gt_den[i, 0].detach().cpu().numpy()
        pden = D[i, 0].detach().cpu().numpy()

        plt.subplot(n, 3, i * 3 + 1)
        plt.title(f"Image\nGT={gt_cnt[i].item():.0f}  Pred={C[i].item():.1f}")
        plt.imshow(np.clip(im, -3, 3))
        plt.axis("off")

        plt.subplot(n, 3, i * 3 + 2)
        plt.title(f"GT Density (sum={gden.sum():.2f})")
        plt.imshow(gden)
        plt.axis("off")

        plt.subplot(n, 3, i * 3 + 3)
        plt.title(f"Pred Density (sum={pden.sum():.2f})")
        plt.imshow(pden)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


# Training schedule
@dataclass
class TrainSchedule:
    warm_epochs: int = 5
    kl_max: float = 0.3
    mse_max: float = 0.05
    kl_ramp_epochs: int = 10
    mse_start_after_warm: int = 5
    mse_ramp_epochs: int = 10


def schedule_weights(epoch: int, s: TrainSchedule):
    if epoch <= s.warm_epochs:
        return 30.0, 0.0, 0.0

    tkl = min(1.0, (epoch - s.warm_epochs) / max(1, s.kl_ramp_epochs))
    w_kl = s.kl_max * tkl

    emse0 = s.warm_epochs + s.mse_start_after_warm
    if epoch <= emse0:
        w_mse = 0.0
    else:
        tmse = min(1.0, (epoch - emse0) / max(1, s.mse_ramp_epochs))
        w_mse = s.mse_max * tmse

    return 1.0, w_kl, w_mse


def train_one_epoch(model, loader, optim, device, use_amp: bool, w_cnt: float, w_kl: float, w_mse: float):
    model.train()

    total = 0.0
    logs = {"cnt": 0.0, "kl": 0.0, "mse": 0.0}
    n = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for img, gt_den, gt_cnt, _ in loader:
        img = img.to(device, non_blocking=True)
        gt_den = gt_den.to(device, non_blocking=True)
        gt_cnt = gt_cnt.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            D, C, P, _ = model(img)

            cnt_loss = F.smooth_l1_loss(torch.log1p(C.view(-1)), torch.log1p(gt_cnt.view(-1)))

            gt_sum = gt_den.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            Q = (gt_den / gt_sum).clamp_min(1e-8)
            kl = kl_q_to_p(P, Q)

            mse = F.mse_loss(D, gt_den)

            loss = w_cnt * cnt_loss + w_kl * kl + w_mse * mse

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optim)
        scaler.update()

        bs = img.size(0)
        total += loss.item() * bs
        logs["cnt"] += cnt_loss.item() * bs
        logs["kl"]  += kl.item() * bs
        logs["mse"] += mse.item() * bs
        n += bs

    total /= max(1, n)
    for k in logs:
        logs[k] /= max(1, n)

    return total, logs


# Main
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="/content/runs/odf_box_density")

    parser.add_argument("--odconv_path", type=str, default="/content/ODConv2d.py")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "convnext_small"])
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--out_size", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--tile_train", action="store_true")
    parser.add_argument("--tile_size", type=int, default=640)
    parser.add_argument("--blur_sigma", type=float, default=0.0)

    parser.add_argument("--warm_epochs", type=int, default=5)
    parser.add_argument("--kl_max", type=float, default=0.3)
    parser.add_argument("--mse_max", type=float, default=0.05)
    parser.add_argument("--kl_ramp_epochs", type=int, default=10)
    parser.add_argument("--mse_start_after_warm", type=int, default=5)
    parser.add_argument("--mse_ramp_epochs", type=int, default=10)

    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--viz_each_epoch", action="store_true")

    args = parser.parse_args()

    seed_everything(args.seed)
    mkdirp(args.save_dir)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    import importlib.util
    spec = importlib.util.spec_from_file_location("odconv_module", args.odconv_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ODConv2d = mod.ODConv2d

    train_ds = CocoBBoxDensityDataset(
        root=args.data_root,
        split="train",
        img_size=args.img_size,
        out_size=args.out_size,
        use_imagenet_norm=True,
        tile_train=args.tile_train,
        tile_size=args.tile_size,
        blur_sigma=args.blur_sigma
    )

    val_ds = CocoBBoxDensityDataset(
        root=args.data_root,
        split="val",
        img_size=args.img_size,
        out_size=args.out_size,
        use_imagenet_norm=True,
        tile_train=False,
        blur_sigma=args.blur_sigma
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    model = ODFNet(
        ODConv2d=ODConv2d,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        out_size=args.out_size
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    sched = TrainSchedule(
        warm_epochs=args.warm_epochs,
        kl_max=args.kl_max,
        mse_max=args.mse_max,
        kl_ramp_epochs=args.kl_ramp_epochs,
        mse_start_after_warm=args.mse_start_after_warm,
        mse_ramp_epochs=args.mse_ramp_epochs
    )

    best_mae = 1e9
    best_path = str(Path(args.save_dir) / "best.pth")
    last_path = str(Path(args.save_dir) / "last.pth")

    for epoch in range(1, args.epochs + 1):
        w_cnt, w_kl, w_mse = schedule_weights(epoch, sched)

        tr_loss, parts = train_one_epoch(
            model, train_loader, optim, device,
            use_amp=args.use_amp,
            w_cnt=w_cnt, w_kl=w_kl, w_mse=w_mse
        )

        val_mae, val_rmse = eval_mae_rmse(model, val_loader, device)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={tr_loss:.4f} "
            f"(cnt={parts['cnt']:.4f} kl={parts['kl']:.4f} mse={parts['mse']:.4f}) | "
            f"w_cnt={w_cnt:.2f} w_kl={w_kl:.3f} w_mse={w_mse:.3f} | "
            f"val_MAE={val_mae:.3f} val_RMSE={val_rmse:.3f}"
        )

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "args": vars(args),
        }

        atomic_save(payload, last_path)

        if val_mae < best_mae:
            best_mae = val_mae
            atomic_save(payload, best_path)
            print(f"    Best updated: MAE={best_mae:.3f} -> {best_path}")

        if args.viz_each_epoch:
            viz_path = str(Path(args.save_dir) / f"viz_epoch_{epoch:03d}.png")
            save_epoch_viz(model, val_loader, device, viz_path, max_samples=4)
            print(f"    Saved: {viz_path}")

    print("Done. Best model:", best_path)


if __name__ == "__main__":
    main()