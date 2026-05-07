import argparse
import csv
import importlib.util
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

VALID_CLASSES = ["Human", "Umbrella", "Pickets"]

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def mkdirp(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def atomic_save(obj, path):
    path = str(path)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def imnet_normalize(img_t):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    return (img_t - mean) / std

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_json_files(json_dir):
    files = sorted(Path(json_dir).glob("*.json"))
    files = [p for p in files if p.name.lower() != "annotations.json"]
    if len(files) == 0:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")
    return files

def find_image_for_json(data, json_path, image_dir):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    candidates = []
    stored = str(data.get("image_path", "")).strip()
    image_id = str(data.get("image_id", "")).strip()
    json_dir = json_path.parent

    if image_dir:
        image_dir = Path(image_dir)
        if stored:
            candidates.append(image_dir / Path(stored).name)
        if image_id:
            for ext in exts:
                candidates.append(image_dir / f"{image_id}{ext}")

    if stored:
        p = Path(stored)
        candidates.append(json_dir / p)
        candidates.append(json_dir / p.name)

    if image_id:
        for ext in exts:
            candidates.append(json_dir / f"{image_id}{ext}")

    stem = json_path.stem
    if stem.endswith("_labels"):
        stem = stem[:-7]
    if image_dir:
        for ext in exts:
            candidates.append(Path(image_dir) / f"{stem}{ext}")
    for ext in exts:
        candidates.append(json_dir / f"{stem}{ext}")

    seen = set()
    for c in candidates:
        c = Path(c)
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists() and c.is_file():
            return c
    return None

def clip_box_xyxy(x1, y1, x2, y2, W, H):
    x1 = float(max(0.0, min(W - 1.0, x1)))
    y1 = float(max(0.0, min(H - 1.0, y1)))
    x2 = float(max(0.0, min(W - 1.0, x2)))
    y2 = float(max(0.0, min(H - 1.0, y2)))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]

def parse_boxes(data, target_class, W, H):
    boxes = []
    for b in data.get("boxes", []):
        cls = b.get("class", "")
        if target_class != "All" and cls != target_class:
            continue
        if all(k in b for k in ["x_min", "y_min", "x_max", "y_max"]):
            clipped = clip_box_xyxy(b["x_min"], b["y_min"], b["x_max"], b["y_max"], W, H)
        elif "bbox" in b and len(b["bbox"]) == 4:
            x, y, w, h = b["bbox"]
            clipped = clip_box_xyxy(x, y, x + w, y + h, W, H)
        else:
            clipped = None
        if clipped is not None:
            boxes.append(clipped)
    return boxes

def _map_x(x, W0, out_size):
    return (x / max(1.0, W0 - 1)) * (out_size - 1)

def _map_y(y, H0, out_size):
    return (y / max(1.0, H0 - 1)) * (out_size - 1)

def add_box_density(dmap, box, W0, H0, out_size):
    x1, y1, x2, y2 = box
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
    if area > 0:
        dmap[oy1:oy2 + 1, ox1:ox2 + 1] += 1.0 / float(area)

def make_density_map(boxes, W, H, out_size, blur_sigma):
    dmap = np.zeros((out_size, out_size), dtype=np.float32)
    for box in boxes:
        add_box_density(dmap, box, W, H, out_size)
    gt = float(len(boxes))
    s = float(dmap.sum())
    if gt > 0 and s > 1e-8:
        dmap *= gt / s
    if blur_sigma > 0:
        dmap = cv2.GaussianBlur(dmap, ksize=(0, 0), sigmaX=float(blur_sigma))
        s2 = float(dmap.sum())
        if gt > 0 and s2 > 1e-8:
            dmap *= gt / s2
    return dmap.astype(np.float32)

def rotate_image_and_boxes(img, boxes, angle):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], dtype=np.float32)
        rc = corners @ M.T
        clipped = clip_box_xyxy(rc[:, 0].min(), rc[:, 1].min(), rc[:, 0].max(), rc[:, 1].max(), W, H)
        if clipped is not None:
            new_boxes.append(clipped)
    return out, new_boxes

def hflip_image_and_boxes(img, boxes):
    H, W = img.shape[:2]
    img = np.ascontiguousarray(img[:, ::-1, :])
    new_boxes = []
    for x1, y1, x2, y2 in boxes:
        clipped = clip_box_xyxy(W - 1 - x2, y1, W - 1 - x1, y2, W, H)
        if clipped is not None:
            new_boxes.append(clipped)
    return img, new_boxes

def apply_color_jitter_rgb(img, brightness=0.15, saturation=0.15, hue=5, contrast=0.15):
    out = img.astype(np.float32) / 255.0
    c = 1.0 + random.uniform(-contrast, contrast)
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * c + mean
    b = 1.0 + random.uniform(-brightness, brightness)
    out = np.clip(out * b, 0.0, 1.0)
    out_u8 = (out * 255.0).astype(np.uint8)
    hsv = cv2.cvtColor(out_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 1.0 + random.uniform(-saturation, saturation)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-hue, hue)) % 180.0
    hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_grayscale_rgb(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return np.stack([gray, gray, gray], axis=-1)

def apply_impulse_noise(img, amount):
    if amount <= 0:
        return img
    out = img.copy()
    H, W = out.shape[:2]
    n = int(H * W * amount)
    if n <= 0:
        return out
    ys = np.random.randint(0, H, size=n)
    xs = np.random.randint(0, W, size=n)
    vals = np.random.choice([0, 255], size=n).astype(np.uint8)
    out[ys, xs, :] = vals[:, None]
    return out

class ODFDirectJsonDataset(Dataset):
    def __init__(self, json_files, image_dir, target_class, img_size, out_size, augment, rotate_deg, gray_prob, hflip_prob, impulse_amount, blur_sigma, drop_empty):
        self.items = []
        self.image_dir = image_dir
        self.target_class = target_class
        self.img_size = int(img_size)
        self.out_size = int(out_size)
        self.augment = bool(augment)
        self.rotate_deg = float(rotate_deg)
        self.gray_prob = float(gray_prob)
        self.hflip_prob = float(hflip_prob)
        self.impulse_amount = float(impulse_amount)
        self.blur_sigma = float(blur_sigma)

        for jp in json_files:
            jp = Path(jp)
            try:
                data = read_json(jp)
                img_path = find_image_for_json(data, jp, image_dir)
                if img_path is None:
                    continue
                if drop_empty:
                    tmp = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if tmp is None:
                        continue
                    H, W = tmp.shape[:2]
                    if len(parse_boxes(data, target_class, W, H)) == 0:
                        continue
                self.items.append((jp, img_path))
            except Exception:
                continue

        if len(self.items) == 0:
            raise RuntimeError("No usable image and JSON pairs found")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        json_path, img_path = self.items[idx]
        data = read_json(json_path)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image {img_path}")

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        boxes = parse_boxes(data, self.target_class, W, H)

        if self.augment:
            if random.random() < self.hflip_prob:
                img, boxes = hflip_image_and_boxes(img, boxes)
            if self.rotate_deg > 0:
                img, boxes = rotate_image_and_boxes(img, boxes, random.uniform(-self.rotate_deg, self.rotate_deg))
            if random.random() < self.gray_prob:
                img = apply_grayscale_rgb(img)
            else:
                img = apply_color_jitter_rgb(img)
            img = apply_impulse_noise(img, self.impulse_amount)

        H2, W2 = img.shape[:2]
        dmap = make_density_map(boxes, W2, H2, self.out_size, self.blur_sigma)
        gt_count = float(len(boxes))
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        img_t = imnet_normalize(img_t)
        den_t = torch.from_numpy(dmap).unsqueeze(0).float()
        cnt_t = torch.tensor([gt_count], dtype=torch.float32)
        return img_t, den_t, cnt_t, str(img_path), str(json_path)

def kl_q_to_p(P, Q, eps=1e-8):
    P = P.clamp_min(eps)
    Q = Q.clamp_min(eps)
    return (Q * (Q.log() - P.log())).sum(dim=(-2, -1)).mean()

@dataclass
class TrainSchedule:
    warm_epochs: int = 5
    kl_max: float = 0.3
    mse_max: float = 0.05
    kl_ramp_epochs: int = 10
    mse_start_after_warm: int = 5
    mse_ramp_epochs: int = 10

def schedule_weights(epoch, s):
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

def train_one_epoch(model, loader, optimizer, device, use_amp, w_cnt, w_kl, w_mse):
    model.train()
    total_loss = 0.0
    logs = {"cnt": 0.0, "kl": 0.0, "mse": 0.0}
    n = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for img, gt_den, gt_cnt, _, _ in loader:
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

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        bs = img.size(0)
        total_loss += loss.item() * bs
        logs["cnt"] += cnt_loss.item() * bs
        logs["kl"] += kl.item() * bs
        logs["mse"] += mse.item() * bs
        n += bs

    total_loss /= max(1, n)
    for k in logs:
        logs[k] /= max(1, n)
    return total_loss, logs

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    rows = []
    abs_errs = []
    sq_errs = []

    for img, _, cnt, img_paths, json_paths in loader:
        img = img.to(device)
        cnt = cnt.to(device)
        _, C, _, _ = model(img)
        pred = C.view(-1)
        gt = cnt.view(-1)
        err = pred - gt
        abs_errs.append(err.abs().detach().cpu().numpy())
        sq_errs.append((err ** 2).detach().cpu().numpy())

        for i in range(img.size(0)):
            rows.append({
                "image_path": img_paths[i],
                "json_path": json_paths[i],
                "gt_count": float(gt[i].detach().cpu().item()),
                "pred_count": float(pred[i].detach().cpu().item()),
                "abs_error": float(abs(err[i]).detach().cpu().item())
            })

    mae = float(np.concatenate(abs_errs).mean()) if abs_errs else 0.0
    rmse = float(math.sqrt(np.concatenate(sq_errs).mean())) if sq_errs else 0.0
    return mae, rmse, rows

def save_eval_csv(rows, path):
    if len(rows) == 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def split_train_val(files, val_ratio, seed):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 else 0
    return files[n_val:], files[:n_val]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json_dir", type=str, required=True)
    parser.add_argument("--val_json_dir", type=str, default=None)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, default=None)
    parser.add_argument("--odfnet_path", type=str, required=True)
    parser.add_argument("--odconv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./runs/odf_direct_json")
    parser.add_argument("--target_class", type=str, default="Human", choices=["Human", "Umbrella", "Pickets", "All"])
    parser.add_argument("--backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "convnext_small"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--drop_empty_train", action="store_true")
    parser.add_argument("--drop_empty_val", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--rotate_deg", type=float, default=15.0)
    parser.add_argument("--gray_prob", type=float, default=0.15)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--impulse_amount", type=float, default=0.001)
    parser.add_argument("--blur_sigma", type=float, default=0.0)
    parser.add_argument("--warm_epochs", type=int, default=5)
    parser.add_argument("--kl_max", type=float, default=0.3)
    parser.add_argument("--mse_max", type=float, default=0.05)
    parser.add_argument("--kl_ramp_epochs", type=int, default=10)
    parser.add_argument("--mse_start_after_warm", type=int, default=5)
    parser.add_argument("--mse_ramp_epochs", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    mkdirp(args.save_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    odconv_mod = import_from_path("odconv_module", args.odconv_path)
    odfnet_mod = import_from_path("odfnet_module", args.odfnet_path)
    ODConv2d = odconv_mod.ODConv2d
    ODFNet = odfnet_mod.ODFNet

    train_all = collect_json_files(args.train_json_dir)
    if args.val_json_dir:
        train_files = train_all
        val_files = collect_json_files(args.val_json_dir)
    else:
        train_files, val_files = split_train_val(train_all, args.val_ratio, args.seed)

    val_image_dir = args.val_image_dir if args.val_image_dir else args.image_dir

    train_ds = ODFDirectJsonDataset(train_files, args.image_dir, args.target_class, args.img_size, args.out_size, args.augment, args.rotate_deg, args.gray_prob, args.hflip_prob, args.impulse_amount, args.blur_sigma, args.drop_empty_train)
    val_ds = ODFDirectJsonDataset(val_files if len(val_files) > 0 else train_files, val_image_dir, args.target_class, args.img_size, args.out_size, False, args.rotate_deg, args.gray_prob, args.hflip_prob, args.impulse_amount, args.blur_sigma, args.drop_empty_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), drop_last=(len(train_ds) >= args.batch))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    model = ODFNet(ODConv2d=ODConv2d, backbone_name=args.backbone, pretrained=args.pretrained, out_size=args.out_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    sched = TrainSchedule(args.warm_epochs, args.kl_max, args.mse_max, args.kl_ramp_epochs, args.mse_start_after_warm, args.mse_ramp_epochs)
    best_mae = float("inf")
    best_path = Path(args.save_dir) / f"best_{args.target_class}.pth"
    last_path = Path(args.save_dir) / f"last_{args.target_class}.pth"
    log_csv = Path(args.save_dir) / "training_log.csv"

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "cnt_loss", "kl_loss", "mse_loss", "w_cnt", "w_kl", "w_mse", "val_mae", "val_rmse"])
        writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        w_cnt, w_kl, w_mse = schedule_weights(epoch, sched)
        train_loss, parts = train_one_epoch(model, train_loader, optimizer, device, args.use_amp, w_cnt, w_kl, w_mse)
        val_mae, val_rmse, val_rows = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch:03d}/{args.epochs}] loss={train_loss:.4f} cnt={parts['cnt']:.4f} kl={parts['kl']:.4f} mse={parts['mse']:.4f} val_MAE={val_mae:.3f} val_RMSE={val_rmse:.3f}")

        payload = {"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(), "val_mae": val_mae, "val_rmse": val_rmse, "target_class": args.target_class, "args": vars(args)}
        atomic_save(payload, last_path)

        if val_mae < best_mae:
            best_mae = val_mae
            atomic_save(payload, best_path)
            save_eval_csv(val_rows, str(Path(args.save_dir) / f"best_val_predictions_{args.target_class}.csv"))
            print(f"Best updated: MAE={best_mae:.3f} -> {best_path}")

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "cnt_loss", "kl_loss", "mse_loss", "w_cnt", "w_kl", "w_mse", "val_mae", "val_rmse"])
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "cnt_loss": parts["cnt"], "kl_loss": parts["kl"], "mse_loss": parts["mse"], "w_cnt": w_cnt, "w_kl": w_kl, "w_mse": w_mse, "val_mae": val_mae, "val_rmse": val_rmse})

    print("Training finished")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)

if __name__ == "__main__":
    main()
