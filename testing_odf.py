import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def mkdirp(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def imnet_normalize(img_t):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img_t.dtype, device=img_t.device).view(3, 1, 1)
    return (img_t - mean) / std

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_images(image_dir):
    image_dir = Path(image_dir)
    files = []
    for e in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]:
        files.extend(image_dir.glob(e))
        files.extend(image_dir.glob(e.upper()))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return files

def find_json_for_image(img_path, json_dir):
    if not json_dir:
        return None
    json_dir = Path(json_dir)
    candidates = [json_dir / f"{img_path.stem}_labels.json", json_dir / f"{img_path.stem}.json"]
    for p in candidates:
        if p.exists():
            return p
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

def parse_gt_count(json_path, target_class, W, H):
    if json_path is None or not json_path.exists():
        return None
    data = read_json(json_path)
    count = 0
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
            count += 1
    return count

def preprocess_image(img_rgb, img_size, device):
    img_resized = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_t = imnet_normalize(img_t)
    return img_t.unsqueeze(0).to(device)

def load_model(ckpt_path, ODFNet, ODConv2d, backbone, out_size, device):
    model = ODFNet(ODConv2d=ODConv2d, backbone_name=backbone, pretrained=False, out_size=out_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

@torch.no_grad()
def predict_one(model, img_t):
    D, C, _, _ = model(img_t)
    d = D[0, 0].detach().cpu().numpy().astype(np.float32)
    c = float(C.view(-1)[0].detach().cpu().item())
    return d, c

def save_density_png(density, path):
    arr = density.astype(np.float32)
    if arr.max() > arr.min():
        norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        norm = np.zeros_like(arr, dtype=np.float32)
    plt.figure(figsize=(5, 5))
    plt.imshow(norm)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_visualization(img_rgb, maps, counts, out_path):
    names = list(maps.keys())
    n = 1 + len(names)
    plt.figure(figsize=(4 * n, 4))
    plt.subplot(1, n, 1)
    plt.imshow(img_rgb)
    plt.title("Image")
    plt.axis("off")

    for i, name in enumerate(names, start=2):
        plt.subplot(1, n, i)
        plt.imshow(maps[name])
        plt.title(f"{name}\ncount={counts[name]:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def write_csv(rows, path):
    if len(rows) == 0:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def summarize_errors(rows, pred_key, gt_key):
    vals = []
    sqs = []
    for r in rows:
        gt = r.get(gt_key, "")
        if gt == "" or gt is None:
            continue
        e = float(r[pred_key]) - float(gt)
        vals.append(abs(e))
        sqs.append(e * e)
    if not vals:
        return None, None
    return float(np.mean(vals)), float(math.sqrt(np.mean(sqs)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, default=None)
    parser.add_argument("--odfnet_path", type=str, required=True)
    parser.add_argument("--odconv_path", type=str, required=True)
    parser.add_argument("--single_ckpt", type=str, default=None)
    parser.add_argument("--single_class", type=str, default="Human", choices=["Human", "Umbrella", "Pickets", "All"])
    parser.add_argument("--human_ckpt", type=str, default=None)
    parser.add_argument("--umbrella_ckpt", type=str, default=None)
    parser.add_argument("--pickets_ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./odf_direct_json_test")
    parser.add_argument("--backbone", type=str, default="convnext_tiny", choices=["convnext_tiny", "convnext_small"])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_maps", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    args = parser.parse_args()

    mkdirp(args.out_dir)
    maps_dir = Path(args.out_dir) / "density_maps"
    viz_dir = Path(args.out_dir) / "visualizations"

    if args.save_maps:
        mkdirp(maps_dir)
    if args.save_viz:
        mkdirp(viz_dir)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    odconv_mod = import_from_path("odconv_module", args.odconv_path)
    odfnet_mod = import_from_path("odfnet_module", args.odfnet_path)
    ODConv2d = odconv_mod.ODConv2d
    ODFNet = odfnet_mod.ODFNet

    models = {}

    if args.single_ckpt:
        models[args.single_class] = load_model(args.single_ckpt, ODFNet, ODConv2d, args.backbone, args.out_size, device)
        mode = "single"
    else:
        if args.human_ckpt:
            models["Human"] = load_model(args.human_ckpt, ODFNet, ODConv2d, args.backbone, args.out_size, device)
        if args.umbrella_ckpt:
            models["Umbrella"] = load_model(args.umbrella_ckpt, ODFNet, ODConv2d, args.backbone, args.out_size, device)
        if args.pickets_ckpt:
            models["Pickets"] = load_model(args.pickets_ckpt, ODFNet, ODConv2d, args.backbone, args.out_size, device)
        mode = "fusion"

    if len(models) == 0:
        raise RuntimeError("No checkpoint provided")

    rows = []
    image_files = collect_images(args.image_dir)

    for idx, img_path in enumerate(image_files, start=1):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        img_t = preprocess_image(img_rgb, args.img_size, device)
        json_path = find_json_for_image(img_path, args.json_dir)

        pred_maps = {}
        pred_counts = {}
        gt_counts = {}

        for cls, model in models.items():
            d, c = predict_one(model, img_t)
            pred_maps[cls] = d
            pred_counts[cls] = c
            gt_counts[cls] = parse_gt_count(json_path, cls, W, H)

        if mode == "fusion":
            fused = None
            total_pred = 0.0
            total_gt = 0
            has_gt = False

            for cls in ["Human", "Umbrella", "Pickets"]:
                if cls in pred_maps:
                    fused = pred_maps[cls].copy() if fused is None else fused + pred_maps[cls]
                    total_pred += float(pred_counts[cls])
                    if gt_counts.get(cls) is not None:
                        total_gt += int(gt_counts[cls])
                        has_gt = True

            if fused is not None:
                pred_maps["Fused"] = fused
                pred_counts["Fused"] = total_pred
                gt_counts["Fused"] = total_gt if has_gt else None

        row = {"image": img_path.name, "image_path": str(img_path), "json_path": str(json_path) if json_path else ""}

        for name in pred_counts:
            row[f"pred_{name}"] = float(pred_counts[name])
            row[f"gt_{name}"] = "" if gt_counts.get(name) is None else int(gt_counts[name])
            row[f"abs_error_{name}"] = "" if gt_counts.get(name) is None else abs(float(pred_counts[name]) - float(gt_counts[name]))

        rows.append(row)

        stem = img_path.stem

        if args.save_maps:
            for name, d in pred_maps.items():
                np.save(maps_dir / f"{stem}_{name}_density.npy", d)
                save_density_png(d, str(maps_dir / f"{stem}_{name}_density.png"))

        if args.save_viz:
            save_visualization(img_rgb, pred_maps, pred_counts, str(viz_dir / f"{stem}_viz.png"))

        print(f"[{idx}/{len(image_files)}] {img_path.name}")

    csv_path = Path(args.out_dir) / "predictions.csv"
    write_csv(rows, str(csv_path))
    print("Saved:", csv_path)

    if rows:
        for key in rows[0].keys():
            if key.startswith("pred_"):
                name = key.replace("pred_", "")
                gt_key = f"gt_{name}"
                if gt_key in rows[0]:
                    mae, rmse = summarize_errors(rows, key, gt_key)
                    if mae is not None:
                        print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}")

if __name__ == "__main__":
    main()
