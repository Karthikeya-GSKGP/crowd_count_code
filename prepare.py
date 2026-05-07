import argparse
import json
import random
import shutil
from pathlib import Path

VALID_CLASSES = {"Human", "Umbrella", "Pickets"}

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def find_image(data, json_path, image_dir):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]
    image_dir = Path(image_dir)
    image_id = str(data.get("image_id", "")).strip()
    image_path = str(data.get("image_path", "")).strip()
    candidates = []

    if image_path:
        candidates.append(image_dir / Path(image_path).name)
        candidates.append(json_path.parent / image_path)
        candidates.append(json_path.parent / Path(image_path).name)

    if image_id:
        for ext in exts:
            candidates.append(image_dir / f"{image_id}{ext}")
            candidates.append(json_path.parent / f"{image_id}{ext}")

    stem = json_path.stem
    if stem.endswith("_labels"):
        stem = stem[:-7]

    for ext in exts:
        candidates.append(image_dir / f"{stem}{ext}")
        candidates.append(json_path.parent / f"{stem}{ext}")

    seen = set()
    for p in candidates:
        p = Path(p)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            return p

    return None

def clip_box(box, width, height):
    x1 = float(box.get("x_min", 0))
    y1 = float(box.get("y_min", 0))
    x2 = float(box.get("x_max", 0))
    y2 = float(box.get("y_max", 0))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))

    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None

    out = dict(box)
    out["x_min"] = round(x1, 4)
    out["y_min"] = round(y1, 4)
    out["x_max"] = round(x2, 4)
    out["y_max"] = round(y2, 4)
    out["width"] = round(x2 - x1, 4)
    out["height"] = round(y2 - y1, 4)
    return out

def clean_json(data):
    width = int(data.get("image_width", 0))
    height = int(data.get("image_height", 0))

    if width <= 0 or height <= 0:
        return None

    boxes = []
    counts = {"Human": 0, "Umbrella": 0, "Pickets": 0}

    for box in data.get("boxes", []):
        cls = box.get("class", "")
        if cls not in VALID_CLASSES:
            continue
        clipped = clip_box(box, width, height)
        if clipped is None:
            continue
        clipped["box_id"] = len(boxes) + 1
        boxes.append(clipped)
        counts[cls] += 1

    out = dict(data)
    out["class_counts"] = counts
    out["total_annotations"] = len(boxes)
    out["boxes"] = boxes
    return out

def copy_pair(img_path, json_data, out_root, split, new_stem):
    img_ext = img_path.suffix.lower()
    img_out = out_root / split / "images" / f"{new_stem}{img_ext}"
    json_out = out_root / split / "labels" / f"{new_stem}_labels.json"

    img_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, img_out)

    json_data = dict(json_data)
    json_data["image_id"] = new_stem
    json_data["image_path"] = f"../images/{new_stem}{img_ext}"
    json_data["annotated_image_path"] = ""

    write_json(json_data, json_out)

def split_items(items, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    train_ratio = train_ratio / total
    val_ratio = val_ratio / total

    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    if len(train) == 0 and len(items) > 0:
        train = [items[0]]
        val = items[1:]

    return train, val, test

def make_manifest(out_root, rows):
    write_json({"num_samples": len(rows), "samples": rows}, out_root / "manifest.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prefix", type=str, default="odfa")
    parser.add_argument("--keep_empty", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    json_dir = Path(args.json_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    items = []
    skipped = []

    for jp in sorted(json_dir.glob("*.json")):
        if jp.name.lower() == "annotations.json":
            continue

        try:
            data = read_json(jp)
        except Exception as e:
            skipped.append({"json": str(jp), "reason": f"json_read_error: {e}"})
            continue

        img_path = find_image(data, jp, image_dir)
        if img_path is None:
            skipped.append({"json": str(jp), "reason": "image_not_found"})
            continue

        cleaned = clean_json(data)
        if cleaned is None:
            skipped.append({"json": str(jp), "reason": "invalid_image_size"})
            continue

        if not args.keep_empty and cleaned["total_annotations"] == 0:
            skipped.append({"json": str(jp), "reason": "empty_annotations"})
            continue

        items.append({"image": img_path, "json": jp, "data": cleaned})

    train, val, test = split_items(items, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    split_map = {"train": train, "val": val, "test": test}
    manifest_rows = []

    for split, split_items_list in split_map.items():
        for i, item in enumerate(split_items_list, start=1):
            new_stem = f"{args.prefix}_{split}_{i:05d}"
            copy_pair(item["image"], item["data"], out_root, split, new_stem)
            manifest_rows.append({
                "split": split,
                "image": str(item["image"]),
                "json": str(item["json"]),
                "new_id": new_stem,
                "total_annotations": item["data"]["total_annotations"],
                "class_counts": item["data"]["class_counts"]
            })

    make_manifest(out_root, manifest_rows)
    write_json({"num_skipped": len(skipped), "skipped": skipped}, out_root / "skipped.json")

    print("Done")
    print("Output:", out_root)
    print("Train:", len(train))
    print("Val:", len(val))
    print("Test:", len(test))
    print("Skipped:", len(skipped))

if __name__ == "__main__":
    main()
