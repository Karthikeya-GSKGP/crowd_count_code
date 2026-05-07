# Counting the Uncounted: Correcting Occlusion-caused Blind Spots in Crowd Counting Benchmarks

This repository contains the code for **ODFNet**, an occlusion-aware crowd-counting framework for images with embedded occluders such as umbrellas and picket signs.

ODFNet uses a direct JSON annotation pipeline and does **not** require COCO conversion.

```text
Human Density + Umbrella Density + Pickets Density = Final Crowd Density
```

---

## Main Files

| File | Description |
|---|---|
| `ODFlow.py` | Annotation tool for Human, Umbrella, and Pickets boxes |
| `prepare.py` | Prepares train/val/test splits from image-level JSON labels |
| `training_odf.py` | Trains ODFNet directly from JSON labels |
| `testing_odf.py` | Tests trained models and performs density fusion |
| `odfnet.py` | ODFNet model architecture |
| `ODConv2d.py` | Dynamic convolution module |

---

## Dataset Format

Before preparation:

```text
ODF-A/
├── images/
│   ├── img_0001.jpg
│   └── ...
└── labels/
    ├── img_0001_labels.json
    └── ...
```

Each JSON file contains image metadata, class counts, and bounding boxes for:

```text
Human
Umbrella
Pickets
```

---

## Installation

```bash
conda create -n odfnet python=3.10 -y
conda activate odfnet

pip install torch torchvision
pip install numpy opencv-python matplotlib pillow tqdm
```

---

## Dataset Preparation

```bash
python prepare.py \
  --image_dir /content/ODF-A/images \
  --json_dir /content/ODF-A/labels \
  --out_root /content/ODF-A-prepared \
  --train_ratio 0.7 \
  --val_ratio 0.1 \
  --test_ratio 0.2 \
  --seed 123
```

Output:

```text
ODF-A-prepared/
├── train/images/
├── train/labels/
├── val/images/
├── val/labels/
├── test/images/
├── test/labels/
├── manifest.json
└── skipped.json
```

---

## Training

Train one model per class.

### Human

```bash
python training_odf.py \
  --train_json_dir /content/ODF-A-prepared/train/labels \
  --val_json_dir /content/ODF-A-prepared/val/labels \
  --image_dir /content/ODF-A-prepared/train/images \
  --val_image_dir /content/ODF-A-prepared/val/images \
  --odfnet_path /content/odfnet.py \
  --odconv_path /content/ODConv2d.py \
  --save_dir /content/runs/Human \
  --target_class Human \
  --img_size 512 \
  --out_size 128 \
  --epochs 60 \
  --batch 2 \
  --device cuda \
  --augment \
  --use_amp
```

For Umbrella and Pickets, change:

```text
--save_dir /content/runs/Umbrella --target_class Umbrella
--save_dir /content/runs/Pickets  --target_class Pickets
```

Training outputs:

```text
best_Human.pth
last_Human.pth
training_log.csv
best_val_predictions_Human.csv
```

---

## Testing and Fusion

```bash
python testing_odf.py \
  --image_dir /content/ODF-A-prepared/test/images \
  --json_dir /content/ODF-A-prepared/test/labels \
  --human_ckpt /content/runs/Human/best_Human.pth \
  --umbrella_ckpt /content/runs/Umbrella/best_Umbrella.pth \
  --pickets_ckpt /content/runs/Pickets/best_Pickets.pth \
  --odfnet_path /content/odfnet.py \
  --odconv_path /content/ODConv2d.py \
  --out_dir /content/test_fusion \
  --img_size 512 \
  --out_size 128 \
  --device cuda \
  --save_maps \
  --save_viz
```

Testing outputs:

```text
test_fusion/
├── predictions.csv
├── density_maps/
└── visualizations/
```

---

## Model Summary

ODFNet uses:

- ConvNeXt backbone
- ODConv2d dynamic convolution
- ASPP multi-scale context module
- Count head
- Shape head
- Count-consistent density map

The model predicts a global count and a spatial density distribution. The final density map is computed as:

```text
D = softmax(S) × C
```

where `S` is the shape map and `C` is the predicted count.

---

## Metrics

The main evaluation metrics are:

```text
MAE
RMSE
```

For fusion testing, the final count is:

```text
C_final = C_Human + C_Umbrella + C_Pickets
```

---

## Citation

```bibtex
@inproceedings{anonymous2026odfnet,
  title={ODFNet: Occlusion-Density Fusion for Crowd Counting Under Embedded Occlusions},
  author={Anonymous Authors},
  booktitle={Submitted to NeurIPS},
  year={2026}
}
```

---

## License

This repository is for academic and research use. Please check dataset-specific licenses before redistributing images.
