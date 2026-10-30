
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanou.config import resolve
from nanou.evaluate import split_files, split_dataset
from nanou.predictor import load_predictor


parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "val", "test"],
                    default="test")
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()


config = resolve("tinyagri")
images, masks = split_files(config, args.split)

if not 0 <= args.index < len(images):
    raise ValueError(f"Index phải từ 0 đến {len(images) - 1}")

image_path = Path(images[args.index])
mask_map = {Path(p).name: p for p in masks}
mask_path = mask_map[image_path.name]

print("Ảnh:", image_path.name)

dataset = split_dataset(
    config,
    [str(image_path)],
    [mask_path],
    batch_size=1,
)

image_batch, mask_batch = next(iter(dataset))

predictor = load_predictor(
    "models/TinyAgri/nanou.tflite"
)

prob = predictor.predict(image_batch.numpy())[0, :, :, 0]
pred = (prob >= args.threshold).astype(np.uint8)

mean = np.asarray(config.normalization["mean"])
std = np.asarray(config.normalization["std"])

rgb = np.clip(
    image_batch.numpy()[0] * std + mean,
    0, 1
)
gt = mask_batch.numpy()[0, :, :, 0]

gt_binary = gt >= 0.5
pred_binary = pred.astype(bool)

intersection = np.logical_and(gt_binary, pred_binary).sum()
union = np.logical_or(gt_binary, pred_binary).sum()
road_iou = intersection / union if union else 1.0

print(f"Road IoU của ảnh: {road_iou:.4f}")

# 6. Hiển thị kết quả thành 4 cột
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(rgb)
axes[0].set_title("Input RGB")

axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Ground Truth")

axes[2].imshow(prob, cmap="viridis", vmin=0, vmax=1)
axes[2].set_title("Probability")

axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
axes[3].set_title(f"Prediction (t={args.threshold})")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

# 7. Lưu ảnh, không ghi đè báo cáo gốc
output_dir = Path("results/TinyAgri/nanou/custom_samples")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / (
    f"{args.split}_{args.index:03d}_{image_path.stem}.png"
)

plt.savefig(output_path, dpi=180)
plt.close()

print("Đã lưu:", output_path)
