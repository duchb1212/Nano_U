
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nanou.config import resolve
from nanou.evaluate import split_files, split_dataset
from nanou.predictor import load_predictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    # 1. Lay danh sach anh va mask
    config = resolve("tinyagri")
    images, masks = split_files(config, args.split)

    if not 0 <= args.index < len(images):
        raise ValueError(f"Index phai nam trong [0, {len(images)-1}]")

    image_path = Path(images[args.index])
    mask_lookup = {Path(p).name: p for p in masks}
    mask_path = mask_lookup[image_path.name]

    print("Dang xu ly anh:", image_path.name, flush=True)

    # 2. Tien xu ly bang pipeline cua Nano-U
    dataset = split_dataset(
        config,
        [str(image_path)],
        [mask_path],
        batch_size=1
    )

    image_batch, mask_batch = next(iter(dataset))

    # 3. Chay Nano-U INT8
    predictor = load_predictor(
        "models/TinyAgri/nanou.tflite"
    )

    probabilities = predictor.predict(
        image_batch.numpy()
    )[0, :, :, 0]

    prediction = probabilities >= args.threshold
    ground_truth = mask_batch.numpy()[0, :, :, 0] >= 0.5

    # 4. Tinh Road IoU cho mot anh
    intersection = np.logical_and(
        prediction, ground_truth
    ).sum()

    union = np.logical_or(
        prediction, ground_truth
    ).sum()

    iou = intersection / union if union > 0 else 1.0
    print(f"Road IoU: {iou:.4f}", flush=True)

    # 5. Khoi phuc anh RGB
    mean = np.array(config.normalization["mean"])
    std = np.array(config.normalization["std"])

    rgb = np.clip(
        image_batch.numpy()[0] * std + mean,
        0, 1
    )

    # 6. Ve 4 hinh
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")

    axes[1].imshow(ground_truth, cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(probabilities, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Probability")

    axes[3].imshow(prediction, cmap="gray")
    axes[3].set_title(f"Prediction (t={args.threshold})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    # 7. Luu ket qua
    output_dir = Path("results/TinyAgri/nanou/custom_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{args.split}_{args.index:03d}_{image_path.stem}.png"
    output_path = output_dir / filename

    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print("Da luu:", output_path.resolve(), flush=True)


if __name__ == "__main__":
    main()
