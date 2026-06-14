"""
Runs inside the Docker container (x86 or Jetson).
Segments the most prominent object in an image using SAM2 small (default params).
Outputs a visualization + per-mask PNGs with white background, ready for TripoSR/TRELLIS.

Usage:
  python pipeline.py --input /input/image.png --output /output --name mug
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, '/opt/sam2')


def load_model(device: str = 'cuda'):
    from sam2.build_sam import build_sam2
    ckpt = '/opt/sam2/checkpoints/sam2.1_hiera_small.pt'
    cfg  = 'configs/sam2.1/sam2.1_hiera_s.yaml'
    return build_sam2(cfg, ckpt, device=device)


def segment(sam2, image_np: np.ndarray) -> list[dict]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = SAM2AutomaticMaskGenerator(sam2)
    return gen.generate(image_np)


def most_centered_mask(masks: list[dict], h: int, w: int) -> dict:
    """Select the mask whose centroid is closest to the image center."""
    cx, cy = w / 2, h / 2
    def dist(m):
        x, y, bw, bh = m['bbox']
        return (x + bw / 2 - cx) ** 2 + (y + bh / 2 - cy) ** 2
    return min(masks, key=dist)


def save_visualization(image_np, masks, output_path: Path, name: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(image_np)
    for mask in sorted(masks, key=lambda x: x['area'], reverse=True):
        color   = np.concatenate([np.random.random(3), [0.45]])
        h, w    = mask['segmentation'].shape
        overlay = np.zeros((h, w, 4))
        overlay[mask['segmentation']] = color
        axes[1].imshow(overlay)
        x, y, bw, bh = mask['bbox']
        axes[1].add_patch(patches.Rectangle((x, y), bw, bh, linewidth=1, edgecolor='white', facecolor='none'))
    axes[1].set_title(f'SAM2 small — {len(masks)} masks')
    axes[1].axis('off')

    plt.tight_layout()
    viz_path = output_path / f'{name}_viz.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    return viz_path


def save_isolated(image_np, mask_bool, output_path: Path, suffix: str) -> Path:
    """Save object on white background as PNG."""
    h, w = mask_bool.shape
    rgba = np.ones((h, w, 4), dtype=np.uint8) * 255
    rgba[:, :, :3] = image_np
    rgba[:, :, 3]  = mask_bool.astype(np.uint8) * 255

    # Crop to bounding box with small padding
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 20
    r0, r1 = max(0, r0 - pad), min(h, r1 + pad)
    c0, c1 = max(0, c0 - pad), min(w, c1 + pad)

    crop = Image.fromarray(rgba[r0:r1, c0:c1])
    # White background version (for TripoSR/TRELLIS)
    white = Image.new('RGB', crop.size, (255, 255, 255))
    white.paste(crop, mask=crop.split()[3])

    out = output_path / f'{suffix}.png'
    white.save(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--name',   required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--all-masks', action='store_true',
                        help='Save every mask as an individual PNG')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image_np = np.array(Image.open(args.input).convert('RGB'))
    h, w     = image_np.shape[:2]
    print(f"Input: {args.input.name} ({w}x{h})")

    print("Loading SAM2 small...")
    t0  = time.perf_counter()
    sam = load_model(args.device)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    print("Segmenting...")
    t1    = time.perf_counter()
    masks = segment(sam, image_np)
    print(f"Found {len(masks)} masks in {time.perf_counter() - t1:.2f}s")

    # Visualization
    viz = save_visualization(image_np, masks, args.output, args.name)
    print(f"Visualization: {viz}")

    # Most centered mask → main output for 3D pipeline
    best = most_centered_mask(masks, h, w)
    main_out = save_isolated(image_np, best['segmentation'], args.output, args.name)
    print(f"Main object:   {main_out}  (area={best['area']} px²)")

    # Full-size binary mask (bool, original image dimensions) for depth masking
    segmask_out = args.output / f'{args.name}_segmask.npy'
    np.save(str(segmask_out), best['segmentation'].astype(np.uint8))
    print(f"Segmask:       {segmask_out}")

    # All masks individually
    if args.all_masks:
        ranked = sorted(masks, key=lambda x: x['area'], reverse=True)
        for i, m in enumerate(ranked):
            save_isolated(image_np, m['segmentation'], args.output, f'{args.name}_mask{i:02d}')
        print(f"Saved {len(masks)} individual masks")

    print(f"\nDone — outputs in {args.output}")


if __name__ == '__main__':
    main()
