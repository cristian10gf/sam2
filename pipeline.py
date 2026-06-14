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


MODELS = {
    'tiny':      ('sam2.1_hiera_tiny.pt',      'configs/sam2.1/sam2.1_hiera_t.yaml'),
    'small':     ('sam2.1_hiera_small.pt',     'configs/sam2.1/sam2.1_hiera_s.yaml'),
    'base_plus': ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
}


def load_model(model: str = 'small', device: str = 'cuda'):
    from sam2.build_sam import build_sam2
    ckpt_name, cfg = MODELS[model]
    ckpt = f'/opt/sam2/checkpoints/{ckpt_name}'
    return build_sam2(cfg, ckpt, device=device)


def segment(sam2, image_np: np.ndarray) -> list[dict]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = SAM2AutomaticMaskGenerator(sam2)
    return gen.generate(image_np)


def merge_centered_masks(masks: list[dict], h: int, w: int,
                         max_area_frac: float = 0.40,
                         dilation_px: int = 160) -> dict:
    """Iterative pixel-dilation merge starting from the most centered non-background mask.
    Absorbs any candidate mask that overlaps with the dilated union segmentation.
    More accurate than bbox padding — bridges thin gaps without pulling in distant objects.
    Masks covering >max_area_frac of the frame are excluded (background).
    """
    from scipy.ndimage import distance_transform_edt

    cx, cy = w / 2, h / 2
    total_px = h * w
    max_px = max_area_frac * total_px

    def centroid(m):
        x, y, bw, bh = m['bbox']
        return x + bw / 2, y + bh / 2

    # Exclude background masks from candidates
    candidates = [m for m in masks if m['area'] < max_px]
    if not candidates:
        candidates = masks

    # Seed = most centered candidate
    seed = min(candidates, key=lambda m: (centroid(m)[0] - cx) ** 2 + (centroid(m)[1] - cy) ** 2)

    merged_seg = seed['segmentation'].copy()
    merged_set = {id(seed)}

    # Iteratively absorb masks within dilation_px Euclidean distance of the union.
    # EDT (~merged_seg) = distance from each pixel to nearest True pixel in merged_seg.
    # O(n) vs O(n*k^2) for binary_dilation — safe at large radii.
    changed = True
    while changed:
        changed = False
        dist = distance_transform_edt(~merged_seg)
        for m in candidates:
            if id(m) in merged_set:
                continue
            if np.any(dist[m['segmentation']] <= dilation_px):
                merged_seg |= m['segmentation']
                merged_set.add(id(m))
                changed = True

    rows = np.any(merged_seg, axis=1)
    cols = np.any(merged_seg, axis=0)
    if rows.any():
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        final_bbox = [int(c0), int(r0), int(c1 - c0), int(r1 - r0)]
    else:
        x, y, bw, bh = seed['bbox']
        final_bbox = [x, y, bw, bh]

    return {
        'segmentation': merged_seg,
        'area': int(merged_seg.sum()),
        'bbox': final_bbox,
        'predicted_iou': seed['predicted_iou'],
        'stability_score': seed['stability_score'],
    }


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
    parser.add_argument('--model', default='small', choices=list(MODELS.keys()),
                        help='SAM2 model variant (default: small)')
    parser.add_argument('--all-masks', action='store_true',
                        help='Save every mask as an individual PNG')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image_np = np.array(Image.open(args.input).convert('RGB'))
    h, w     = image_np.shape[:2]
    print(f"Input: {args.input.name} ({w}x{h})")

    print(f"Loading SAM2 {args.model}...")
    t0  = time.perf_counter()
    sam = load_model(args.model, args.device)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    print("Segmenting...")
    t1    = time.perf_counter()
    masks = segment(sam, image_np)
    print(f"Found {len(masks)} masks in {time.perf_counter() - t1:.2f}s")

    # Visualization
    viz = save_visualization(image_np, masks, args.output, args.name)
    print(f"Visualization: {viz}")

    # Merge nearby centered masks → handles large fragmented objects
    best = merge_centered_masks(masks, h, w)
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
