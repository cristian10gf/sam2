"""
SAM2 segmentation pipeline. Auto-relaunches inside Docker when run from the host.

Usage (host — auto Docker):
  python3 submodules/sam2/pipeline.py --input data/images/taladro.JPG --output data/outputs --name taladro

Usage (inside container directly):
  python3 /opt/sam2/pipeline.py --input /input/image.png --output /output --name mug
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

IN_DOCKER = Path('/.dockerenv').exists()

if IN_DOCKER:
    sys.path.insert(0, '/opt/sam2')

import numpy as np
from PIL import Image


DOCKER_IMAGE   = 'sam2:x86'
CKPTS_DIR_HOST = Path.home() / 'models' / 'sam2'
SCRIPT_HOST    = Path(__file__).resolve()


def _relaunch_in_docker(args_raw: list[str]) -> None:
    """Re-exec this script inside the sam2:x86 Docker container, forwarding all args."""
    input_path  = None
    output_path = None
    for i, a in enumerate(args_raw):
        if a in ('--input',  '-input')  and i + 1 < len(args_raw):
            input_path  = Path(args_raw[i + 1]).resolve()
        if a in ('--output', '-output') and i + 1 < len(args_raw):
            output_path = Path(args_raw[i + 1]).resolve()

    if input_path is None or output_path is None:
        print('pipeline.py: --input and --output required', file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Forward every arg, replacing host paths with container paths
    forwarded = []
    i = 0
    while i < len(args_raw):
        a = args_raw[i]
        if a in ('--input', '-input') and i + 1 < len(args_raw):
            forwarded += ['--input', f'/input/{Path(args_raw[i+1]).name}']
            i += 2
        elif a in ('--output', '-output') and i + 1 < len(args_raw):
            forwarded += ['--output', '/output']
            i += 2
        else:
            forwarded.append(a)
            i += 1

    cmd = [
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '-v', f'{input_path.parent}:/input:ro',
        '-v', f'{output_path}:/output',
        '-v', f'{CKPTS_DIR_HOST}:/opt/sam2/checkpoints:ro',
        '-v', f'{SCRIPT_HOST}:/opt/sam2/pipeline.py:ro',
        '--entrypoint', 'python3',
        DOCKER_IMAGE,
        '/opt/sam2/pipeline.py',
    ] + forwarded

    print(f'Launching Docker: {" ".join(cmd)}')
    sys.exit(subprocess.run(cmd).returncode)


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


def segment(sam2, image_np: np.ndarray, points_per_side: int = 32,
            pred_iou_thresh: float = 0.88, stability_score_thresh: float = 0.95) -> list[dict]:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    gen = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    return gen.generate(image_np)


def merge_centered_masks(masks: list[dict], h: int, w: int,
                         max_area_frac: float = 0.30,
                         dilation_px: int = 30) -> dict:
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
    if not IN_DOCKER:
        _relaunch_in_docker(sys.argv[1:])

    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--name',   required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--model', default='small', choices=list(MODELS.keys()),
                        help='SAM2 model variant (default: small)')
    parser.add_argument('--points-per-side', type=int, default=32,
                        help='AMG grid density — 32=default, 64=detects thin structures (4× slower)')
    parser.add_argument('--pred-iou-thresh', type=float, default=0.88,
                        help='AMG predicted IoU threshold (default 0.88; lower=more masks, may include arc)')
    parser.add_argument('--stability-score-thresh', type=float, default=0.95,
                        help='AMG stability threshold (default 0.95; lower=more masks)')
    parser.add_argument('--all-masks', action='store_true',
                        help='Save every mask as an individual PNG')
    parser.add_argument('--max-area-frac', type=float, default=0.30,
                        help='Max fraction of frame area for seed mask — larger masks treated as background (default 0.30)')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        default=None,
                        help='Use SAM2ImagePredictor with box prompt instead of AMG. '
                             'Args: x1 y1 x2 y2 in pixel coords.')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    image_np = np.array(Image.open(args.input).convert('RGB'))
    h, w     = image_np.shape[:2]
    print(f"Input: {args.input.name} ({w}x{h})")

    print(f"Loading SAM2 {args.model}...")
    t0  = time.perf_counter()
    sam = load_model(args.model, args.device)
    print(f"Model loaded in {time.perf_counter() - t0:.2f}s")

    if args.bbox is not None:
        # ── bbox prompt mode ──────────────────────────────────────────────
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print(f"SAM2 bbox mode: {args.bbox}")
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(image_np)
        box = np.array([[args.bbox[0], args.bbox[1], args.bbox[2], args.bbox[3]]])
        t1 = time.perf_counter()
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        print(f"Predicted in {time.perf_counter() - t1:.2f}s  scores={scores.round(3)}")
        best_mask = masks[scores.argmax()].astype(bool)
        # Save outputs in the same format as AMG mode
        dummy_mask = {'segmentation': best_mask,
                      'area': int(best_mask.sum()),
                      'bbox': args.bbox,
                      'predicted_iou': float(scores.max()),
                      'stability_score': float(scores.max())}
        viz = save_visualization(image_np, [dummy_mask], args.output, args.name)
        print(f"Visualization: {viz}")
        main_out = save_isolated(image_np, best_mask, args.output, args.name)
        print(f"Main object:   {main_out}  (area={best_mask.sum()} px²)")
        segmask_out = args.output / f'{args.name}_segmask.npy'
        np.save(str(segmask_out), best_mask.astype(np.uint8))
        print(f"Segmask:       {segmask_out}")
    else:
        # ── AMG mode (existing code) ──────────────────────────────────────
        print("Segmenting...")
        t1    = time.perf_counter()
        masks = segment(sam, image_np, args.points_per_side,
                        args.pred_iou_thresh, args.stability_score_thresh)
        print(f"Found {len(masks)} masks in {time.perf_counter() - t1:.2f}s")

        # Visualization
        viz = save_visualization(image_np, masks, args.output, args.name)
        print(f"Visualization: {viz}")

        # Merge nearby centered masks → handles large fragmented objects
        best = merge_centered_masks(masks, h, w, max_area_frac=args.max_area_frac)
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
