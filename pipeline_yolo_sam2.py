"""
YOLO + SAM2 pipeline. Detects object with YOLOv11, then segments precisely with SAM2.
Auto-relaunches inside Docker when run from the host.

Usage (host — auto Docker):
  python3 submodules/sam2/pipeline_yolo_sam2.py --input data/images/taza/taza.jpeg --output data/outputs --name taza --class-name cup
  python3 submodules/sam2/pipeline_yolo_sam2.py --input data/images/taladro.JPG --output data/outputs --name taladro --any-class
  python3 submodules/sam2/pipeline_yolo_sam2.py --input data/images/objeto.jpg --output data/outputs --name obj --interactive
  python3 submodules/sam2/pipeline_yolo_sam2.py --input data/images/objeto.jpg --output data/outputs --name obj --point 320,240

Usage (inside container):
  python3 /opt/sam2/pipeline_yolo_sam2.py --input /input/taza.jpeg --output /output --name taza --class-name cup
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

IN_DOCKER = Path('/.dockerenv').exists()

if IN_DOCKER:
    sys.path.insert(0, '/opt/sam2')
    os.environ.setdefault('YOLO_CONFIG_DIR', '/opt/yolo')

import numpy as np
from PIL import Image

DOCKER_IMAGE   = 'sam2:x86'
CKPTS_DIR_HOST = Path.home() / 'models' / 'sam2'
YOLO_DIR_HOST  = Path.home() / 'models' / 'yolo'
SCRIPT_HOST    = Path(__file__).resolve()

SAM2_MODELS = {
    'tiny':      ('sam2.1_hiera_tiny.pt',      'configs/sam2.1/sam2.1_hiera_t.yaml'),
    'small':     ('sam2.1_hiera_small.pt',     'configs/sam2.1/sam2.1_hiera_s.yaml'),
    'base_plus': ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
}

YOLO_MODEL = 'yolo11n.pt'

DEFAULT_POINT_BOX_SIZE = 100  # px — fallback bbox half-side when YOLO finds no detection


def _relaunch_in_docker(args_raw: list[str]) -> None:
    input_path = output_path = None
    for i, a in enumerate(args_raw):
        if a in ('--input', '-input') and i + 1 < len(args_raw):
            input_path = Path(args_raw[i + 1]).resolve()
        if a in ('--output', '-output') and i + 1 < len(args_raw):
            output_path = Path(args_raw[i + 1]).resolve()

    if input_path is None or output_path is None:
        print('ERROR: --input and --output are required', file=sys.stderr)
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    YOLO_DIR_HOST.mkdir(parents=True, exist_ok=True)

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

    display = os.environ.get('DISPLAY', '')
    cmd = [
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '-v', f'{input_path.parent}:/input:ro',
        '-v', f'{output_path}:/output',
        '-v', f'{CKPTS_DIR_HOST}:/opt/sam2/checkpoints:ro',
        '-v', f'{YOLO_DIR_HOST}:/opt/yolo',
        '-v', f'{SCRIPT_HOST}:/opt/sam2/pipeline_yolo_sam2.py:ro',
        '-e', 'YOLO_CONFIG_DIR=/opt/yolo',
        '--entrypoint', 'python3',
        DOCKER_IMAGE,
        '/opt/sam2/pipeline_yolo_sam2.py',
    ] + forwarded

    # X11 only needed for --interactive mode
    if '--interactive' in args_raw and display:
        cmd = cmd[:3] + [
            '--network', 'host',
            '-e', f'DISPLAY={display}',
            '-v', '/tmp/.X11-unix:/tmp/.X11-unix',
        ] + cmd[3:]
        xauth = os.environ.get('XAUTHORITY', '')
        if xauth:
            cmd = cmd[:3] + ['-e', f'XAUTHORITY={xauth}', '-v', f'{xauth}:{xauth}'] + cmd[3:]

    print(f'Launching Docker: {" ".join(cmd)}')
    sys.exit(subprocess.run(cmd).returncode)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO + SAM2 segmentation pipeline')
    parser.add_argument('--input',      required=True, type=Path)
    parser.add_argument('--output',     required=True, type=Path)
    parser.add_argument('--name',       required=True)
    parser.add_argument('--sam2-model', default='small', choices=list(SAM2_MODELS.keys()))
    parser.add_argument('--device',     default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--yolo-model', default=YOLO_MODEL,
                        help='YOLOv11 model variant (default: yolo11n.pt)')
    parser.add_argument('--conf',       type=float, default=0.25,
                        help='YOLO confidence threshold (default 0.25)')
    parser.add_argument('--fill-holes', action='store_true',
                        help='Fill enclosed cavities in mask (500-20K px²). '
                             'Use for objects with hollow surfaces (e.g. earpad interiors). '
                             'Avoid for objects with intentional holes (e.g. cup handle).')

    # Object selection — mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--class-name',  type=str,
                       help='COCO class name to detect (e.g. cup, bottle, scissors)')
    group.add_argument('--class-id',    type=int,
                       help='COCO class ID to detect (e.g. 41 for cup)')
    group.add_argument('--any-class',   action='store_true',
                       help='Use highest-confidence detection regardless of class')
    group.add_argument('--interactive', action='store_true',
                       help='Show all YOLO detections, user picks one via matplotlib')
    group.add_argument('--point',       type=str, metavar='U,V',
                       help='Select nearest YOLO bbox to pixel (U,V). No X11 needed.')
    return parser.parse_args()


def run_yolo(image_np: np.ndarray, model_name: str, conf: float,
             class_name: str | None, class_id: int | None, any_class: bool,
             device: str) -> list[dict]:
    """Run YOLOv11 on image_np. Returns list of detections sorted by confidence desc.
    Each detection: {'bbox': [x1,y1,x2,y2], 'conf': float, 'class_id': int, 'class_name': str}
    """
    from ultralytics import YOLO
    model_path = Path('/opt/yolo') / model_name
    if not model_path.exists():
        # Download to cache dir so it persists across runs via the host volume mount.
        import urllib.request
        url = f'https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}'
        print(f'Downloading {model_name} → {model_path}')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, model_path)
    model = YOLO(str(model_path))

    results = model(image_np, conf=conf, device=device, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cid  = int(box.cls.item())
            cname = model.names[cid]
            if class_id is not None and cid != class_id:
                continue
            if class_name is not None and cname.lower() != class_name.lower():
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                'bbox':       [x1, y1, x2, y2],
                'conf':       float(box.conf.item()),
                'class_id':   cid,
                'class_name': cname,
            })

    detections.sort(key=lambda d: d['conf'], reverse=True)
    return detections


def pick_detection_interactive(image_np: np.ndarray, detections: list[dict]) -> dict | None:
    """Show image with all YOLO bboxes. User clicks inside one to select it.

    If the user clicks outside every detected bounding box (e.g. on an object
    that YOLO missed), a fallback bbox of DEFAULT_POINT_BOX_SIZE px is created
    centred on the click and returned as a synthetic detection so SAM2 can still
    segment that object.
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    image_h, image_w = image_np.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_np)
    if detections:
        ax.set_title(
            'Click inside a bounding box — or anywhere on an undetected object.\nQ = quit'
        )
    else:
        ax.set_title('No YOLO detections. Click anywhere on the object to segment.\nQ = quit')
    ax.axis('off')

    colors = plt.cm.Set1.colors
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d['bbox']
        color = colors[i % len(colors)]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"{i}: {d['class_name']} {d['conf']:.2f}",
                color=color, fontsize=9, backgroundcolor='black')

    selected = [None]

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        px, py = event.xdata, event.ydata
        # Prefer clicking inside a detected box.
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            if x1 <= px <= x2 and y1 <= py <= y2:
                selected[0] = d
                print(f"Selected: {d['class_name']} conf={d['conf']:.2f} bbox={[round(v) for v in d['bbox']]}")
                plt.close(fig)
                return
        # Fallback: click outside all boxes → synthetic bbox centred on click.
        half = DEFAULT_POINT_BOX_SIZE // 2
        fb = [
            float(max(0, int(px) - half)),
            float(max(0, int(py) - half)),
            float(min(image_w, int(px) + half)),
            float(min(image_h, int(py) + half)),
        ]
        selected[0] = {'bbox': fb, 'conf': 0.0, 'class_id': -1, 'class_name': 'manual_click'}
        print(f"No YOLO box at click ({int(px)}, {int(py)}); using fallback bbox {[round(v) for v in fb]}")
        plt.close(fig)

    def on_key(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
    return selected[0]


def pick_detection_by_point(detections: list[dict], u: float, v: float,
                             image_w: int, image_h: int) -> list[float]:
    """Return the YOLO detection bbox that best covers pixel (u, v).

    Selection order:
      1. Detections whose bbox *contains* (u, v) — smallest area wins (tightest
         fit avoids large background boxes, e.g. a bin surface covering most of
         the frame whose centre happens to be near the projected centroid).
      2. Fixed square centred on (u, v) — used when no detection contains the
         point or when detections is empty.  Nearest-centre fallback is
         intentionally avoided: a large background bbox can have its centre
         arbitrarily close to the projected object centroid.
    """
    containing = [d for d in detections if (
        d['bbox'][0] <= u <= d['bbox'][2] and
        d['bbox'][1] <= v <= d['bbox'][3]
    )]
    if containing:
        return min(
            containing,
            key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]),
        )['bbox']

    half = DEFAULT_POINT_BOX_SIZE // 2
    x1 = max(0, int(u) - half)
    y1 = max(0, int(v) - half)
    x2 = min(image_w, int(u) + half)
    y2 = min(image_h, int(v) + half)
    return [float(x1), float(y1), float(x2), float(y2)]


def run_sam2(image_np: np.ndarray, bbox: list[float],
             sam2_model: str, device: str, fill_holes: bool = False) -> np.ndarray:
    """Run SAM2ImagePredictor with bbox prompt. Returns bool mask (H, W)."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_name, cfg = SAM2_MODELS[sam2_model]
    sam = build_sam2(cfg, f'/opt/sam2/checkpoints/{ckpt_name}', device=device)
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(image_np)

    box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    mask = masks[scores.argmax()].astype(bool)

    if fill_holes:
        # Fill enclosed holes in 500-20K px² band: targets hollow surfaces like
        # earpad interiors. Skips large open spaces (>20K) and noise (<500).
        # Do NOT use for objects with intentional holes (cup handle, etc.).
        from scipy.ndimage import label, binary_fill_holes
        filled_all = binary_fill_holes(mask)
        holes = filled_all & ~mask
        hole_labeled, n_holes = label(holes)
        for i in range(1, n_holes + 1):
            hole = hole_labeled == i
            if 500 <= hole.sum() <= 20_000:
                mask |= hole

    return mask


def save_outputs(image_np: np.ndarray, mask: np.ndarray,
                 output_path: Path, name: str,
                 detection: dict, bbox: list[float]) -> None:
    """Save <name>.png (white bg), <name>_segmask.npy, <name>_viz.png."""
    h, w = mask.shape

    # White-background crop (same as pipeline.py save_isolated)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    pad = 20
    r0 = max(0, r0 - pad); r1 = min(h, r1 + pad)
    c0 = max(0, c0 - pad); c1 = min(w, c1 + pad)

    rgba = np.ones((h, w, 4), dtype=np.uint8) * 255
    rgba[:, :, :3] = image_np
    rgba[:, :, 3]  = mask.astype(np.uint8) * 255
    crop  = Image.fromarray(rgba[r0:r1, c0:c1])
    white = Image.new('RGB', crop.size, (255, 255, 255))
    white.paste(crop, mask=crop.split()[3])
    out_png = output_path / f'{name}.png'
    white.save(out_png)
    print(f'Saved: {out_png}  (area={mask.sum():,} px²)')

    # Segmask
    out_npy = output_path / f'{name}_segmask.npy'
    np.save(str(out_npy), mask.astype(np.uint8))
    print(f'Saved: {out_npy}')

    # Viz: original + mask overlay + YOLO bbox
    from PIL import ImageDraw
    viz = Image.fromarray(image_np).convert('RGBA')
    overlay = Image.fromarray(
        np.where(mask[:, :, None], np.array([0, 200, 0, 100], dtype=np.uint8),
                 np.zeros((h, w, 4), dtype=np.uint8)).astype(np.uint8), 'RGBA'
    )
    viz.alpha_composite(overlay)
    viz_rgb = viz.convert('RGB')
    draw = ImageDraw.Draw(viz_rgb)
    x1, y1, x2, y2 = [round(v) for v in bbox]
    draw.rectangle([x1, y1, x2, y2], outline='yellow', width=3)
    draw.text((x1 + 4, y1 + 4),
              f"{detection['class_name']} {detection['conf']:.2f}",
              fill='yellow')
    out_viz = output_path / f'{name}_viz.png'
    viz_rgb.save(out_viz)
    print(f'Saved: {out_viz}')


def main():
    if not IN_DOCKER:
        _relaunch_in_docker(sys.argv[1:])

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image_np = np.array(Image.open(args.input).convert('RGB'))
    h, w = image_np.shape[:2]
    print(f'Image loaded: {args.input.name} {w}x{h}')

    print(f'Running YOLO ({args.yolo_model})...')
    detections = run_yolo(
        image_np, args.yolo_model, args.conf,
        class_name=args.class_name,
        class_id=args.class_id,
        any_class=args.any_class,
        device=args.device,
    )

    if not detections and not args.point and not args.interactive:
        filter_desc = (f'class_name={args.class_name}' if args.class_name else
                       f'class_id={args.class_id}'   if args.class_id   else 'any class')
        print(f'ERROR: No YOLO detections found ({filter_desc}, conf>={args.conf})', file=sys.stderr)
        print('Try: --any-class, lower --conf, or --interactive to see all detections', file=sys.stderr)
        sys.exit(1)

    # Select detection
    if args.interactive:
        if not os.environ.get('DISPLAY'):
            print('ERROR: --interactive requires $DISPLAY (X11)', file=sys.stderr)
            sys.exit(1)
        selected_meta = pick_detection_interactive(image_np, detections)
        if selected_meta is None:
            print('No detection selected. Exiting.')
            sys.exit(1)
        bbox = selected_meta['bbox']
    elif args.point:
        try:
            pu, pv = (float(x) for x in args.point.split(','))
        except ValueError:
            print(f'ERROR: --point must be "U,V" floats, got: {args.point}', file=sys.stderr)
            sys.exit(1)
        bbox = pick_detection_by_point(detections, pu, pv, w, h)
        print(f'Point prompt ({pu:.1f},{pv:.1f}) → bbox {[round(b) for b in bbox]}')
        selected_meta = {'class_name': 'point_prompt', 'conf': 1.0, 'bbox': bbox}
    else:
        selected_meta = detections[0]
        print(f'Detected: {selected_meta["class_name"]} conf={selected_meta["conf"]:.2f} bbox={[round(v) for v in selected_meta["bbox"]]}')
        bbox = selected_meta['bbox']

    print(f'Running SAM2 ({args.sam2_model}) with bbox {[round(b) for b in bbox]}...')
    mask = run_sam2(image_np, bbox, args.sam2_model, args.device, fill_holes=args.fill_holes)
    print(f'Mask computed: {mask.sum():,} px²')

    save_outputs(image_np, mask, args.output, args.name, selected_meta, bbox)
    print(f'\nDone — outputs in {args.output}')


if __name__ == '__main__':
    main()
