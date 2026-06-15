"""
YOLO + SAM2 pipeline. Detects object with YOLOv11, then segments precisely with SAM2.
Auto-relaunches inside Docker when run from the host.

Usage (host — auto Docker):
  python3 submodules/sam2/pipeline_yolo_sam2.py --input data/images/taza/taza.jpeg --output data/outputs --name taza --class-name cup
  python3 submodules/sam2/pipeline_yolo_sam2.py --input data/images/taladro.JPG --output data/outputs --name taladro --any-class
  python3 submodules/sam2/pipeline_yolo_sam2.py --input data/images/objeto.jpg --output data/outputs --name obj --interactive

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
        # ultralytics downloads to YOLO_CONFIG_DIR on first use
        model_path = model_name
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
    """Show image with all YOLO bboxes. User clicks inside one to select it."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not detections:
        return None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image_np)
    ax.set_title('Click inside the bounding box of the object you want to segment.\nQ = quit')
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
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            if x1 <= px <= x2 and y1 <= py <= y2:
                selected[0] = d
                print(f"Selected: {d['class_name']} conf={d['conf']:.2f} bbox={[round(v) for v in d['bbox']]}")
                plt.close(fig)
                return

    def on_key(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    plt.show()
    return selected[0]


def run_sam2(image_np: np.ndarray, bbox: list[float],
             sam2_model: str, device: str) -> np.ndarray:
    """Run SAM2ImagePredictor with bbox prompt. Returns bool mask (H, W)."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt_name, cfg = SAM2_MODELS[sam2_model]
    sam = build_sam2(cfg, f'/opt/sam2/checkpoints/{ckpt_name}', device=device)
    predictor = SAM2ImagePredictor(sam)
    predictor.set_image(image_np)

    box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    # masks: (3, H, W) bool; scores: (3,)
    return masks[scores.argmax()].astype(bool)


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

    if not detections:
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
        selected = pick_detection_interactive(image_np, detections)
        if selected is None:
            print('No detection selected. Exiting.')
            sys.exit(1)
    else:
        selected = detections[0]
        print(f'Detected: {selected["class_name"]} conf={selected["conf"]:.2f} bbox={[round(v) for v in selected["bbox"]]}')

    bbox = selected['bbox']

    print(f'Running SAM2 ({args.sam2_model}) with bbox {[round(v) for v in bbox]}...')
    mask = run_sam2(image_np, bbox, args.sam2_model, args.device)
    print(f'Mask computed: {mask.sum():,} px²')

    save_outputs(image_np, mask, args.output, args.name, selected, bbox)
    print(f'\nDone — outputs in {args.output}')


if __name__ == '__main__':
    main()
