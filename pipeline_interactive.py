"""
SAM2 interactive mask selector. Auto-relaunches inside Docker when run from the host.
Left click = positive point, right click = negative point.
R = reset, Enter = save, Q = quit without saving.

Usage (host — auto Docker):
  python3 submodules/sam2/pipeline_interactive.py --input data/images/taza/taza.jpeg --output data/outputs --name taza

Usage (inside container):
  python3 /opt/sam2/pipeline_interactive.py --input /input/taza.jpeg --output /output --name taza
"""
# matplotlib backend MUST be set before any other import that might pull in pyplot
import matplotlib
matplotlib.use('TkAgg')

import argparse
import os
import subprocess
import sys
from pathlib import Path

IN_DOCKER = Path('/.dockerenv').exists()

if IN_DOCKER:
    sys.path.insert(0, '/opt/sam2')

import numpy as np
from PIL import Image

DOCKER_IMAGE   = 'sam2:x86'
CKPTS_DIR_HOST = Path.home() / 'models' / 'sam2'
SCRIPT_HOST    = Path(__file__).resolve()

MODELS = {
    'tiny':      ('sam2.1_hiera_tiny.pt',      'configs/sam2.1/sam2.1_hiera_t.yaml'),
    'small':     ('sam2.1_hiera_small.pt',     'configs/sam2.1/sam2.1_hiera_s.yaml'),
    'base_plus': ('sam2.1_hiera_base_plus.pt', 'configs/sam2.1/sam2.1_hiera_b+.yaml'),
}


def _relaunch_in_docker(args_raw: list[str]) -> None:
    display = os.environ.get('DISPLAY', '')
    if not display:
        print('ERROR: $DISPLAY not set. Run: xhost +local:docker && export DISPLAY=:0', file=sys.stderr)
        sys.exit(1)

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
        '--network', 'host',
        '-e', f'DISPLAY={display}',
        '-v', '/tmp/.X11-unix:/tmp/.X11-unix',
        '-v', f'{input_path.parent}:/input:ro',
        '-v', f'{output_path}:/output',
        '-v', f'{CKPTS_DIR_HOST}:/opt/sam2/checkpoints:ro',
        '-v', f'{SCRIPT_HOST}:/opt/sam2/pipeline_interactive.py:ro',
        '--entrypoint', 'python3',
        DOCKER_IMAGE,
        '/opt/sam2/pipeline_interactive.py',
    ] + forwarded

    xauth = os.environ.get('XAUTHORITY', '')
    if xauth:
        cmd = cmd[:3] + ['-e', f'XAUTHORITY={xauth}', '-v', f'{xauth}:{xauth}'] + cmd[3:]

    print(f'Launching Docker: {" ".join(cmd)}')
    sys.exit(subprocess.run(cmd).returncode)


def parse_args():
    parser = argparse.ArgumentParser(description='SAM2 interactive mask selector')
    parser.add_argument('--input',  required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--name',   required=True)
    parser.add_argument('--model',  default='small', choices=list(MODELS.keys()))
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    return parser.parse_args()


def main():
    if not IN_DOCKER:
        _relaunch_in_docker(sys.argv[1:])

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if not os.environ.get('DISPLAY'):
        print('ERROR: $DISPLAY not set inside container', file=sys.stderr)
        sys.exit(1)

    image_np = np.array(Image.open(args.input).convert('RGB'))
    h, w = image_np.shape[:2]
    print(f'Image loaded: {args.input.name} {w}x{h}')

    print(f'Loading SAM2 {args.model}...')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import matplotlib.pyplot as plt

    ckpt_name, cfg = MODELS[args.model]
    sam = build_sam2(cfg, f'/opt/sam2/checkpoints/{ckpt_name}', device=args.device)
    predictor = SAM2ImagePredictor(sam)

    print('Encoding image (one-time, ~1-2s)...')
    predictor.set_image(image_np)
    print('Ready. Left click = add point | Right click = exclude | R = reset | Enter = save | Q = quit')

    points       = []
    labels       = []
    current_mask = None

    fig, (ax_orig, ax_result) = plt.subplots(1, 2, figsize=(14, 7))
    fig.canvas.manager.set_window_title(
        'SAM2 Interactive — Left=add | Right=exclude | R=reset | Enter=save | Q=quit'
    )

    ax_orig.imshow(image_np)
    ax_orig.set_title('Left click = add object point\nRight click = exclude point')
    ax_orig.axis('off')

    result_im = ax_result.imshow(np.ones_like(image_np) * 255)
    ax_result.set_title('Result (no points yet)')
    ax_result.axis('off')

    plt.tight_layout()

    point_artists = []
    mask_artist   = [None]

    def _update_display():
        for a in point_artists:
            a.remove()
        point_artists.clear()
        for (px, py), lbl in zip(points, labels):
            color = 'lime' if lbl == 1 else 'red'
            artist, = ax_orig.plot(
                px, py, 'o', color=color,
                markersize=8, markeredgecolor='white', markeredgewidth=1.5
            )
            point_artists.append(artist)

        if mask_artist[0] is not None:
            mask_artist[0].remove()
            mask_artist[0] = None

        if current_mask is not None:
            result = np.ones_like(image_np) * 255
            result[current_mask] = image_np[current_mask]
            result_im.set_data(result)
            ax_result.set_title(f'Mask area: {current_mask.sum():,} px²')

            overlay = np.zeros((h, w, 4), dtype=np.float32)
            overlay[current_mask] = [0.0, 1.0, 0.0, 0.35]
            mask_artist[0] = ax_orig.imshow(overlay)
        else:
            result_im.set_data(np.ones_like(image_np) * 255)
            ax_result.set_title('Result (no points yet)')

        fig.canvas.draw_idle()

    def _run_predict():
        nonlocal current_mask
        if not points:
            return
        pts_arr = np.array(points, dtype=np.float32)
        lbl_arr = np.array(labels, dtype=np.int32)
        masks, scores, _ = predictor.predict(
            point_coords=pts_arr,
            point_labels=lbl_arr,
            multimask_output=True,
        )
        # masks: (3, H, W) bool; scores: (3,)
        current_mask = masks[scores.argmax()].astype(bool)

    def on_click(event):
        if event.inaxes is not ax_orig:
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        label = 1 if event.button == 1 else 0
        points.append([x, y])
        labels.append(label)
        _run_predict()
        _update_display()

    def on_key(event):
        nonlocal current_mask
        if event.key == 'r':
            points.clear()
            labels.clear()
            current_mask = None
            _update_display()
            fig.canvas.manager.set_window_title('SAM2 Interactive — Reset. Click on the object.')
        elif event.key == 'enter':
            if not points or current_mask is None:
                fig.canvas.manager.set_window_title(
                    'SAM2 Interactive — No points! Click the object first.'
                )
                return
            _save_and_exit()
        elif event.key == 'q':
            print('Quit without saving.')
            plt.close(fig)
            sys.exit(1)

    def _save_and_exit():
        rows = np.any(current_mask, axis=1)
        cols = np.any(current_mask, axis=0)
        r0, r1 = np.where(rows)[0][[0, -1]]
        c0, c1 = np.where(cols)[0][[0, -1]]
        pad = 20
        r0 = max(0, r0 - pad); r1 = min(h, r1 + pad)
        c0 = max(0, c0 - pad); c1 = min(w, c1 + pad)

        rgba = np.ones((h, w, 4), dtype=np.uint8) * 255
        rgba[:, :, :3] = image_np
        rgba[:, :, 3]  = current_mask.astype(np.uint8) * 255
        crop  = Image.fromarray(rgba[r0:r1, c0:c1])
        white = Image.new('RGB', crop.size, (255, 255, 255))
        white.paste(crop, mask=crop.split()[3])
        out_png = args.output / f'{args.name}.png'
        white.save(out_png)
        print(f'Saved: {out_png}')

        out_npy = args.output / f'{args.name}_segmask.npy'
        np.save(str(out_npy), current_mask.astype(np.uint8))
        print(f'Saved: {out_npy}')

        # viz: original + green mask overlay + clicked points
        viz_rgba = np.array(Image.fromarray(image_np).convert('RGBA'))
        overlay  = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[current_mask] = [0, 200, 0, 100]
        viz_rgba = np.clip(viz_rgba.astype(int) + overlay.astype(int), 0, 255).astype(np.uint8)
        viz_pil  = Image.fromarray(viz_rgba, 'RGBA').convert('RGB')
        # Draw points as colored circles
        from PIL import ImageDraw
        draw = ImageDraw.Draw(viz_pil)
        for (px, py), lbl in zip(points, labels):
            color = (0, 255, 0) if lbl == 1 else (255, 0, 0)
            r = 6
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline='white')
        out_viz = args.output / f'{args.name}_viz.png'
        viz_pil.save(out_viz)
        print(f'Saved: {out_viz}')

        plt.close(fig)
        sys.exit(0)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()
    sys.exit(1)  # window closed via X button — no save


if __name__ == '__main__':
    main()
