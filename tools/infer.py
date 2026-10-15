"""Single-head HReCSNet inference.

Executes **only** the SMT-T encoder, the shared progressive decoder and the
saliency head.  The path taken is ``StructureGenerator.forward(x, only_sal=True)``
(via the :class:`~models.hrecsnet.SalOnly` wrapper): that branch returns
``{'sal': up(base.sal(feat), x)}`` before any ``edge``/``coarse``/``aux2``/
``aux3``/``unc`` head is ever evaluated, so the five auxiliary heads contribute
no computation at inference time.  They are not computed and then sliced away.

Before the first image is processed, :func:`~models.hrecsnet.check_single_head`
wraps all six heads in call counters, runs one ``only_sal=True`` forward and
asserts the counts are ``sal=1, others=0``.  The five training-only heads are
then detached from the module tree entirely.

Input is resized to 384x384; the output probability map is bilinearly resized
back to the original resolution and written as an 8-bit PNG (or ``.npy``) named
after the test image.

Examples
--------
    python tools/infer.py --config configs/eorssd.yaml --data-root D:/data \
        --out-root D:/release --weights weights/EORSSD.pth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hrecsnet import load_hrecsnet  # noqa: E402

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
#: Extension preference, identical to the research `collect_pairs`.
IMAGE_PREFERENCE = ('.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.png')
MASK_PREFERENCE = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')


def _by_stem_with_preference(paths, preferred_exts):
    preference = {ext: i for i, ext in enumerate(preferred_exts)}
    grouped = {}
    for path in paths:
        grouped.setdefault(path.stem, []).append(path)
    return {stem: sorted(items, key=lambda p: (preference.get(p.suffix.lower(), 999), p.name))[0]
            for stem, items in grouped.items()}


def collect_pairs(image_dir, mask_dir):
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f'Missing image or mask directory: {image_dir}, {mask_dir}')
    images = _by_stem_with_preference(
        [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS], IMAGE_PREFERENCE)
    masks = _by_stem_with_preference(
        [p for p in sorted(mask_dir.iterdir()) if p.suffix.lower() in IMG_EXTS], MASK_PREFERENCE)
    pairs = [(images[k], masks[k]) for k in sorted(images.keys() & masks.keys())]
    if not pairs:
        raise RuntimeError(f'No image/mask pairs found in {image_dir} and {mask_dir}')
    return pairs


def eval_transform(size):
    """Identical to the research evaluation transform."""
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                               transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path, data_root):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (Path(data_root) / path).resolve()


def resolve_checkpoint(path, data_root):
    """Relative checkpoints are looked up next to the code first, then under --data-root."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in (REPO_ROOT, data_root, Path.cwd()):
        candidate = (Path(base) / path).resolve()
        if candidate.is_file():
            return candidate
    return (REPO_ROOT / path).resolve()


@torch.no_grad()
def predict(model, pairs, size, device, out_dir, gt_dir, batch_size=1, save_format='png'):
    """Run single-head inference and write one probability map per test image."""
    transform = eval_transform(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    if gt_dir is not None:
        gt_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in tqdm(range(0, len(pairs), batch_size), desc=f'infer {len(pairs)}', mininterval=10):
        chunk = pairs[start:start + batch_size]
        tensors, shapes, stems = [], [], []
        for image_path, mask_path in chunk:
            with Image.open(image_path) as handle:
                image = handle.convert('RGB')
                tensors.append(transform(image))
            shapes.append(np.asarray(Image.open(mask_path).convert('L')).shape)
            stems.append(image_path.stem)
        batch = torch.stack(tensors).to(device)
        # ---- single-head path: only the saliency head is evaluated ----
        sal = torch.sigmoid(model(batch)['sal'])
        for index, stem in enumerate(stems):
            probability = F.interpolate(sal[index:index + 1], size=shapes[index],
                                        mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            probability = np.clip(probability, 0.0, 1.0)
            if save_format == 'npy':
                np.save(out_dir / f'{stem}.npy', probability.astype(np.float32))
            else:
                Image.fromarray(np.uint8(probability * 255)).save(out_dir / f'{stem}.png')
            rows.append({'name': stem, 'height': int(shapes[index][0]), 'width': int(shapes[index][1])})
        if gt_dir is not None:
            for image_path, mask_path in chunk:
                with Image.open(mask_path) as handle:
                    mask = np.asarray(handle.convert('L'), np.float32)
                Image.fromarray(((mask > 127).astype(np.uint8) * 255)).save(gt_dir / f'{image_path.stem}.png')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='dataset config yaml (configs/*.yaml)')
    parser.add_argument('--weights', help='checkpoint .pth (overrides config checkpoint)')
    parser.add_argument('--data-root', help='root the config data dirs are relative to')
    parser.add_argument('--image-dir', help='test image directory (overrides config)')
    parser.add_argument('--mask-dir', help='test GT directory (overrides config)')
    parser.add_argument('--out-root', help='root receiving predictions/<DATASET>/ and gt/<DATASET>/')
    parser.add_argument('--out-dir', help='exact prediction output directory (overrides --out-root)')
    parser.add_argument('--smt-root', help='path to the official SMT repository (default external/SMT)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--save-format', choices=('png', 'npy'), default='png')
    parser.add_argument('--no-export-gt', action='store_true', help='do not copy GT into gt/<DATASET>/')
    parser.add_argument('--limit', type=int, default=0, help='only the first N pairs (debugging)')
    parser.add_argument('--skip-single-head-check', action='store_true')
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    dataset = str(config.get('dataset') or config['data']['name'])
    data = config['data']
    data_root = Path(args.data_root or data.get('root', REPO_ROOT)).expanduser().resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else resolve(data['test_image_dir'], data_root)
    mask_dir = Path(args.mask_dir).resolve() if args.mask_dir else resolve(data['test_mask_dir'], data_root)
    weights = args.weights or config.get('checkpoint')
    weights = resolve_checkpoint(weights, data_root) if weights else None
    if weights is None or not Path(weights).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weights}')
    size = int(data.get('image_size', 384))

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    else:
        out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (REPO_ROOT / 'outputs')
        out_dir = out_root / 'predictions' / dataset
    gt_dir = None if args.no_export_gt else (out_dir.parent.parent / 'gt' / dataset)

    device = args.device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    pairs = collect_pairs(image_dir, mask_dir)
    if args.limit:
        pairs = pairs[:args.limit]

    model, checkpoint, report = load_hrecsnet(
        weights, device='cpu', smt_root=args.smt_root,
        verify_single_head=not args.skip_single_head_check)
    model = model.to(device)

    print(json.dumps({'dataset': dataset, 'checkpoint': str(weights),
                      'checkpoint_sha256': sha256(weights), 'checkpoint_epoch': checkpoint.get('epoch'),
                      'pairs': len(pairs), 'image_size': size, 'device': str(device),
                      'inference_path': 'StructureGenerator.forward(x, only_sal=True)',
                      'head_calls': report.get('head_calls'),
                      'detached_heads': report.get('detached_heads'),
                      'predictions': str(out_dir), 'gt': str(gt_dir)}, ensure_ascii=False))

    started = time.time()
    rows = predict(model, pairs, size, device, out_dir, gt_dir,
                   batch_size=args.batch_size, save_format=args.save_format)
    meta = {'dataset': dataset, 'checkpoint': str(weights), 'checkpoint_sha256': sha256(weights),
            'checkpoint_epoch': checkpoint.get('epoch'), 'count': len(rows),
            'image_size': size, 'device': str(device), 'save_format': args.save_format,
            'inference_path': 'StructureGenerator.forward(x, only_sal=True)',
            'evaluated_heads': ['sal'], 'head_calls': report.get('head_calls'),
            'detached_heads': report.get('detached_heads'),
            'image_dir': str(image_dir), 'mask_dir': str(mask_dir),
            'predictions': str(out_dir), 'gt': str(gt_dir),
            'seconds': round(time.time() - started, 2), 'images': rows}
    (out_dir / 'run_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'predictions': str(out_dir), 'count': len(rows),
                      'seconds': meta['seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
