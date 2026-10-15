"""HReCSNet evaluation.

Metric protocol (identical to the paper and to the research implementation)
--------------------------------------------------------------------------
``py_sod_metrics`` accumulates one entry per image:

* :math:`S_\\alpha` (S-measure) and MAE are **per-image values averaged over the
  test set** (the ``step``/``get_results`` contract of ``Smeasure`` and ``MAE``);
* :math:`F_\\beta^{\\max}` and :math:`E_\\xi^{\\max}` are the **maximum of the
  cross-image averaged curve**, i.e. ``max(fm['curve'])`` / ``max(em['curve'])``,
  where each curve point is already the mean over images.

This is the wrapper used by the research code (``orsi_sod/metrics.py``); it is
reproduced here verbatim so the released numbers can be re-derived from the
saved probability maps alone.

Examples
--------
    python tools/evaluate.py --pred-dir D:/release/predictions/ORSSD \
        --gt-dir D:/release/gt/ORSSD
    python tools/evaluate.py --config configs/eorssd.yaml --out-root D:/release
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
#: Metrics reported with six decimals; the first four are the paper columns.
REPORT_KEYS = ('Smeasure', 'MAE', 'maxF', 'meanF', 'adpF', 'maxE', 'meanE', 'adpE', 'wF')


class PySODMetric:
    """Per-image py_sod_metrics accumulator (verbatim research wrapper)."""

    def __init__(self):
        self.mae = MAE()
        self.fm = Fmeasure()
        self.em = Emeasure()
        self.sm = Smeasure()
        self.wfm = WeightedFmeasure()

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = np.squeeze(pred)
        gt = np.squeeze(gt)
        pred_u8 = np.clip(pred * 255 if pred.max() <= 1 else pred, 0, 255).astype(np.uint8)
        gt_u8 = ((gt > 0.5).astype(np.uint8) * 255)
        self.mae.step(pred=pred_u8, gt=gt_u8)
        self.fm.step(pred=pred_u8, gt=gt_u8)
        self.em.step(pred=pred_u8, gt=gt_u8)
        self.sm.step(pred=pred_u8, gt=gt_u8)
        self.wfm.step(pred=pred_u8, gt=gt_u8)

    def compute(self) -> dict[str, float]:
        # maxF/maxE: maximum of the cross-image averaged curve (paper protocol).
        fm = self.fm.get_results()["fm"]
        em = self.em.get_results()["em"]
        return {
            "Smeasure": float(self.sm.get_results()["sm"]),
            "MAE": float(self.mae.get_results()["mae"]),
            "maxF": float(np.max(fm["curve"])),
            "meanF": float(np.mean(fm["curve"])),
            "adpF": float(fm["adp"]),
            "maxE": float(np.max(em["curve"])),
            "meanE": float(np.mean(em["curve"])),
            "adpE": float(em["adp"]),
            "wF": float(self.wfm.get_results()["wfm"]),
        }


def score(metrics: dict[str, float]) -> float:
    return metrics["Smeasure"] + metrics["maxF"] + metrics["maxE"] + metrics["wF"] - metrics["MAE"]


def load_prediction(path):
    """Load a probability map.  ``.npy`` is used as-is, images are scaled to [0, 1]."""
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return np.asarray(np.load(path), dtype=np.float32)
    with Image.open(path) as handle:
        return np.asarray(handle.convert('L'), dtype=np.float32) / np.float32(255.0)


def load_gt(path):
    """Load a GT mask and binarize it at 127, exactly as the research test loop does
    (``target = (gt > 127)``).  A few ORSI-4199 masks are not strictly 0/255, so the
    threshold matters: judging them with ``> 0.5`` would silently inflate MAE.
    ``PySODMetric.update`` then applies its own idempotent ``gt > 0.5`` binarization."""
    with Image.open(path) as handle:
        gt = np.asarray(handle.convert('L'), dtype=np.float32)
    return (gt > 127).astype(np.float32)


def index_by_stem(directory, exts=IMG_EXTS):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Not a directory: {directory}')
    return {p.stem: p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts}


def evaluate(pred_dir, gt_dir, limit=0):
    predictions = index_by_stem(pred_dir)
    ground_truths = index_by_stem(gt_dir)
    missing = sorted(set(ground_truths) - set(predictions))
    extra = sorted(set(predictions) - set(ground_truths))
    if missing or extra:
        raise ValueError(f'Prediction/GT mismatch: {len(missing)} missing (e.g. {missing[:5]}), '
                         f'{len(extra)} extra (e.g. {extra[:5]})')
    names = sorted(ground_truths)
    if limit:
        names = names[:limit]
    metric = PySODMetric()
    for name in names:
        metric.update(load_prediction(predictions[name]), load_gt(ground_truths[name]))
    results = metric.compute()
    results['score'] = score(results)
    results['count'] = len(names)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred-dir', help='directory of predicted probability maps')
    parser.add_argument('--gt-dir', help='directory of ground-truth masks')
    parser.add_argument('--config', help='dataset config yaml; derives <out-root>/predictions|gt/<DATASET>')
    parser.add_argument('--out-root', help='release root used with --config')
    parser.add_argument('--dataset', help='override the dataset name used with --config')
    parser.add_argument('--output-json', help='write the metric dictionary to this JSON file')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    if args.config:
        import yaml
        config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
        dataset = args.dataset or str(config.get('dataset') or config['data']['name'])
        out_root = Path(args.out_root).expanduser().resolve() if args.out_root else (REPO_ROOT / 'outputs')
        pred_dir = args.pred_dir or (out_root / 'predictions' / dataset)
        gt_dir = args.gt_dir or (out_root / 'gt' / dataset)
    else:
        dataset = args.dataset or Path(args.pred_dir).name
        pred_dir, gt_dir = args.pred_dir, args.gt_dir
    if not pred_dir or not gt_dir:
        parser.error('provide either --config or both --pred-dir and --gt-dir')

    results = evaluate(pred_dir, gt_dir, limit=args.limit)
    payload = {'dataset': dataset, 'predictions': str(pred_dir), 'gt': str(gt_dir),
               'protocol': 'Smeasure/MAE per-image mean; maxF/maxE = max of the cross-image averaged curve',
               **results}
    summary = {'dataset': dataset, 'count': results['count']}
    summary.update({key: round(results[key], 6) for key in REPORT_KEYS})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
