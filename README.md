# HReCSNet

Official code and pretrained weights for **HReCSNet: Cross-Scale High-Frequency Recalibration and Region-Boundary-Error Collaborative Supervision for Lightweight ORSI Salient Object Detection**.

> Lightweight salient object detection in optical remote sensing images (ORSI-SOD). HReCSNet takes **SMT-T** as a lightweight backbone and places the design focus on the **shared progressive decoder**: a *Cross-Scale High-Frequency Recalibration* (CHFR) module performs structure-aware recalibration with a bounded residual write-back at the D3/D2 levels, and a *Region-Boundary-Error Collaborative Supervision* (RBE-CS) scheme assigns complementary objectives to several auxiliary heads during training only.

This repository is an **inference + evaluation** release: it contains the model
definition, the single-head inference script, the evaluation script and the
checkpoint loaders. Training code, the contrastive ablation variants and the
training-only initialization fixtures of the internal study are not part of it.
The RBE-CS objective is nevertheless shipped (clearly marked *training only*)
because it is part of the method.

---

## Results

Our method evaluates the **saliency head only**; the auxiliary heads are
training-time only. Reproduced exactly from the released checkpoints
(see [`docs/reproduction.md`](docs/reproduction.md) for the full protocol,
commands and verification log).

| Dataset | Backbone | Param. (M) | FLOPs (G) | $S_\alpha \uparrow$ | $F_\beta^{\max} \uparrow$ | $E_\xi^{\max} \uparrow$ | $M \downarrow$ |
|---|---|---|---|---|---|---|---|
| ORSSD     | SMT-T | 11.90 | 18.70 | 0.9563 | 0.9296 | 0.9891 | 0.0060 |
| EORSSD    | SMT-T | 11.90 | 18.70 | 0.9473 | 0.9016 | 0.9832 | 0.0043 |
| ORSI-4199 | SMT-T | 11.90 | 18.70 | 0.8852 | 0.8850 | 0.9540 | 0.0262 |

Three-dataset mean $S_\alpha$: **0.9296**.

### Reproduced from the released checkpoints (single-head inference)

| Dataset | #Test | $S_\alpha$ | $F_\beta^{\max}$ | $E_\xi^{\max}$ | MAE |
|---|---|---|---|---|---|
| ORSSD     | 200  | 0.956256 | 0.929551 | 0.989069 | 0.005990 |
| EORSSD    | 600  | 0.947292 | 0.901629 | 0.983165 | 0.004336 |
| ORSI-4199 | 2199 | 0.885244 | 0.884964 | 0.954029 | 0.026215 |

Every column agrees with the paper table to four decimals (largest deviation
$4.9\times10^{-5}$).

---

## Pretrained weights

Download: **Baidu Netdisk** — <https://pan.baidu.com/s/104Uko-mpUsiEdwlWJGU2Fw> (extraction code: **pnmq**)

The archive contains one checkpoint per dataset:

```
weights/ORSSD.pth         # trained on ORSSD      (best epoch 30)
weights/EORSSD.pth        # trained on EORSSD     (best epoch 35)
weights/ORSI-4199.pth     # trained on ORSI-4199  (best epoch 12)
```

SHA-256 checksums (verify after download):

```
ORSSD.pth       df073b537d0d131c04cbfad16f40189930ee17b6fa70c7e70458d10dd332a223
EORSSD.pth      8bf55f503c8c7d6a7c716f60c67194948c723e7af1bfae788511fbe26c6be0bd
ORSI-4199.pth   2bbc5970abb587249528bf005a685c7e1e06c96ce5e0e7cd80eb4c48b286c2ab
```

Each checkpoint stores its own training configuration and the recorded best metrics
(`config.data.name`, `epoch`, `metrics`), so the mapping above can be verified directly.
The trained parameters live under the `'generator'` key and are tagged with
`format='smt_structure_20261006_v1'`. They load into `StructureGenerator` with
`strict=True` — no key rewriting or filtering is needed.

---

## Environment

* Python ≥ 3.10
* PyTorch ≥ 2.0 (+ matching `torchvision`), CUDA optional (CPU inference works)
* `timm` (required by the official SMT backbone)
* `py_sod_metrics` (evaluation), `numpy`, `Pillow`, `PyYAML`, `tqdm`

```bash
pip install -r requirements.txt
```

The SMT-T backbone is **not** vendored here. Clone the official repository of
[SMT](https://github.com/AeroVolution/SMT) (or the version used by the paper) and
place it so that `<repo>/external/SMT/models/smt.py` exists:

```bash
cd HReCSNet
git clone <official-SMT-repository-url> external/SMT
```

Alternatively point anywhere with `--smt-root /path/to/SMT` or the `SMT_ROOT`
environment variable. The ImageNet-pretrained SMT-T weights
(`external/SMT/smt_tiny_full.pth`) are **not needed for inference**, because the
released checkpoints contain the complete backbone: the loader builds the
architecture and then overwrites every parameter from the checkpoint.

---

## Project structure

```
HReCSNet/
├── models/
│   ├── chfr.py         # up / highpass / cgn / ResidualAdapter / CHFR
│   ├── decoder.py      # SMT-T loading + minimal shared progressive decoder (adapt/fuse1-3/refine/six heads)
│   └── hrecsnet.py     # GroupGenerator, StructureGenerator (CHFR at D3/D2), SalOnly, checkpoint loader
├── losses/
│   └── rbe_cs.py       # GROUPS / ALL_HEADS / grouped_loss  (TRAINING ONLY)
├── tools/
│   ├── infer.py        # single-head saliency inference, 384x384 in -> original-size probability map
│   └── evaluate.py     # paper-protocol evaluation (py_sod_metrics)
├── configs/
│   ├── orssd.yaml
│   ├── eorssd.yaml
│   └── orsi4199.yaml
├── docs/
│   └── reproduction.md
├── external/SMT/       # official SMT repository (clone it yourself; git-ignored)
├── requirements.txt
├── weights_sha256.txt
├── LICENSE
└── README.md
```

Data directories are **not** shipped. Every path under `data:` in a config is
relative to `--data-root`, which must be the directory that contains `dataset/`
(the same convention as the paper's training scripts).

---

## Usage

### 1. Single-head inference

```bash
# ORSSD (200 test images)
python tools/infer.py --config configs/orssd.yaml \
    --data-root /path/to/data_root \
    --weights weights/ORSSD.pth \
    --out-root /path/to/release

# EORSSD (600 test images)
python tools/infer.py --config configs/eorssd.yaml \
    --data-root /path/to/data_root \
    --weights weights/EORSSD.pth \
    --out-root /path/to/release

# ORSI-4199 (2199 test images)
python tools/infer.py --config configs/orsi4199.yaml \
    --data-root /path/to/data_root \
    --weights weights/ORSI-4199.pth \
    --out-root /path/to/release
```

Each run writes `<out-root>/predictions/<DATASET>/<name>.png` (8-bit probability
maps in `[0, 255]`, original image resolution, one file per test image) and also
exports the binarized ground truth to `<out-root>/gt/<DATASET>/<name>.png` so the
result is self-contained and re-evaluable. Use `--no-export-gt` to skip the GT
export and `--save-format npy` for float probability maps.

**Inference is strictly single-head.** `tools/infer.py` calls
`StructureGenerator.forward(x, only_sal=True)` through the `SalOnly` wrapper, so
the path is *encoder → shared progressive decoder (fuse1/2/3 + refine) → saliency
head* and nothing else. The five auxiliary heads (`edge`, `coarse`, `aux2`,
`aux3`, `unc`) are never evaluated — they are not computed and then sliced away.
The loader proves this at start-up by wrapping all six heads in call counters,
running one `only_sal=True` forward and asserting `{'sal': 1, others: 0}`; the
five auxiliary heads are then detached from the module tree entirely.

### 2. Evaluation

```bash
python tools/evaluate.py --pred-dir /path/to/release/predictions/ORSSD \
                         --gt-dir   /path/to/release/gt/ORSSD
# or, deriving the directories from a config:
python tools/evaluate.py --config configs/orssd.yaml --out-root /path/to/release
```

Reported JSON: `Smeasure`, `MAE`, `maxF`, `meanF`, `adpF`, `maxE`, `meanE`,
`adpE`, `wF`.

| Flag | Meaning |
|---|---|
| `--limit` | evaluate only the first N pairs (debugging) |
| `--output-json` | also write the metric dictionary to a file |

### Evaluation protocol

Identical to the paper and to the research implementation:

* $S_\alpha$ (S-measure) and MAE are **per-image values averaged over the test set**;
* $F_\beta^{\max}$ and $E_\xi^{\max}$ are the **maximum of the cross-image averaged
  curve** (`max(fm['curve'])`, `max(em['curve'])`), not the maximum over per-image
  maxima;
* predictions are 8-bit maps in `[0, 255]` and GT masks are binarized at
  `value > 127`.

---

## Citation

```bibtex
@article{hrecsnet,
  title   = {HReCSNet: Cross-Scale High-Frequency Recalibration and Region-Boundary-Error Collaborative Supervision for Lightweight ORSI Salient Object Detection},
  author  = {Anonymous},
  journal = {To appear},
  year    = {2026}
}
```

(The citation entry will be updated once the paper is accepted.)

---

## License

Released under the MIT License — see [LICENSE](LICENSE).

The SMT backbone is third-party code distributed under its own license; clone it
from its official repository and comply with that license.
