# HReCSNet

## Requirements

python 3.10+ / pytorch 2.0+ / timm / py_sod_metrics

```bash
pip install -r requirements.txt
```

## Pretrained models

Baidu Netdisk: <https://pan.baidu.com/s/104Uko-mpUsiEdwlWJGU2Fw> (extraction code: **pnmq**)

```
weights/ORSSD.pth
weights/EORSSD.pth
weights/ORSI-4199.pth
```

## Inference

1. Download the checkpoints into `weights/`.
2. The SMT-T backbone is not vendored; clone the official SMT repository so that `external/SMT/models/smt.py` exists (or pass `--smt-root`).
3. Adjust the data path in `configs/*.yaml` (or pass `--data-root`), then run:

```bash
python tools/infer.py --config configs/eorssd.yaml \
                      --weights weights/EORSSD.pth \
                      --data-root /path/to/data_root \
                      --out-root /path/to/output
```

For the other datasets, only `--config` and `--weights` change
(`configs/orssd.yaml` + `weights/ORSSD.pth`, `configs/orsi4199.yaml` + `weights/ORSI-4199.pth`).
Saliency maps are written to `<out-root>/predictions/<DATASET>/`.

Inference runs the saliency head only; the five auxiliary heads are training-time
only and are not evaluated.

## Evaluation

```bash
python tools/evaluate.py --pred-dir <out-root>/predictions/EORSSD --gt-dir <gt-dir> --dataset EORSSD
```

The evaluation follows the same protocol as the paper.

## License

MIT License.
