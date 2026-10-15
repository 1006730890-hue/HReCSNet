# HReCSNet

Official code and pretrained weights for **HReCSNet: Cross-Scale High-Frequency Recalibration and Region-Boundary-Error Collaborative Supervision for Lightweight ORSI Salient Object Detection**.

> Lightweight salient object detection in optical remote sensing images (ORSI-SOD). HReCSNet takes **SMT-T** as a lightweight backbone and places the design focus on the **shared progressive decoder**: a *Cross-Scale High-Frequency Recalibration* (CHFR) module performs structure-aware recalibration with a bounded residual write-back at the D3/D2 levels, and a *Region-Boundary-Error Collaborative Supervision* (RBE-CS) scheme assigns complementary objectives to several auxiliary heads during training only.

---

## Results

Our method (saliency head only at inference). Reproduced exactly from the released checkpoints.

| Dataset | Backbone | Param. (M) | FLOPs (G) | $S_\alpha \uparrow$ | $F_\beta^{\max} \uparrow$ | $E_\xi^{\max} \uparrow$ | $M \downarrow$ |
|---|---|---|---|---|---|---|---|
| ORSSD     | SMT-T | 11.90 | 18.70 | 0.9563 | 0.9296 | 0.9891 | 0.0060 |
| EORSSD    | SMT-T | 11.90 | 18.70 | 0.9473 | 0.9016 | 0.9832 | 0.0043 |
| ORSI-4199 | SMT-T | 11.90 | 18.70 | 0.8852 | 0.8850 | 0.9540 | 0.0262 |

Three-dataset mean $S_\alpha$: **0.9296**.

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

---

## Environment

To be finalised together with the released code (PyTorch + `py_sod_metrics` based evaluation).

---

## Usage

The inference and evaluation scripts (`tools/infer.py`, `tools/evaluate.py`) and the model
definitions (`models/`) will be pushed in the next commit. Inference follows the paper strictly:
only the SMT-T encoder, the shared progressive decoder and the **saliency head** are executed
(the auxiliary heads are training-time only), and evaluation uses the same protocol as the paper
($S_\alpha$ and MAE averaged per image; $F_\beta^{\max}$ and $E_\xi^{\max}$ taken as the maximum of
the cross-image averaged curve).

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
