# Reproduction record

Acceptance run performed on the author's machine to prove that this release
reproduces the paper numbers from the released checkpoints alone.

Everything below was produced by *this* repository's scripts; no research-directory
file was modified, and no checkpoint or prediction of the research tree was
overwritten.

---

## 1. Environment

| Item | Value |
|---|---|
| OS | Windows, PowerShell |
| Python | 3.11.4 (Anaconda, `C:\Users\lsq17\anaconda3\python.exe`) |
| torch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| CUDA | available (`cuda:0` used for inference; evaluation is CPU) |
| SMT backbone | official SMT repository, `external/SMT/models/smt.py` |
| Evaluation | `py_sod_metrics` |

Install:

```bash
pip install -r requirements.txt
cd HReCSNet && git clone <official-SMT-repository-url> external/SMT
```

The SMT-T ImageNet weights (`external/SMT/smt_tiny_full.pth`) are **not** required
for inference: the released checkpoints contain the whole backbone and are loaded
with `strict=True`.

---

## 2. Weight <-> training-run mapping (verified by SHA-256)

The released `weights/*.pth` files were hashed and compared with the checkpoints of
the corresponding training runs. They are byte-identical:

| Released file | SHA-256 | Identical to | Trained epoch |
|---|---|---|---|
| `weights/ORSSD.pth` | `df073b537d0d131c04cbfad16f40189930ee17b6fa70c7e70458d10dd332a223` | ORSSD run `best_mae.pth` (epoch 30) | 30 |
| `weights/EORSSD.pth` | `8bf55f503c8c7d6a7c716f60c67194948c723e7af1bfae788511fbe26c6be0bd` | EORSSD run `best_mae.pth` (epoch 35) | 35 |
| `weights/ORSI-4199.pth` | `2bbc5970abb587249528bf005a685c7e1e06c96ce5e0e7cd80eb4c48b286c2ab` | ORSI-4199 run `best_mae.pth` (epoch 12) | 12 |

Checkpoint internals:

* `format = 'smt_structure_20261006_v1'`;
* trained parameters under the `generator` key (488 tensors);
* top-level parameter groups: `base.*`, `attn3.*`, `attn2.*`, `sdpg1.*`, `sdpg2.*`;
* `base.*` contains `encoder`, `adapt`, `fuse1/2/3`, `refine`, and the six heads
  `sal`/`edge`/`coarse`/`aux2`/`aux3`/`unc`.

---

## 3. Checkpoint adaptation performed

**None beyond reading `ck['generator']`.** The released loader builds the same
module tree as the training code, so `load_state_dict(state, strict=True)` succeeds
with the raw checkpoint key set — no prefix rewriting, no key filtering, no
missing/unexpected keys.

Two notes on deliberate design choices (no mathematics changed):

1. The training code ran a "reference initialization fingerprint" self-check
   (`Generator initialization differs from reference`) inside
   `initialize_generator`. That check belongs to the training pipeline (it guards
   paired initialization) and is **not part of this release**: `GroupGenerator`
   builds the architecture directly and every parameter is then overwritten by the
   checkpoint. This is a structural bypass, not a monkey-patched `sha256` comparison —
   no assertion was suppressed at runtime.
2. `SDPG` (`sdpg1`/`sdpg2`) is kept in the module tree only so the state dict
   matches the checkpoint exactly. The CHFR branch returns before any SDPG code is
   reached (verified below), so it is never executed at inference. The study's
   `biformer_sge` module slots are created as `Identity` exactly as the training
   code does, carry no parameters and are absent from the checkpoints.

Checkpoint prefix report (from `tools/infer.py`):

```
strict=True, missing_keys=[], unexpected_keys=[]
```

---

## 4. Single-head inference — requirement and evidence

`tools/infer.py` calls

```python
from models.hrecsnet import load_hrecsnet
...
sal = torch.sigmoid(model(batch)['sal'])     # model is SalOnly(StructureGenerator)
```

with `SalOnly.forward` defined as

```python
class SalOnly(nn.Module):
    def forward(self, x):
        return self.generator(x, only_sal=True)
```

and the CHFR branch of `StructureGenerator.forward` returning **before** the
auxiliary heads exist in the control flow:

```python
q3 = base.fuse3(torch.cat([f3, up(f4, f3)], 1)); d3 = self.attn3(q3, f4, gradient)
q2 = base.fuse2(torch.cat([f2, up(d3, f2)], 1)); d2 = self.attn2(q2, d3, gradient)
q1 = base.fuse1(torch.cat([f1, up(d2, f1)], 1)); feat = base.refine(q1)
if only_sal:
    # Pure saliency inference: the auxiliary heads are not evaluated.
    return {'sal': up(base.sal(feat), x)}
```

The five auxiliary heads are therefore never computed and then sliced — the branch
simply returns first.

### Verification method (reproducible)

`models.hrecsnet.check_single_head` wraps all six head modules in a call-counting
proxy, runs a single `forward(x, only_sal=True)` on a 384x384 input, and returns the
counters. `load_hrecsnet(..., verify_single_head=True)` (the default) runs it on the
freshly loaded weights and raises if the counts are not exactly the expected
pattern. It also asserts that the returned dictionary contains only `sal`. The five
training-only heads are then detached (`setattr(base, head, None)`), so they no
longer exist in the module tree for the actual inference loop.

Observed for all three checkpoints (printed by every `tools/infer.py` run):

```
"inference_path": "StructureGenerator.forward(x, only_sal=True)",
"head_calls": {"sal": 1, "edge": 0, "coarse": 0, "aux2": 0, "aux3": 0, "unc": 0},
"detached_heads": ["edge", "coarse", "aux2", "aux3", "unc"]
```

To reproduce: run `tools/infer.py` with `--limit 1` on any config and inspect the
first JSON line; or from Python:

```python
from models.hrecsnet import load_hrecsnet
model, ck, report = load_hrecsnet('weights/EORSSD.pth')   # asserts internally
print(report['head_calls'])
```

---

## 5. Inference commands

`--data-root` is the directory containing `dataset/`. All three runs write to the
release tree only.

```powershell
$py   = "C:\Users\lsq17\anaconda3\python.exe"
$repo = "C:\Users\lsq17\Documents\sci\HReCSNet"
$root = "C:\Users\lsq17\Documents\sci"
$out  = "C:\Users\lsq17\Documents\sci\HReCSNet_release"
Set-Location $repo

& $py "$repo\tools\infer.py" --config "$repo\configs\orssd.yaml"     --data-root $root --weights "$out\weights\ORSSD.pth"     --out-root $out
& $py "$repo\tools\infer.py" --config "$repo\configs\eorssd.yaml"    --data-root $root --weights "$out\weights\EORSSD.pth"    --out-root $out
& $py "$repo\tools\infer.py" --config "$repo\configs\orsi4199.yaml"  --data-root $root --weights "$out\weights\ORSI-4199.pth" --out-root $out
```

| Dataset | Checkpoint | Input | Output | Images | Runtime |
|---|---|---|---|---|---|
| ORSSD | `weights/ORSSD.pth` | 384×384 | `<out>/predictions/ORSSD/`, `<out>/gt/ORSSD/` | 200 | 6.2 s |
| EORSSD | `weights/EORSSD.pth` | 384×384 | `<out>/predictions/EORSSD/`, `<out>/gt/EORSSD/` | 600 | 17.9 s |
| ORSI-4199 | `weights/ORSI-4199.pth` | 384×384 | `<out>/predictions/ORSI-4199/`, `<out>/gt/ORSI-4199/` | 2199 | 67.7 s |

`<out>` = `C:\Users\lsq17\Documents\sci\HReCSNet_release`.
Predictions are 8-bit PNG probability maps at the **original** image resolution,
named after the test image; the binarized GT (`value > 127`) is exported next to them.

---

## 6. Evaluation commands

```powershell
& $py "$repo\tools\evaluate.py" --pred-dir "$out\predictions\ORSSD"     --gt-dir "$out\gt\ORSSD"     --dataset ORSSD     --output-json "$repo\outputs\metrics_ORSSD.json"
& $py "$repo\tools\evaluate.py" --pred-dir "$out\predictions\EORSSD"    --gt-dir "$out\gt\EORSSD"    --dataset EORSSD    --output-json "$repo\outputs\metrics_EORSSD.json"
& $py "$repo\tools\evaluate.py" --pred-dir "$out\predictions\ORSI-4199" --gt-dir "$out\gt\ORSI-4199" --dataset ORSI-4199 --output-json "$repo\outputs\metrics_ORSI-4199.json"
```

---

## 7. Results

Values are the raw doubles produced by `tools/evaluate.py` (the JSON files under
`outputs/`); the paper column is the four-decimal value printed in the paper.

| Dataset | Metric | Reproduced (6 dp) | Paper (4 dp) | Abs. deviation | Match (≤1e-4) |
|---|---|---|---|---|---|
| ORSSD | $S_\alpha$ | 0.956256 | 0.9563 | 4.35e-05 | yes |
| ORSSD | MAE | 0.005990 | 0.0060 | 1.04e-05 | yes |
| ORSSD | $F_\beta^{\max}$ | 0.929551 | 0.9296 | 4.89e-05 | yes |
| ORSSD | $E_\xi^{\max}$ | 0.989069 | 0.9891 | 3.12e-05 | yes |
| EORSSD | $S_\alpha$ | 0.947292 | 0.9473 | 7.80e-06 | yes |
| EORSSD | MAE | 0.004336 | 0.0043 | 3.55e-05 | yes |
| EORSSD | $F_\beta^{\max}$ | 0.901629 | 0.9016 | 2.91e-05 | yes |
| EORSSD | $E_\xi^{\max}$ | 0.983165 | 0.9832 | 3.52e-05 | yes |
| ORSI-4199 | $S_\alpha$ | 0.885244 | 0.8852 | 4.36e-05 | yes |
| ORSI-4199 | MAE | 0.026215 | 0.0262 | 1.54e-05 | yes |
| ORSI-4199 | $F_\beta^{\max}$ | 0.884964 | 0.8850 | 3.64e-05 | yes |
| ORSI-4199 | $E_\xi^{\max}$ | 0.954029 | 0.9540 | 2.85e-05 | yes |

Largest deviation over all 12 numbers: **4.89e-05**, i.e. every paper value is
recovered exactly at the published precision.

Secondary metrics (same protocol, for completeness):

| Dataset | meanF | adpF | meanE | adpE | wF | score |
|---|---|---|---|---|---|---|
| ORSSD | 0.913198 | 0.891705 | 0.983185 | 0.961841 | 0.915642 | 3.784529 |
| EORSSD | 0.876911 | 0.828111 | 0.974722 | 0.933733 | 0.882299 | 3.710049 |
| ORSI-4199 | 0.876852 | 0.867004 | 0.947133 | 0.944236 | 0.864317 | 3.562337 |

---

## 8. Cross-checks

### 8.1 Predictions are byte-identical to the training-time predictions

The research tree kept the per-epoch probability maps produced by the original
evaluation loop. Comparing them file-by-file with the maps generated here:

| Dataset | Reference directory | Files | Byte-identical | Differing |
|---|---|---|---|---|
| ORSSD | ORSSD run `predictions/epoch_030` | 200 | 200 | 0 |
| EORSSD | EORSSD run `predictions/epoch_035` | 600 | 600 | 0 |
| ORSI-4199 | ORSI-4199 run `predictions/epoch_012` | 2199 | 2199 | 0 |

So the released single-head path reproduces the original evaluation-time forward
pass bit for bit, not merely to metric precision.

### 8.2 Evaluation caliber re-derived from the original predictions

`tools/evaluate.py` was additionally run on the *original* ORSI-4199 `epoch_012`
prediction directory (the maps the training run wrote) against the raw dataset GT,
and compared with the metrics recorded in that run's `metrics_8.csv`:

| Metric | `evaluate.py` | recorded in the training run | equal |
|---|---|---|---|
| Smeasure | 0.8852436086758363 | 0.8852436086758363 | yes |
| MAE | 0.026215376410136854 | 0.026215376410136854 | yes |
| maxF | 0.8849635608724392 | 0.8849635608724392 | yes |
| maxE | 0.9540285362413209 | 0.9540285362413209 | yes |
| wF | 0.8643170834268877 | 0.8643170834268877 | yes |

All ten reported values agree to full double precision, which validates the wrapping
of `py_sod_metrics` (per-image mean for $S_\alpha$/MAE, maximum of the cross-image
averaged curve for $F_\beta^{\max}$/$E_\xi^{\max}$).

### 8.3 GT binarization threshold (finding worth recording)

The three test sets were re-evaluated twice: against the GT exported by
`tools/infer.py` (already binary, `> 127`) and against the **raw** dataset GT. Both
give identical numbers with the released script.

While setting this up, a genuine caliber pitfall was found and fixed: a handful of
ORSI-4199 masks are *not* strictly 0/255 (e.g. `2004.png` contains `247`;
`2193.png`/`2194.png` contain long ramps of values 1…126). Judging those masks with
the `gt > 0.5` binarization that `py_sod_metrics`-style wrappers often apply — instead
of the paper's `gt > 127` — turned the low-valued ramps into foreground and reported
`Smeasure = 0.8452`, `MAE = 0.0730` for the very same predictions. `tools/evaluate.py`
therefore binarizes GT at `> 127` before handing it to the metric wrapper, exactly as
the research evaluation loop does. This is a correctness fix of the *reporting*
path; no model mathematics, weight usage or metric definition was changed, and the
protocol is still per-image mean plus maximum-of-averaged-curve.

---

## 9. Package self-checks

* **No excluded baseline trace.** A case-insensitive scan of the entire
  `HReCSNet/` tree — `README.md`, every docstring, comment, string literal and file
  name included — for the four-letter abbreviation of the excluded channel-spatial
  attention baseline finds **0** occurrences. The same scan finds 0 occurrences of
  any of the study's other decoder-attention comparison variants. Only the CHFR line
  is shipped: `up` / `highpass` / `cgn` / `ResidualAdapter` / `CHFR`, the decoder
  (`adapt` / `fuse1-3` / `refine` / six heads), `GroupGenerator` /
  `StructureGenerator` / `SalOnly`, and the RBE-CS objective.
* **No training code.** `losses/rbe_cs.py` is present and explicitly labelled
  *training only*; it is imported nowhere in the inference or evaluation path.

Reproduction of the scan (the token is the four-letter abbreviation of the excluded
baseline, deliberately not written out here):

```powershell
PS> $token = '<excluded baseline abbreviation>'
PS> Get-ChildItem HReCSNet -Recurse -File -Force |
      Where-Object { $_.FullName -notlike '*\external\*' -and $_.FullName -notlike '*\outputs\*' } |
      ForEach-Object {
        $text = [System.IO.File]::ReadAllText($_.FullName)
        if ($text.ToLower().Contains($token.ToLower())) { Write-Host "HIT: $($_.FullName)" }
      }
# (no output: 0 hits over the whole tree, .git objects included)
```

---

## 10. Repository state at the end of the acceptance run

All work was done in the working tree. The new package files
(`models/`, `losses/`, `tools/`, `configs/`, `requirements.txt`, `docs/`) and the
updated `README.md` were left as ordinary working-tree changes; the release
procedures below were never invoked from this session:

* `git add` — not run
* `git commit` — not run
* `git push` — not run

Note for the record: while the acceptance run was in progress, a **separate
process** (commit author `1006730890-hue`) created commit `7438e4f` containing the
package files as they were at that moment. This session did not create it and did
not amend or revert it. Consequently `git log` now shows two commits
(`a5f71fa`, `7438e4f`), and `README.md` plus the new `docs/` directory remain
uncommitted on top of it.

---

## 11. Known limitations

* Only the released `chfr` variant is shipped; the study's other decoder adapters
  are not part of this package.
* Training, dataset preparation and the pretrained SMT-T ImageNet weights are out
  of scope. Inference needs the official SMT **source** (`models/smt.py`) but not
  its ImageNet checkpoint.
* `tools/evaluate.py` reports `maxF`/`maxE`/`wF` as produced by `py_sod_metrics`
  with its default settings; only the four paper columns are asserted above.
