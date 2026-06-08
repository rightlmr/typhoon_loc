# Claude Handoff: TC Locator Project State

Last updated: 2026-06-08 Asia/Shanghai

This document summarizes the work completed after the initial code generation of the typhoon localization project, the data/environment adaptations, the training/evaluation history, the bugs found, and the current usable state.

## 1. Project Scope And Source Of Truth

- Project root: `F:\typhoon_loc`
- Authoritative spec: `F:\typhoon_loc\TC_LOCATOR_BUILD_SPEC .md`
  - Note the actual filename contains a space before `.md`.
- The project was implemented from scratch according to the spec.
- Core design decisions from spec D1-D6 must still be respected:
  - D1: full-field CenterNet-style heatmap U-Net, not tiled detection.
  - D2: ERA5 pretrain, then AIFS short-lead fine-tune with frozen encoder.
  - D3: labels use field-consistent centers, not long-lead raw best-track centers.
  - D4: per-domain normalization.
  - D5: ERA5/AIFS must use the same channel definitions and same `calc_vo850` derivation where possible.
  - D6: full-field input, padded only for U-Net shape compatibility.

## 2. Repository Implementation Completed

The generated code covers the full spec structure:

- Core package: `tclocator/`
  - `common.py`: grid/domain helpers, coordinate conversion, haversine, config loading, device/seed helpers.
  - `io_era5.py`: ERA5 NetCDF reading and configured channel stacking.
  - `io_aifs.py`: AIFS GRIB/PT reading, filename parsing, cropping, `vo_850` derivation.
  - `vorticity.py`: shared `calc_vo850`.
  - `normalization.py`: per-domain streaming stats and normalization.
  - `labels.py`: IBTrACS lookup, field-center search, heatmap/offset/mask generation.
  - `dataset.py`: ERA5/AIFS datasets plus synthetic smoke dataset.
  - `model.py`: full-field U-Net with heatmap and offset heads.
  - `losses.py`: CenterNet focal heatmap loss plus masked offset L1.
  - `decode.py`: 3x3 NMS, thresholding, offset decode, lat filtering.
  - `tracking.py`: simple greedy track linking.
  - `metrics.py`: evaluation metrics and PR calculation.
- Scripts:
  - `scripts/phase0_consistency_and_displacement.py`
  - `scripts/compute_norm_stats.py`
  - `scripts/build_label_cache.py`
  - `scripts/pretrain.py`
  - `scripts/finetune.py`
  - `scripts/predict.py`
  - `scripts/evaluate.py`
  - `scripts/diagnose_aifs_transfer.py` was added later for transfer debugging.
- Configs:
  - `configs/pretrain.yaml`
  - `configs/finetune.yaml`
  - `configs/infer.yaml`
- Tests:
  - `tests/test_decode.py`
  - `tests/test_overfit.py`
  - `tests/test_smoke_scripts.py`
  - `tests/test_metrics.py` was added later for the lead-aware matching fix.

## 3. Data Currently Present

Data root: `F:\typhoon_loc\data`

- ERA5:
  - `data/era5`: 3748 `.nc` files.
  - Each inspected file has shape roughly `[time=1, lat=281, lon=881]`.
  - Variables present: `msl`, `sst`, `fg10`, `i10fg`, `vo_850`, `t_300`, `t_500`.
  - There is no ERA5 `u850/v850`, so `era5.vo850_from_uv` is set to `false`.
  - D5 risk remains: ERA5 `vo_850` is precomputed and cannot be automatically verified against `calc_vo850` from ERA5 wind.
- AIFS:
  - `data/aifs`: 1230 `.pt` files.
  - Each is a tensor shaped `[16, 721, 1440]`.
  - Filename pattern: `AIFS_YYYY_MM_DD_HH_FCST_XXXh.pt`.
  - Covered period inspected: 2024-09 init cycle data, leads 0-240h by 6h.
  - Configured/inferred tensor channel order:
    - `u10`, `v10`, `msl`, `t2`, `u850`, `v850`, `q850`, `t850`, `u700`, `v700`, `q700`, `t700`, `u500`, `v500`, `q500`, `t500`
  - This order was inferred from physical ranges and appears consistent:
    - channel 2 looks like MSL,
    - channels 4/5 look like 850 hPa wind,
    - channel 15 looks like 500 hPa temperature.
  - If the actual producer order differs, update `aifs.tensor_channel_order`.
- IBTrACS/georef:
  - `data/ibtracs/georef.csv`
  - 9313 rows.
  - Time range: 2020-04-25 06:00:00 to 2024-12-26 06:00:00.
- Label caches:
  - `data/label_cache/era5`: 3748 `.npz`.
  - `data/label_cache/aifs`: 1230 `.npz`.

## 4. Environment State

Training/inference Python:

```powershell
D:\study\envs\tc_loc\python.exe
```

Verified PyTorch/CUDA:

```text
torch 2.6.0+cu126
torch.version.cuda 12.6
torch.cuda.is_available() True
GPU NVIDIA GeForce RTX 4070 Ti SUPER
```

Known non-fatal warnings:

- `pynvml` FutureWarning from torch CUDA import.
- `pyproj unable to set PROJ database path` warning is suppressed around `pygrib`.
- `tc_loc` does not currently have `pytest`; tests were run with system Python, while `tc_loc` was checked with `compileall`.

## 5. Phase 0 Conclusions

Phase 0 outputs are under:

```text
outputs/phase0
```

Config conclusions filled in:

```yaml
labels.mode: "in_field"
labels.search_radius_km: 500
finetune.lead_max: 24
```

Important caveat:

- The Phase 0 `vo_850` consistency check cannot fully validate D5 because the provided ERA5 data does not include `u850/v850`.
- AIFS `vo_850` is derived from `u850/v850` using `calc_vo850`.
- ERA5 uses provided `vo_850`.

## 6. Major Fixes And Adaptations After Initial Code Generation

### 6.1 Adapted To Provided Data

The original spec expected AIFS GRIB2. The provided AIFS data is `.pt`, so `tclocator/io_aifs.py` was extended to read `.pt` tensors while preserving the GRIB reader path.

The configs were updated to the actual data paths:

```yaml
paths.era5_dir: "F:/typhoon_loc/data/era5"
paths.aifs_dir: "F:/typhoon_loc/data/aifs"
paths.ibtracs_csv: "F:/typhoon_loc/data/ibtracs/georef.csv"
```

ERA5 config was adapted:

```yaml
era5:
  var_map: {msl: "msl", t_500: "t_500", vo_850: "vo_850"}
  vo850_from_uv: false
```

### 6.2 Normalization Was Made Streaming

Normalization stat calculation was changed to avoid loading all fields into memory.

Current stats:

```text
outputs/norm_stats_era5.json
outputs/norm_stats_aifs.json
```

### 6.3 Label Cache Attachment Was Fixed

Real ERA5/AIFS samples now attach their conventional `.npz` label cache paths. Without this, fine-tuning could silently see all-zero labels when `ibtracs_records` is not passed into the dataset.

### 6.4 Pretrain Validation MAE Was Fixed

Original validation MAE assumed one center per field and could compare a correct detection against the wrong storm in multi-TC fields. `scripts/pretrain.py` was fixed to greedily match decoded peaks to all positive label centers inside the decode latitude range.

After this fix, the apparent 4000 km validation MAE problem was identified as a validation metric bug, not necessarily failed pretraining.

### 6.5 Evaluation Matching Was Fixed

Important latest fix: `tclocator/metrics.py` now matches predictions and references by `ISO_TIME + LEAD_HOUR` when both columns exist.

Reason:

- AIFS files can share the same valid time across different initialization times and forecast leads.
- Matching only by `ISO_TIME` lets predictions from one forecast field satisfy references from another forecast field.

Added regression test:

```text
tests/test_metrics.py
```

## 7. Training And Evaluation Timeline

### 7.1 M2 ERA5 Pretraining

Current checkpoint:

```text
outputs/pretrain_best.ckpt
```

Pretraining finished successfully. After the validation metric fix, M2 was considered usable enough for M3 baseline inference.

### 7.2 M3 Pretrain Direct AIFS Baseline

Original M3 baseline outputs were archived:

```text
outputs/m3_pretrain_baseline
```

After the lead-aware evaluation fix, recomputed M3 metrics were written to:

```text
outputs/diagnostics/m3_metrics_by_lead_fixed.csv
outputs/diagnostics/m3_precision_recall_fixed.csv
```

Fixed M3 summary:

```text
lead 000-024 recall=0.0367 loc_median=2171.7 km
lead 024-048 recall=0.0339 loc_median=3109.6 km
lead 048-096 recall=0.0180 loc_median=2893.5 km
lead 096-120 recall=0.0191 loc_median=3253.9 km
```

M3 was poor on AIFS, mostly because the pretrained ERA5 model fired strongly on non-target AIFS lows/vortices and weakly at the field-consistent TC labels.

### 7.3 First M4 Fine-Tune Failed

The first M4 result was bad and has been archived:

```text
outputs/m4_bad_baseline
```

Old bad M4 behavior:

- Predictions collapsed from M3's 3614 detections to 552 detections at `conf_thresh=0.3`.
- Confidence distribution collapsed.
- Recall was near zero.
- It was worse than M3 under the fixed evaluation.

Old bad M4 fixed-eval summary:

```text
lead 000-024 recall=0.0098 loc_median=4005.3 km
lead 024-048 recall=0.0000 loc_median=6029.9 km
lead 048-096 recall=0.0012 loc_median=6126.8 km
lead 096-120 recall=0.0072 loc_median=2535.8 km
```

### 7.4 Transfer Debugging

Added diagnostic script:

```powershell
D:\study\envs\tc_loc\python.exe scripts\diagnose_aifs_transfer.py --config configs\finetune.yaml
```

This script:

- rebuilds AIFS references,
- recomputes M3/M4 metrics with lead-aware matching,
- runs a non-destructive tiny AIFS overfit test.

Tiny overfit result:

```text
step=0   label_mean=0.00836 top_median_km=15147.8 top<100=0/5
step=25  label_mean=0.05602 top_median_km=9635.6  top<100=0/5
step=50  label_mean=0.06777 top_median_km=1801.4  top<100=1/5
step=100 label_mean=0.16973 top_median_km=134.8   top<100=2/5
step=200 label_mean=0.52724 top_median_km=26.5    top<100=5/5
```

Conclusion:

- Data/label/model/loss pipeline is learnable.
- The failure was not a completely broken channel or label chain.
- The old M4 hyperparameters were too weak for the AIFS shift.

### 7.5 Current Fixed M4

The fixed candidate used:

```yaml
train:
  batch_size: 1
  epochs: 30
  lr: 0.0001
  weight_decay: 0.0
  patience: 8
finetune:
  freeze_encoder: true
  lead_max: 24
```

This preserves D2 because the encoder remains frozen and only short leads are used.

The candidate checkpoint was promoted to:

```text
outputs/finetune_best.ckpt
```

The old failed checkpoint was copied to:

```text
outputs/m4_bad_baseline/finetune_best.ckpt
```

## 8. Current Official Outputs

Current official inference/evaluation was regenerated after promoting the fixed M4 checkpoint.

Prediction file:

```text
outputs/predictions.csv
```

Prediction count at current `infer.yaml` threshold:

```text
36397 detections
conf_min=0.1000
conf_median=0.1233
conf_max=0.8936
```

Current `outputs/metrics_by_lead.csv`:

```text
lead 000-024 n_ref=409 recall=0.4499 loc_median=76.5 km  end2end_median=499.1 km
lead 024-048 n_ref=413 recall=0.3826 loc_median=87.1 km  end2end_median=501.8 km
lead 048-096 n_ref=832 recall=0.2933 loc_median=170.9 km end2end_median=536.3 km
lead 096-120 n_ref=418 recall=0.1842 loc_median=281.0 km end2end_median=610.9 km
```

Current `outputs/precision_recall.csv`:

```text
conf=0.1 precision=0.0259 recall=0.2176
conf=0.2 precision=0.0728 recall=0.0853
conf=0.3 precision=0.0897 recall=0.0328
conf=0.5 precision=0.1592 recall=0.0074
conf=0.7 precision=0.5294 recall=0.0021
```

Interpretation:

- `conf_thresh=0.1` exposes many near-center peaks and gives much better localization/recall, but false alarms are high.
- `conf_thresh=0.2` is a more conservative operating point if fewer false alarms are required.
- The present `infer.yaml` defaults to `0.1` because it is better for diagnosing/retaining weak AIFS TC signals.

## 9. Current Config State

Important current settings:

```yaml
channels: ["msl", "vo_850", "t_500"]
labels:
  mode: "in_field"
  search_radius_km: 500
finetune:
  freeze_encoder: true
  lead_max: 24
train:
  lr: 0.0001
  epochs: 30
  weight_decay: 0.0
decode:
  conf_thresh: 0.1   # in configs/infer.yaml
```

`configs/finetune.yaml` still has `decode.conf_thresh=0.3` because training validation uses an internal low threshold for center MAE. Inference threshold is controlled by `configs/infer.yaml`.

## 10. Verification Already Run

System Python test run:

```powershell
python -m pytest tests
```

Result:

```text
3 passed, 1 skipped
```

The skipped test is the torch-dependent synthetic overfit test in the system Python environment. It is skipped because that environment does not have the same torch setup.

Training environment compile check:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts
```

Result: passed.

## 11. Git State At Handoff

Last committed history:

```text
2472dc8 fix(M2): match multiple centers in validation MAE
7b41aec fix(M0-M4): adapt training pipeline to provided data
6f92161 fix(M1-M4): attach label caches to real training samples
23a8122 docs(M0-M4): add Tier A configs README requirements
99dc778 feat(M0-M4): add phase0 train inference evaluation scripts
108b7ca feat(M1): add keypoint labels model losses decode tests
e182aa7 feat(M2-M4): add data IO normalization tracking metrics
05b39dd feat(M0): add shared grid utilities and vo850 derivation
e9c1103 chore(M0): initialize repository with build spec
```

Uncommitted changes at the time this handoff was written:

```text
M  configs/finetune.yaml
M  configs/infer.yaml
M  tclocator/metrics.py
?? scripts/diagnose_aifs_transfer.py
?? tests/test_metrics.py
?? CLAUDE_HANDOFF.md
```

These changes should be committed as the latest M4 debugging/fix work if the user wants the repo history to reflect the current usable state.

Suggested commit message:

```text
fix(M4): repair AIFS evaluation matching and stabilize fine-tuning
```

## 12. Recommended Next Actions

1. Commit the current handoff/fix changes.
2. Decide production operating threshold:
   - use `conf_thresh=0.1` for higher recall and diagnostic visibility,
   - use `conf_thresh=0.2` for a better precision/recall compromise,
   - avoid judging the model only at `0.3` because weak AIFS TC peaks are often below that.
3. If reducing false alarms is the next priority, improve post-processing/tracking rather than raising threshold too aggressively.
4. If more AIFS data becomes available, re-run:

```powershell
D:\study\envs\tc_loc\python.exe scripts\phase0_consistency_and_displacement.py --config configs\finetune.yaml
D:\study\envs\tc_loc\python.exe scripts\compute_norm_stats.py --config configs\finetune.yaml --domain aifs
D:\study\envs\tc_loc\python.exe scripts\build_label_cache.py --config configs\finetune.yaml --domain aifs
D:\study\envs\tc_loc\python.exe scripts\finetune.py --config configs\finetune.yaml
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --predictions outputs\predictions.csv
```

5. Keep tracking the D5 risk:
   - Best fix is obtaining ERA5 `u850/v850` so ERA5 `vo_850` can also be derived with `calc_vo850`.
   - Until then, transfer performance depends on the provided ERA5 `vo_850` being close enough in definition/scale to AIFS-derived `vo_850`.

## 13. Files And Outputs Claude Should Trust

Trust current official model/results:

```text
outputs/pretrain_best.ckpt
outputs/finetune_best.ckpt
outputs/predictions.csv
outputs/matched_metrics.csv
outputs/metrics_by_lead.csv
outputs/precision_recall.csv
```

Trust diagnostic archives:

```text
outputs/m3_pretrain_baseline
outputs/m3_pretrain_lowconf
outputs/m4_bad_baseline
outputs/m4_candidate
outputs/m4_candidate_lowconf
outputs/diagnostics
```

Do not use `outputs/m4_bad_baseline` as the current model. It exists only to preserve the failed first M4 run.
