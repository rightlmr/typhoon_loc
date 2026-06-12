# Claude Handoff: TC Locator Project State

Last updated: 2026-06-08 Asia/Shanghai

Latest update after Claude `STEP2_IBTRACS_LABEL.md`: Step 2 changed only the short-lead AIFS target/evaluation reference from `in_field` to `ibtracs`, added storm signal diagnostics, rebuilt AIFS labels, and reran fine-tuning/evaluation. The result did not pass: val `end2end@000-024 = 5620.43 km`, train `end2end@000-024 = 2259.09 km`, and `storm_signal.csv` shows weak/offset MSL and vo_850 extrema near truth. Because `.gitignore` intentionally excludes `outputs/`, `*.ckpt`, and `*.npz`, the local real-data artifacts are summarized in Sections 14-16.

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

## 14. Step 0 Field-Center Repair Results

This section records the real-data results from the follow-up plan in `D:\downloads\FIX_FIELD_CENTER.md`. These results are local artifacts under `F:\typhoon_loc\outputs`, but the CSV/checkpoint/cache files are intentionally ignored by Git and are therefore not visible to Claude through GitHub.

### 14.1 What Changed

The root issue identified by Claude was that `find_field_min_center` used a large-disk global MSL argmin around the IBTrACS position. In multi-low scenes this snapped the reference/label center to unrelated deeper lows, which inflated `track_bias_median_km` to roughly 490 km and made the previous M4 metrics physically misleading.

Implemented fix:

- `tclocator/labels.py::find_field_min_center` now uses bounded local 8-neighbor pressure-basin descent from the nearest IBTrACS grid point.
- The old global argmin path is not retained in production code.
- Optional NumPy-only box smoothing was added as `labels.field_center_smooth_px`, currently set to `0`.
- `scripts/phase0_consistency_and_displacement.py`, `scripts/evaluate.py`, and `scripts/diagnose_aifs_transfer.py` now use the same new field-center function and label config.
- Added `scripts/check_field_center.py` to compare old global argmin against new local descent.
- Added `tests/test_labels.py` for synthetic two-low regression tests.

Config change:

```yaml
labels:
  mode: "in_field"
  search_radius_km: 100
  sigma_px: 3
  field_center_smooth_px: 0
```

Claude's note suggested trying `250 km`, but real-data scanning showed that `250 km` still allowed descent to drift along large-scale pressure slopes:

```text
radius=50:  median=34.66 km p90=45.81 km lt100=200/200
radius=75:  median=56.72 km p90=71.48 km lt100=200/200
radius=100: median=77.74 km p90=94.41 km lt100=200/200
radius=125: median=107.81 km p90=121.22 km lt100=64/200
radius=150: median=128.07 km p90=146.01 km lt100=37/200
radius=200: median=179.34 km p90=195.92 km lt100=37/200
radius=250: median=224.25 km p90=245.45 km lt100=37/200
```

Therefore `100 km` was selected as the safe descent cap for this dataset.

### 14.2 `check_field_center.py` Gate

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\check_field_center.py --config configs\finetune.yaml --max-samples 10000
```

Result on all available short-lead AIFS/IBTrACS samples (`n=512`):

```text
old_global_argmin: median_dist_to_truth=492.75 km (n=512)
new_local_descent: median_dist_to_truth=78.60 km (n=512)
PASS
```

This passes the Step 0 gate:

- `new_median < 100 km`
- `new_median < old_median * 0.5`

### 14.3 Phase 0 Recomputed Displacement

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\phase0_consistency_and_displacement.py --config configs\finetune.yaml
```

Outputs written locally:

```text
outputs/phase0/displacement_vs_lead.csv
outputs/phase0/displacement_summary_by_lead.csv
outputs/phase0/displacement_vs_lead.png
```

Recomputed `outputs/phase0/displacement_summary_by_lead.csv`:

```text
lead_bin,n,mean_km,median_km,p75_km,p90_km
000-024,409,75.39163097002496,78.61268551847417,86.67838325911845,94.36343390957076
024-048,413,75.89362606323378,79.14641663836808,86.67838325911845,94.31239706175941
048-096,832,76.52016858988539,79.41544518901765,87.38986417696383,95.14730273388481
096-120,418,75.98683148293267,79.4105538422786,87.56790249940752,94.8550376157602
```

The script printed:

```text
建议 labels.mode = in_field
建议 finetune.lead_max = 24
建议 labels.search_radius_km = 100
```

D5 warning remains unchanged:

```text
ERA5 只有预计算 vo_850，缺少 u850/v850，无法自动验证 D5 口径一致性
```

### 14.4 Label Caches Rebuilt

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\build_label_cache.py --config configs\finetune.yaml --domain all
```

Local rebuilt cache counts:

```text
data/label_cache/era5: 3748 .npz
data/label_cache/aifs: 1230 .npz
```

These `.npz` files remain ignored by Git.

### 14.5 Current Checkpoint Re-Evaluated Without Retraining

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --predictions outputs\predictions.csv
```

Recomputed `outputs/metrics_by_lead.csv` using the repaired reference centers:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,409,0.029339853300733496,405.17500080284145,78.61268551847417,462.38708523812954
024-048,413,0.04116222760290557,405.2196307884753,79.14641663836808,460.9396159466754
048-096,832,0.030048076923076924,451.6144349544209,79.41544518901765,492.8076030704881
096-120,418,0.02631578947368421,504.80607607577906,79.4105538422786,550.3660606662456
```

Recomputed `outputs/precision_recall.csv`:

```text
conf_thresh,precision,recall
0.1,0.002912327939115861,0.024508670520231216
0.2,0.004341819617130452,0.005086705202312139
0.3,0.00505369551484523,0.0018497109826589595
0.5,0.009950248756218905,0.00046242774566473987
0.7,0.058823529411764705,0.00023121387283236994
```

Interpretation:

- Step 0 succeeded for the reference/label definition: `track_bias_median_km` fell from about 490 km to about 79 km.
- The current checkpoint should not be judged as final after this repair because it was trained on the old, wrong field centers.
- The lower recall/precision after re-evaluation is expected and is evidence that the previous M4 checkpoint had learned to follow the old mis-snapped labels.
- The next valid project step is to retrain/fine-tune using the rebuilt label caches, after implementing or confirming the no-leakage split strategy requested in Claude's plan.

### 14.6 Validation Commands

Commands run after the repair:

```powershell
D:\study\envs\tc_loc\python.exe scripts\check_field_center.py --config configs\finetune.yaml --max-samples 10000
$env:TMP='F:\typhoon_loc\.pytest_tmp'; $env:TEMP='F:\typhoon_loc\.pytest_tmp'; python -m pytest tests
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts
```

Results:

```text
check_field_center: PASS
pytest: 5 passed, 1 skipped
compileall: passed
```

### 14.7 Git Commit

The Step 0 repair was committed and pushed:

```text
998cc55 fix(labels): replace global argmin field center
```

This commit is on `origin/main` at `https://github.com/rightlmr/typhoon_loc.git`.

## 15. Step 1 Leakage-Free Split And Retrain Results

This section records the real-data results from `D:\downloads\STEP1_SPLIT_RETRAIN.md`. Local outputs are under `F:\typhoon_loc\outputs`, but checkpoints/CSV/NPZ outputs are ignored by Git.

### 15.1 Code Changes

Implemented within the scope of the Step 1 plan:

- Added `tclocator/split.py` with `grouped_split`, `sample_group_id`, `split_config`, and `select_aifs_files`.
- Replaced real-data `random_split` in `scripts/pretrain.py` and `scripts/finetune.py`.
- AIFS fine-tune now groups by init cycle and writes `outputs/split_aifs.json`.
- `scripts/predict.py` and `scripts/evaluate.py` now accept `--split {all,train,val}`.
- Split-specific outputs use suffixes such as `predictions_val.csv`, `metrics_by_lead_val.csv`, and `precision_recall_val.csv`.
- `tclocator/labels.py::find_field_min_center` gained optional `return_stop_reason`; default return type and descent behavior are unchanged.
- `scripts/check_field_center.py` now reports `local_min`/`cap` stop counts and median distances by stop reason.
- Added `tests/test_split.py` for group-aware split behavior.
- Added config split sections:

```yaml
# configs/pretrain.yaml
split: {group_by: "year_month", val_fraction: 0.2, seed: 42}

# configs/finetune.yaml / configs/infer.yaml
split: {group_by: "init_time", val_fraction: 0.2, seed: 42}
```

No model, loss, decode, local descent algorithm, 100 km cap, or D1-D6 design decisions were changed.

### 15.2 Validation Before Retraining

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\check_field_center.py --config configs\finetune.yaml --max-samples 10000
```

Result:

```text
old_global_argmin: median_dist_to_truth=492.75 km (n=512)
new_local_descent: median_dist_to_truth=78.60 km (n=512)
stop_reason: local_min=95 cap=417 cap_fraction=0.814
median_dist local_min=50.00 km cap=80.55 km
PASS
```

Interpretation:

- The Step 0 field-center repair still passes.
- `cap_fraction=0.814` is well above the Step 1 warning threshold of roughly 0.40.
- Most short-lead AIFS cases do not naturally descend into a local MSL minimum near the storm; the center is being limited by the 100 km cap.
- This strongly suggests msl-only field centers are too noisy for a large fraction of AIFS samples.

### 15.3 Backup Before Retraining

Before overwriting the current M4 outputs, these files were backed up to:

```text
outputs/m4_step0_labels_baseline/
```

Backed up files:

```text
finetune_best.ckpt
predictions.csv
metrics_by_lead.csv
precision_recall.csv
matched_metrics.csv
```

### 15.4 ERA5 Pretraining Was Redone

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\pretrain.py --config configs\pretrain.yaml
```

Split:

```text
ERA5 split group_by=year_month train n=3198 val n=550
val_groups=['2021-02', '2021-04', '2021-06', '2022-05', '2022-07', '2022-08', '2023-01', '2023-04', '2023-12']
```

Training summary:

```text
epoch=1 train_loss=1.6508 val_center_mae_km=54.37
epoch=2 train_loss=1.0269 val_center_mae_km=61.73
epoch=3 train_loss=0.9614 val_center_mae_km=49.49
epoch=4 train_loss=0.9033 val_center_mae_km=39.32
epoch=5 train_loss=0.8709 val_center_mae_km=40.52
epoch=6 train_loss=0.8180 val_center_mae_km=42.99
epoch=7 train_loss=0.7758 val_center_mae_km=47.98
epoch=8 train_loss=0.7408 val_center_mae_km=37.15
epoch=9 train_loss=0.6982 val_center_mae_km=35.76
epoch=10 train_loss=0.6676 val_center_mae_km=43.61
epoch=11 train_loss=0.6277 val_center_mae_km=40.62
epoch=12 train_loss=0.6006 val_center_mae_km=43.29
epoch=13 train_loss=0.5608 val_center_mae_km=45.64
epoch=14 train_loss=0.5189 val_center_mae_km=46.00
epoch=15 train_loss=0.4772 val_center_mae_km=57.49
Wrote F:\typhoon_loc\outputs\pretrain_best.ckpt
```

### 15.5 AIFS Fine-Tuning Was Redone

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\finetune.py --config configs\finetune.yaml
```

Split:

```text
AIFS split group_by=init_time train n=120 val n=30
val_groups=[
  '2024-09-07T12:00:00+00:00',
  '2024-09-11T12:00:00+00:00',
  '2024-09-15T12:00:00+00:00',
  '2024-09-20T12:00:00+00:00',
  '2024-09-23T12:00:00+00:00',
  '2024-09-27T12:00:00+00:00'
]
```

Training summary:

```text
Loaded F:\typhoon_loc\outputs\pretrain_best.ckpt
epoch=1 train_loss=6.3908 val_center_mae_km=1109.58
epoch=2 train_loss=4.1824 val_center_mae_km=1118.16
epoch=3 train_loss=4.0845 val_center_mae_km=1060.12
epoch=4 train_loss=4.0344 val_center_mae_km=1053.93
epoch=5 train_loss=3.9908 val_center_mae_km=1111.98
epoch=6 train_loss=3.9466 val_center_mae_km=1115.98
epoch=7 train_loss=3.8894 val_center_mae_km=1215.18
epoch=8 train_loss=3.8190 val_center_mae_km=1734.24
epoch=9 train_loss=3.7343 val_center_mae_km=1541.69
epoch=10 train_loss=3.6353 val_center_mae_km=1382.82
epoch=11 train_loss=3.5212 val_center_mae_km=1778.44
epoch=12 train_loss=3.3868 val_center_mae_km=2139.03
Wrote F:\typhoon_loc\outputs\finetune_best.ckpt
```

`outputs/split_aifs.json`:

```json
{
  "group_by": "init_time",
  "val_fraction": 0.2,
  "seed": 42,
  "val_groups": [
    "2024-09-07T12:00:00+00:00",
    "2024-09-11T12:00:00+00:00",
    "2024-09-15T12:00:00+00:00",
    "2024-09-20T12:00:00+00:00",
    "2024-09-23T12:00:00+00:00",
    "2024-09-27T12:00:00+00:00"
  ]
}
```

### 15.6 Validation-Only Inference And Evaluation

Commands:

```powershell
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split val
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split val --predictions outputs\predictions_val.csv
```

`outputs/metrics_by_lead_val.csv`:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,94,0.0,1755.3500766750099,78.84959033456511,1815.8273747083335
024-048,95,0.042105263157894736,1631.659272180731,78.08268421981114,1586.5963826514298
048-096,170,0.058823529411764705,1481.0569102551867,79.99228037091818,1467.3963206666026
096-120,108,0.0,2599.838775788575,80.5194569437135,2560.9332803664947
```

`outputs/precision_recall_val.csv`:

```text
conf_thresh,precision,recall
0.1,0.012658227848101266,0.018008474576271187
0.2,0.05405405405405406,0.00211864406779661
0.3,0.0,0.0
0.5,0.0,0.0
0.7,0.0,0.0
```

Step 1 judgment:

- This does not pass.
- The Step 0 full-data old-model baseline was `end2end@000-024 ~= 462 km`.
- After leakage-free retraining, val `end2end@000-024 = 1815.83 km`, much worse than baseline and far from the expected 100-200 km range.
- `track_bias_median_km` remains about 79-80 km, so the repaired reference center definition is stable; the failure is in model prediction or target learnability, not the evaluation reference.

### 15.7 Extra Diagnostic: Train Split Is Also Poor

To distinguish pure validation generalization failure from training/target failure, train split inference/evaluation was also run.

Commands:

```powershell
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split train
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split train --predictions outputs\predictions_train.csv
```

`outputs/metrics_by_lead_train.csv`:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,315,0.12380952380952381,1253.0195461559304,78.61268551847417,1262.0765157221506
024-048,318,0.08490566037735849,1475.6512958546623,79.82811224426311,1453.816591727207
048-096,662,0.04229607250755287,1984.1977937624997,79.40518904019766,1967.2702155044826
096-120,310,0.025806451612903226,2266.082974826154,78.81074084346363,2226.18622456266
```

`outputs/precision_recall_train.csv`:

```text
conf_thresh,precision,recall
0.1,0.019419237749546278,0.03164744158532978
0.2,0.03508771929824561,0.0011830819284235432
0.3,0.0,0.0
0.5,0.0,0.0
0.7,0.0,0.0
```

Interpretation:

- Train split is also poor, including the 0-24h leads used for AIFS fine-tuning.
- Therefore this is not only a no-leakage validation generalization issue.
- AIFS labels are present: short-lead train/val label caches all have positive masks.
- AIFS normalization stats look reasonable and close to ERA5 stats.
- Current strongest evidence points to the msl-only field-center target being too noisy/ambiguous for AIFS: 81.4% of short-lead samples are cap-limited instead of true local-MSL-min stops.

### 15.8 Verification Notes

Commands/checks completed:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts tests
D:\study\envs\tc_loc\python.exe -c "import runpy; ns=runpy.run_path('tests/test_split.py'); ns['test_grouped_split_keeps_all_leads_of_init_together'](); ns['test_select_aifs_files_uses_same_deterministic_split'](); print('split tests ok')"
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split val --smoke-synthetic
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split val --smoke-synthetic
```

Results:

```text
compileall: passed
split tests: passed
predict/evaluate smoke: passed
```

Full `pytest` was not run in `tc_loc` because that environment currently lacks pytest even though `requirements.txt` declares it:

```text
D:\study\envs\tc_loc\python.exe: No module named pytest
```

System Python has pytest but lacks torch, so it is not a valid full test environment for this project.

### 15.9 Recommended Next Step

Do not keep tuning hyperparameters on the current msl-only label definition.

Recommended next implementation step:

1. Add the diagnostic requested in Step 1: inspect 3-5 validation storms with true center, field center, model prediction, local MSL patch, and `vo_850` patch.
2. Then implement a field-center fallback that uses msl local descent when it reaches a local minimum, but uses `vo_850` maximum near the storm when msl descent hits the cap.
3. Rebuild label cache, retrain, and re-evaluate with the same leakage-free split.

Reason:

- The pipeline has labels and normalized inputs.
- Pretraining still works on ERA5.
- AIFS fine-tuning fails on train and val.
- `cap_fraction=0.814` shows the present msl-only target is often not a real local-pressure center.

## 16. Step 2 IBTrACS Short-Lead Target Results

This section records the real-data results from `D:\downloads\STEP2_IBTRACS_LABEL.md`. The goal was to change one variable only: AIFS short-lead labels/evaluation references use IBTrACS truth instead of `in_field` field centers. Pretraining was not rerun.

### 16.1 Code And Config Changes

Changed:

- `configs/finetune.yaml`: `labels.mode` changed from `"in_field"` to `"ibtracs"`.
- `configs/infer.yaml`: `labels.mode` changed from `"in_field"` to `"ibtracs"`.
- `scripts/evaluate.py::_build_references`: when `labels.mode == "ibtracs"`, `LAT_FIELD/LON_FIELD` are set directly to `LAT_TRUE/LON_TRUE`; `find_field_min_center` is only used in the unchanged `in_field` branch.
- Added `scripts/inspect_storms.py`.

Not changed:

- `configs/pretrain.yaml` remains `labels.mode: "in_field"`.
- No model/loss/decode/descent/cap/fine-tune-protocol changes.
- No ERA5 pretraining was rerun.

Note on optional PNGs:

- `inspect_storms.py` writes the required CSV by default.
- Optional PNG output is behind `--plots`; matplotlib was installed but crashed in this environment inside `numpy.linalg` during `savefig`, so PNGs were not used.

### 16.2 Storm Signal Diagnostic

Command after Step2 retraining:

```powershell
D:\study\envs\tc_loc\python.exe scripts\inspect_storms.py --config configs\finetune.yaml --max-cases 6
```

Summary:

```text
cases=6
median msl_min_dist_km = 196.43
median vo_max_dist_km  = 183.94
median dist_pred_truth_km = 12709.98
```

`outputs/diagnostics/storm_signal.csv`:

```text
sid,valid_time,lead_hour,true_lat,true_lon,pred_lat,pred_lon,pred_conf,dist_pred_truth_km,msl_at_truth_pa,msl_min_pa,msl_min_dist_km,vo_at_truth,vo_max,vo_max_dist_km
2024244N09137,2024-09-07T12:00:00+00:00,0,21.0,106.0,20.476185735315084,268.02421379461884,0.12024269998073578,15006.205964441891,101150.8671875,101046.8671875,174.75770781093388,2.135442446160596e-05,5.02247094118502e-05,181.66555536368122
2024246N22147,2024-09-07T12:00:00+00:00,0,44.0,160.39999999999998,20.476185735315084,268.02421379461884,0.12024269998073578,9759.212424410282,101814.8671875,101614.8671875,187.96294206255263,-2.0343461073935032e-05,8.017124491743743e-05,189.08994314153068
2024244N09137,2024-09-07T18:00:00+00:00,6,21.0,105.39999999999998,21.23092552088201,285.2695224098861,0.19615057110786438,15319.200741929113,101056.484375,100936.484375,199.3489578332469,1.2647826224565506e-05,7.560780068160966e-05,186.2238256503907
2024246N22147,2024-09-07T18:00:00+00:00,6,46.4,164.79999999999995,21.23092552088201,285.2695224098861,0.19615057110786438,10413.749278654706,101408.484375,101196.484375,198.37984941011442,6.1562723203678615e-06,5.746651368099265e-05,160.1881325690075
2024244N09137,2024-09-08T00:00:00+00:00,12,21.1,104.8,21.47117779031396,285.28048124164343,0.11308617144823074,15281.100313275088,101010.2421875,100946.2421875,196.50110316661022,2.4673592633916996e-05,5.6192573538282886e-05,188.8845345210406
2024246N22147,2024-09-08T00:00:00+00:00,12,48.6,169.20000000000005,21.47117779031396,285.28048124164343,0.11308617144823074,9982.012782820431,101190.2421875,101038.2421875,196.35257160201462,5.063425487605855e-05,7.125219417503104e-05,61.06201506636
```

Interpretation:

- MSL minima inside 200 km are still nearly at the search boundary.
- vo_850 maxima are only slightly closer in the 6-case sample, except one case at 61 km.
- Model top peaks are very far from truth, typically near unrelated strong signals.

### 16.3 Backup And Label Cache Rebuild

Step 1 in-field outputs were backed up to:

```text
outputs/m4_step1_infield_baseline/
```

Backed up:

```text
finetune_best.ckpt
predictions*.csv
metrics_by_lead*.csv
precision_recall*.csv
matched_metrics*.csv
```

AIFS labels were rebuilt with `labels.mode: ibtracs`:

```powershell
D:\study\envs\tc_loc\python.exe scripts\build_label_cache.py --config configs\finetune.yaml --domain aifs
```

### 16.4 Fine-Tuning Curve

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\finetune.py --config configs\finetune.yaml
```

Training output:

```text
Loaded F:\typhoon_loc\outputs\pretrain_best.ckpt
AIFS split group_by=init_time train n=120 val n=30 val_groups=['2024-09-07T12:00:00+00:00', '2024-09-11T12:00:00+00:00', '2024-09-15T12:00:00+00:00', '2024-09-20T12:00:00+00:00', '2024-09-23T12:00:00+00:00', '2024-09-27T12:00:00+00:00']
epoch=1 train_loss=7.0795 val_center_mae_km=1148.23
epoch=2 train_loss=4.7428 val_center_mae_km=1285.28
epoch=3 train_loss=4.6309 val_center_mae_km=1240.95
epoch=4 train_loss=4.5748 val_center_mae_km=1351.14
epoch=5 train_loss=4.5233 val_center_mae_km=1233.19
epoch=6 train_loss=4.4684 val_center_mae_km=1173.64
epoch=7 train_loss=4.3895 val_center_mae_km=1222.52
epoch=8 train_loss=4.2903 val_center_mae_km=1826.52
epoch=9 train_loss=4.1776 val_center_mae_km=1821.51
Wrote F:\typhoon_loc\outputs\finetune_best.ckpt
```

### 16.5 Validation Metrics

Commands:

```powershell
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split val
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split val --predictions outputs\predictions_val.csv
```

`outputs/metrics_by_lead_val.csv`:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,94,0.0,5620.428737989904,0.0,5620.428737989904
024-048,95,0.0,3762.6404626371677,0.0,3762.6404626371677
048-096,170,0.011764705882352941,2253.505891245527,0.0,2253.505891245527
096-120,108,0.0,7462.809873202623,0.0,7462.809873202623
```

`outputs/precision_recall_val.csv`:

```text
conf_thresh,precision,recall
0.1,0.005154639175257732,0.00211864406779661
0.2,0.125,0.001059322033898305
0.3,0.0,0.0
0.5,0.0,0.0
0.7,0.0,0.0
```

### 16.6 Train Split Metrics

Commands:

```powershell
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split train
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split train --predictions outputs\predictions_train.csv
```

`outputs/metrics_by_lead_train.csv`:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,315,0.009523809523809525,2259.0928400785956,0.0,2259.0928400785956
024-048,318,0.012578616352201259,2117.0371668536154,0.0,2117.0371668536154
048-096,662,0.004531722054380665,3254.4388091227775,0.0,3254.4388091227775
096-120,310,0.00967741935483871,2822.4289771317467,0.0,2822.4289771317467
```

`outputs/precision_recall_train.csv`:

```text
conf_thresh,precision,recall
0.1,0.007859733978234583,0.003845016267376516
0.2,0.0,0.0
0.3,0.0,0.0
0.5,0.0,0.0
0.7,0.0,0.0
```

### 16.7 Decision Tree Result

Step 2 does not pass.

Decision tree category:

```text
train 0-24h still very poor: AIFS msl/vo signal is too weak under the current frozen-encoder + Tier A channel setup.
```

Evidence:

- Val `end2end@000-024 = 5620.43 km`, worse than Step 1 val and far worse than the target 100-200 km.
- Train `end2end@000-024 = 2259.09 km`, so the current setup cannot even fit the training split.
- In `ibtracs` mode, `track_bias_median_km = 0.0`, confirming evaluation correctly uses truth as the reference center.
- `storm_signal.csv` shows local MSL minima and vo_850 maxima near truth are usually close to the 200 km diagnostic boundary, not centered on truth.

Recommended next step:

- Do not keep tuning only the label mode.
- Since vo_850 is only marginally better than MSL in the 6-case diagnostic, the next most defensible independent variable is Tier B input channels, especially adding `t_850`, followed by a controlled comparison of frozen encoder vs partial unfreeze.
- If pursuing a vo target/fallback, first broaden `inspect_storms.py --max-cases` beyond 6 to verify whether `vo_max_dist_km` is consistently better across more validation cases.

### 16.8 Verification Notes

Commands/checks completed:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts tests
D:\study\envs\tc_loc\python.exe -c "import runpy; ns=runpy.run_path('tests/test_split.py'); ns['test_grouped_split_keeps_all_leads_of_init_together'](); ns['test_select_aifs_files_uses_same_deterministic_split'](); print('split tests ok')"
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split val --smoke-synthetic
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split val --smoke-synthetic
```

Results:

```text
compileall: passed
split tests: passed
predict/evaluate smoke: passed
```

Important operational note:

- The smoke `predict/evaluate` commands write `_val` files, so real val prediction/evaluation was rerun after the smoke checks to restore `outputs/predictions_val.csv`, `outputs/metrics_by_lead_val.csv`, and `outputs/precision_recall_val.csv`.

## 17. Step 3 AIFS `.pt` Orientation Fix

Step 3 followed `D:\downloads\STEP3_FIX_AIFS_ORIENTATION.md`. The only code path changed was serialized AIFS `.pt` spatial alignment in `tclocator/io_aifs.py`; channel order, normalization method, `calc_vo850`, model, losses, decode, labels, descent/cap logic, fine-tuning protocol, and `configs/pretrain.yaml` were not changed.

Important consequence:

- All AIFS results produced before this fix, including Step 1 and Step 2, should be treated as results from spatially misaligned AIFS fields and are not reliable.
- New AIFS results below are the first reliable post-alignment metrics.

### 17.1 Orientation Audit

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\audit_aifs_orientation.py --config configs\finetune.yaml
```

Output:

```text
file=F:\typhoon_loc\data\aifs\AIFS_2024_09_07_12_FCST_000h.pt
tensor_shape=(16, 721, 1440) msl_index=2
msl_raw_min_hPa=952.71 max_hPa=1043.91 mean_hPa=1007.56
array	lat_order	lon_mode	row	col	sampled_msl_hPa
raw	north_first	from_0	276	424	1011.50
raw	north_first	roll_180	276	1144	984.97
raw	north_first	from_180	276	1144	984.97
raw	south_first	from_0	444	424	1016.45
raw	south_first	roll_180	444	1144	1022.38
raw	south_first	from_180	444	1144	1022.38
transpose_probe	north_first	from_0	276	424	1015.31
transpose_probe	north_first	roll_180	276	423	1015.13
transpose_probe	north_first	from_180	276	423	1015.13
transpose_probe	south_first	from_0	444	424	1020.30
transpose_probe	south_first	roll_180	444	423	1020.81
transpose_probe	south_first	from_180	444	423	1020.81
WINNER: array=raw lat_order=north_first lon_mode=roll_180 sampled_msl_hPa=984.97
global_argmin_north_first_from_0: row=583 col=1434 lat=-55.75 lon=358.50 msl_hPa=952.71
```

Conclusion:

- The tensor is `[C, 721, 1440]`, raw array, north-first latitude.
- `.pt` longitude is effectively indexed as `[-180, -179.75, ..., 179.75]`, not `[0, 0.25, ..., 359.75]`.
- The production `.pt` crop now uses `PT_GLOBAL_LON = (col * 0.25 - 180) % 360`.
- GRIB cropping still uses the original `GLOBAL_LON = 0..359.75`.

### 17.2 Code Change

Changed `tclocator/io_aifs.py`:

```text
PT_GLOBAL_LON = np.mod(np.arange(1440, dtype=np.float64) * 0.25 - 180.0, 360.0)

def crop_aifs_pt_global(values: np.ndarray, domain: DomainConfig) -> np.ndarray:
    return crop_regular_latlon_grid(values, GLOBAL_LAT, PT_GLOBAL_LON, domain)
```

`read_aifs_channels()` now dispatches:

- `.pt` files: `read_aifs_pt_variable()` then `crop_aifs_pt_global()`.
- GRIB files: `read_aifs_variable()` then `crop_aifs_global()`.

Added scripts:

- `scripts/audit_aifs_orientation.py`
- `scripts/check_aifs_alignment.py`

### 17.3 Alignment Gate

Before the fix, the hard gate failed:

```text
2024244N09137 valid=2024-09-07T12:00:00 truth=(21.00,106.00) msl_truth_hPa=1011.51 min100_hPa=1010.99 min100=(20.25,106.00) min100_dist_km=83.40 FAIL
FAIL
```

After the fix:

```powershell
D:\study\envs\tc_loc\python.exe scripts\check_aifs_alignment.py --config configs\finetune.yaml
```

```text
2024244N09137 valid=2024-09-07T12:00:00 truth=(21.00,106.00) msl_truth_hPa=982.95 min100_hPa=981.59 min100=(21.25,106.00) min100_dist_km=27.80 PASS
PASS
```

The Step 3 hard gate passed before any downstream norm/cache/retrain work was run.

### 17.4 Downstream Rebuild

Backed up Step 2 artifacts to:

```text
outputs/m4_step2_ibtracs_baseline/
```

Backed up files included:

```text
finetune_best.ckpt
predictions.csv
predictions_train.csv
predictions_val.csv
metrics_by_lead.csv
metrics_by_lead_train.csv
metrics_by_lead_val.csv
precision_recall.csv
precision_recall_train.csv
precision_recall_val.csv
matched_metrics.csv
matched_metrics_train.csv
matched_metrics_val.csv
norm_stats_aifs.json
```

Recomputed AIFS normalization stats:

```powershell
D:\study\envs\tc_loc\python.exe scripts\compute_norm_stats.py --config configs\finetune.yaml --domain aifs
```

```text
Wrote F:\typhoon_loc\outputs\norm_stats_aifs.json
```

New `outputs/norm_stats_aifs.json`:

```json
{
  "channels": ["msl", "vo_850", "t_500"],
  "stats": {
    "msl": {"method": "zscore", "mean": 101273.23387616295, "std": 705.6261537486697, "shift": 0.0},
    "vo_850": {"method": "zscore", "mean": -5.771534547605652e-07, "std": 3.2446931898418934e-05, "shift": 0.0},
    "t_500": {"method": "zscore", "mean": 262.37134432207915, "std": 8.34764666990409, "shift": 0.0}
  }
}
```

Rebuilt AIFS label cache:

```powershell
D:\study\envs\tc_loc\python.exe scripts\build_label_cache.py --config configs\finetune.yaml --domain aifs
```

Completed successfully and rewrote AIFS `.npz` label-cache files under `outputs/label_cache_aifs`.

### 17.5 Fine-Tuning Curve

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\finetune.py --config configs\finetune.yaml
```

Output:

```text
Loaded F:\typhoon_loc\outputs\pretrain_best.ckpt
AIFS split group_by=init_time train n=120 val n=30 val_groups=['2024-09-07T12:00:00+00:00', '2024-09-11T12:00:00+00:00', '2024-09-15T12:00:00+00:00', '2024-09-20T12:00:00+00:00', '2024-09-23T12:00:00+00:00', '2024-09-27T12:00:00+00:00']
epoch=1 train_loss=1.9620 val_center_mae_km=68.33
epoch=2 train_loss=1.6037 val_center_mae_km=68.27
epoch=3 train_loss=1.4510 val_center_mae_km=68.12
epoch=4 train_loss=1.3740 val_center_mae_km=74.99
epoch=5 train_loss=1.3135 val_center_mae_km=75.25
epoch=6 train_loss=1.2525 val_center_mae_km=71.31
epoch=7 train_loss=1.1955 val_center_mae_km=73.26
epoch=8 train_loss=1.1315 val_center_mae_km=72.61
epoch=9 train_loss=1.0710 val_center_mae_km=73.11
epoch=10 train_loss=1.0142 val_center_mae_km=72.79
epoch=11 train_loss=0.9574 val_center_mae_km=83.65
Wrote F:\typhoon_loc\outputs\finetune_best.ckpt
```

This is a large correction relative to Step 2, where val center MAE was above 1100 km and train/eval end-to-end errors were in the thousands of km.

### 17.6 Validation Metrics

Commands:

```powershell
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split val
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split val --predictions outputs\predictions_val.csv
```

`outputs/metrics_by_lead_val.csv`:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,94,0.5531914893617021,52.212656399533614,0.0,52.212656399533614
024-048,95,0.5052631578947369,58.10950795713562,0.0,58.10950795713562
048-096,170,0.16470588235294117,158.54689768652085,0.0,158.54689768652085
096-120,108,0.07407407407407407,243.87023417136436,0.0,243.87023417136436
```

`outputs/precision_recall_val.csv`:

```text
conf_thresh,precision,recall
0.1,0.04312590448625181,0.15783898305084745
0.2,0.10219594594594594,0.1281779661016949
0.3,0.1396011396011396,0.1038135593220339
0.5,0.11187214611872145,0.05190677966101695
0.7,0.10029498525073746,0.036016949152542374
```

### 17.7 Train Split Metrics

Commands:

```powershell
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split train
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split train --predictions outputs\predictions_train.csv
```

`outputs/metrics_by_lead_train.csv`:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,315,0.8253968253968254,30.83350527619921,0.0,30.83350527619921
024-048,318,0.5251572327044025,58.06035898232219,0.0,58.06035898232219
048-096,662,0.1691842900302115,137.80876032987337,0.0,137.80876032987337
096-120,310,0.08387096774193549,258.88367265726566,0.0,258.88367265726566
```

`outputs/precision_recall_train.csv`:

```text
conf_thresh,precision,recall
0.1,0.04964138931420743,0.18219461697722567
0.2,0.12487969201154957,0.15350488021295475
0.3,0.16200294550810015,0.13013901212658976
0.5,0.1693548387096774,0.09937888198757763
0.7,0.15809768637532134,0.07275953859804792
```

### 17.8 Signal Diagnostic

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\inspect_storms.py --config configs\finetune.yaml --max-cases 10
```

Output summary:

```text
PNG diagnostics disabled; pass --plots to enable optional matplotlib output.
Loaded F:\typhoon_loc\outputs\finetune_best.ckpt
Wrote F:\typhoon_loc\outputs\diagnostics\storm_signal.csv
cases=10
median msl_min_dist_km = 73.28
median vo_max_dist_km  = 73.28
median dist_pred_truth_km = 2820.96
```

The `dist_pred_truth_km` median is not a direct equivalent of `metrics_by_lead`: this diagnostic compares the same per-field top peak against each listed storm at the same valid time. For valid times containing both Yagi and another storm, the top peak correctly lands on Yagi and is then also compared against the other far-away storm. Use the evaluation CSVs above for official split metrics.

`outputs/diagnostics/storm_signal.csv`:

```text
sid,valid_time,lead_hour,true_lat,true_lon,pred_lat,pred_lon,pred_conf,dist_pred_truth_km,msl_at_truth_pa,msl_min_pa,msl_min_dist_km,vo_at_truth,vo_max,vo_max_dist_km
2024244N09137,2024-09-07T12:00:00+00:00,0,21.0,106.0,21.13771465420723,106.08008122444153,0.9281761050224304,17.422351728555704,98294.8671875,98158.8671875,27.798731661139723,0.0014159309212118387,0.0015353863127529621,25.952349116302408
2024246N22147,2024-09-07T12:00:00+00:00,0,44.0,160.39999999999998,21.13771465420723,106.08008122444153,0.9281761050224304,5567.681104747304,100442.8671875,100350.8671875,55.3740671012096,0.00032278639264404774,0.0008603124879300594,73.31924582215831
2024244N09137,2024-09-07T18:00:00+00:00,6,21.0,105.39999999999998,21.163391940295696,105.34773378074169,0.9067559838294983,18.960361423297073,99604.484375,99568.484375,31.856406748650265,0.0006043613539077342,0.0006966097862459719,31.856406748650265
2024246N22147,2024-09-07T18:00:00+00:00,6,46.4,164.79999999999995,21.163391940295696,105.34773378074169,0.9067559838294983,6000.38972715748,100492.484375,100196.484375,47.06608766849182,0.00031563686206936836,0.0006572998245246708,22.672656007467722
2024244N09137,2024-09-08T00:00:00+00:00,12,21.1,104.8,21.155245564877987,105.35779732465744,0.8391200304031372,58.18014134147351,100134.2421875,99882.2421875,125.53726538148678,0.0001496221375418827,0.00034532544668763876,49.55077128920741
2024246N22147,2024-09-08T00:00:00+00:00,12,48.6,169.20000000000005,21.155245564877987,105.35779732465744,0.8391200304031372,6353.585319588048,100658.2421875,100538.2421875,100.23243455296716,0.0002930602349806577,0.00040337469545193017,153.8115734623359
2024244N09137,2024-09-08T06:00:00+00:00,18,21.4,104.39999999999998,21.425060272216797,105.11665026843548,0.7872138023376465,74.23974794654339,100466.90625,100018.90625,166.5671383297044,0.00010200658289249986,0.000296311016427353,73.23855548138329
2024246N22147,2024-09-08T06:00:00+00:00,18,51.0,173.89999999999998,21.425060272216797,105.11665026843548,0.7872138023376465,6701.898097504073,100590.90625,100546.90625,56.590111605480935,7.351593376370147e-05,0.00026872489252127707,194.72160764544128
2024244N09137,2024-09-08T12:00:00+00:00,24,21.6,104.0,21.898540169000626,103.59931302070618,0.308197557926178,53.051934483442935,100253.1875,100125.1875,185.2764395117866,1.562223224027548e-05,0.00022142416855785996,89.29717414144766
2024246N22147,2024-09-08T12:00:00+00:00,24,53.0,179.5,21.898540169000626,103.59931302070618,0.308197557926178,7148.211829437412,100753.1875,100697.1875,89.96574270763752,3.410916542634368e-05,0.00020354308071546257,137.06824224262044
```

### 17.9 Decision Tree Result

Step 3 passes.

Decision tree category:

```text
train and val both dropped into the ~30-60 km short-lead range, so the AIFS .pt spatial alignment bug was the main blocker and the method is now valid for short-lead AIFS localization.
```

Evidence:

- Alignment gate passed: Yagi lead-0 truth point now reads `982.95 hPa`, and the 100 km local minimum is `981.59 hPa` at `27.80 km`.
- Train `end2end@000-024` improved from Step 2 `2259.09 km` to `30.83 km`.
- Val `end2end@000-024` improved from Step 2 `5620.43 km` to `52.21 km`.
- Val `end2end@024-048` is also usable at `58.11 km`.
- Longer leads still have low recall (`048-096` recall `0.165`, `096-120` recall `0.074` on val), so the next work should focus on more AIFS data and long-lead robustness rather than more spatial-orientation changes.

Recommended next step:

- Treat `.pt` orientation as fixed and keep it gated by `scripts/check_aifs_alignment.py`.
- Do not revisit channel additions or target changes until the current aligned Tier A baseline is preserved.
- Expand AIFS data coverage and rerun the same aligned pipeline for a broader validation split, then decide whether long-lead recall requires Tier B channels or a detection-confidence change.

### 17.10 Verification Notes

Commands/checks completed after the Step 3 fix:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts tests
D:\study\envs\tc_loc\python.exe -c "import runpy; ns=runpy.run_path('tests/test_split.py'); ns['test_grouped_split_keeps_all_leads_of_init_together'](); ns['test_select_aifs_files_uses_same_deterministic_split'](); print('split tests ok')"
```

Results:

```text
compileall: passed
split tests: passed
```

## 23. Step 5 Truth Build And Hybrid Gate

This section records the work from `D:\downloads\STEP5_TRUTH_HYBRID_CROSSYEAR.md`.

Important correction to the prior interpretation:

- `track_bias_median_km` is the distance between the evaluation reference center and IBTrACS truth.
- It is not a tracker-association error.
- Phase C used `ibtracs` references, so track bias was forced to zero.
- Phase D used `in_field` references, so nonzero track bias is expected and measures field-center displacement from truth.
- C/D `loc_error_median_km` are not directly comparable when their reference definitions differ; `end2end_median_km` is the comparable business metric.

### 23.1 Phase T Truth Conversion

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\build_ibtracs_truth.py --config configs\finetune.yaml --input data\ibtracs\full\ibtracs.ALL.list.v04r01.csv
```

Result:

```text
PASS
```

Implementation notes:

- New script: `scripts/build_ibtracs_truth.py`.
- It writes the official-source conversion to `outputs/audit/ibtracs_converted_project_format.csv`.
- It writes the merged local truth file to `data/ibtracs/georef_2020_2025.csv`.
- The original `data/ibtracs/georef.csv` is not overwritten.
- The script imports `pygrib` through `tclocator` before importing `pandas`, avoiding the pygrib/pyproj DLL crash seen when pandas is imported first.
- The converter does not filter to `TRACK_TYPE == main`. That filter would wrongly discard 2025 western/northern-hemisphere `PROVISIONAL` rows from the current official IBTrACS file. The 2024 overlap months still contain only `main` rows and therefore remain exactly consistent with old truth.

Conversion stats (`outputs/audit/ibtracs_conversion_stats.json`):

```json
{
  "raw_rows": 722983,
  "after_track_type_filter": 722983,
  "dropped_bad_numeric_or_time": 0,
  "dropped_non_6h": 369163,
  "dropped_out_of_domain": 147932,
  "dropped_duplicate_time_sid": 0
}
```

Overlap consistency gate (`outputs/audit/truth_overlap_consistency.csv`):

```text
year_month,n_records_old,n_records_new,new_old_ratio,matched_records,position_median_km,position_max_km,match_mode,pass
2024-05,56,56,1.0,56,0.0,2.8272069309896195e-12,time_sid,True
2024-06,47,47,1.0,47,0.0,4.231478842490107e-12,time_sid,True
2024-07,171,171,1.0,171,0.0,2.7952272962114123e-12,time_sid,True
2024-08,479,479,1.0,479,0.0,5.059647494668353e-12,time_sid,True
2024-09,409,409,1.0,409,0.0,4.296401011876695e-12,time_sid,True
2024-10,280,280,1.0,280,0.0,4.205147948181035e-12,time_sid,True
2024-11,244,244,1.0,244,0.0,5.522307669202817e-12,time_sid,True
```

### 23.2 Coverage And Alignment After Truth Build

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\audit_data_coverage.py --config configs\finetune.yaml --max-cases-per-month 20
```

Result:

```text
PASS_WITH_BLOCKED_MONTHS
```

Coverage table (`outputs/audit/aifs_truth_join.csv`):

```text
year_month,n_files,n_inits,lead_min,lead_max,n_leads_per_init_median,missing_leads_examples,truth,n_records,n_sids
2024-04,1230,30,0,240,41.0,,MISSING,0,0
2024-05,1271,31,0,240,41.0,,OK,56,2
2024-06,1230,30,0,240,41.0,,OK,47,5
2024-07,1271,31,0,240,41.0,,OK,171,8
2024-08,1271,31,0,240,41.0,,OK,479,16
2024-09,1230,30,0,240,41.0,,OK,409,21
2024-10,1271,31,0,240,41.0,,OK,280,13
2024-11,1230,30,0,240,41.0,,OK,244,10
2024-12,1271,31,0,240,41.0,,OK,19,1
2025-01,1271,31,0,240,41.0,,MISSING,0,0
2025-02,1066,26,0,240,41.0,,MISSING,0,0
2025-07,1271,31,0,240,41.0,,OK,272,14
2025-08,1271,31,0,240,41.0,,OK,345,20
```

Coverage decision (`outputs/audit/coverage_decision.json`):

```json
{
  "truth_usable_months": ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-07", "2025-08"],
  "truth_blocked_months": ["2024-04", "2025-01", "2025-02"],
  "alignment_pass_months": ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2025-07", "2025-08"],
  "alignment_blocked_months": ["2024-12"],
  "blocked_months": ["2024-04", "2024-12", "2025-01", "2025-02"],
  "recommended_data_usable_months": ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2025-07", "2025-08"]
}
```

Interpretation:

- `2025-07` and `2025-08` are now unlocked by truth coverage and AIFS alignment.
- `2025-01` and `2025-02` remain blocked because the current official source has no in-domain truth rows for those AIFS months.
- `2024-12` remains blocked because the monthly alignment pressure gate fails on weak systems.
- Current `configs/finetune.yaml` and `configs/infer.yaml` point to `data/ibtracs/georef_2020_2025.csv`, but `data.usable_months` remains the original seven 2024 months because Step 5 core must keep the Phase D data/split unchanged.

### 23.3 Step 5 Hybrid Diagnostic

For the diagnostic only, `labels.center_criterion` was set to `msl_vo_hybrid` with:

```yaml
labels:
  mode: "in_field"
  center_criterion: "msl_vo_hybrid"
  search_radius_km: 100
  vo_smooth_px: 1
finetune:
  lead_max: 120
data.usable_months: ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]
split.val_groups_override: ["2024-08", "2024-10"]
```

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\phase0_consistency_and_displacement.py --config configs\finetune.yaml --lead-max 120 --diagnostic-radius-km 500 --progress-every 200
```

Outputs:

```text
outputs/phase0/displacement_vs_lead.csv
outputs/phase0/displacement_summary_by_lead.csv
outputs/phase0/displacement_vs_lead.png
outputs/phase0/displacement_uncensored_raw.csv
outputs/phase0/displacement_uncensored_by_lead.csv
outputs/phase0/displacement_uncensored_by_lead.png
```

Uncensored diagnostic summary (`outputs/phase0/displacement_uncensored_by_lead.csv`):

```text
method,lead_bin,n,mean_km,median_km,p75_km,p90_km,cap_fraction,vo_fraction
hybrid_100,000-024,1686,39.362305282136234,33.80307461416837,55.1969350374903,78.00804116044807,0.0771055753262159,0.0771055753262159
hybrid_100,024-048,1686,57.73564787608074,57.42651204053102,83.39536393490421,93.7697491285814,0.21233689205219455,0.21233689205219455
hybrid_100,048-072,1686,71.16904916919114,80.8096189912002,92.66308308933343,97.24707827807465,0.4543297746144721,0.4543297746144721
hybrid_100,072-096,1686,77.70996265038707,87.56492777335286,94.73767910152577,97.75823500145202,0.6287069988137604,0.6287069988137604
hybrid_100,096-120,1686,80.70347630483332,89.44611875956832,95.28152907010204,98.03330072512199,0.7147093712930012,0.7147093712930012
msl_descent_500,000-024,1686,45.21200275993781,33.844046297660455,56.364362750610425,88.32637163422542,0.002372479240806643,0.0
msl_descent_500,024-048,1686,76.4747045521639,58.68052638498072,93.57978297858389,138.42177797416235,0.011269276393831554,0.0
msl_descent_500,048-072,1686,117.42157603705172,93.5722207804443,147.0781451469336,227.43313218981552,0.022538552787663108,0.0
msl_descent_500,072-096,1686,156.9762784352594,126.8253618626799,201.378512228748,318.68090126029955,0.04151838671411625,0.0
msl_descent_500,096-120,1686,191.25817055522862,157.23805093080796,260.5991851613263,419.0761189207283,0.07473309608540925,0.0
```

Center-source counts from `outputs/phase0/displacement_uncensored_raw.csv`:

```text
method           center_source
hybrid_100       msl              5027
                 vo               3827
msl_descent_500  msl              8854
```

Critical hybrid gate statistic:

```text
vo fallback median distance to truth = 91.60 km (n=3827)
required by Step 5: clearly below the old cap-point reference of 82 km
```

### 23.4 Step 5 Gate Conclusion

Step 5 hybrid gate fails.

Reason:

- The `vo_850` fallback centers are not closer than the msl cap-point reference.
- The overall hybrid curve is better than uncensored 500 km MSL descent at long lead, but the direct fallback quality requirement is not met.
- Per the Step 5 document, no hybrid label-cache rebuild, no hybrid retrain, and no Phase E cross-year run should be performed after this failure.

Current config state after the failed gate:

- `paths.ibtracs_csv` remains `F:/typhoon_loc/data/ibtracs/georef_2020_2025.csv`.
- `labels.center_criterion` is reverted to `"msl"` in both `configs/finetune.yaml` and `configs/infer.yaml`.
- `data.usable_months` remains the original Phase D seven-month set: `2024-05` through `2024-11`.

Recommended next independent variables:

1. Tier B input channels, especially adding `t_850`, because vo-only fallback did not improve the cap-limited label center enough.
2. A controlled partial-unfreeze experiment after Tier B, not mixed into the same step.
3. A separate cross-year validation only after a label criterion passes its own gate; the truth and alignment audit has already unlocked `2025-07` and `2025-08` for that future phase.

### 23.5 Verification

Commands:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts tests
D:\study\envs\tc_loc\python.exe -m pytest -q
```

Results:

```text
compileall: passed
pytest: 11 passed in 15.03s
```

## 24. E-a Zero-Cost Cross-Year Evaluation

This section records the work from `D:\downloads\STEP_EA_CROSSYEAR_EVAL.md`.

Scope obeyed:

- No training was run.
- No label cache was rebuilt.
- No normalization stats were recomputed.
- `configs/finetune.yaml` was not changed.
- `configs/infer.yaml` was changed only for E-a inference/evaluation: `data.usable_months = ["2025-07", "2025-08"]`.
- `scripts/predict.py` gained `--output` and applies configured `finetune.lead_max` to AIFS inference file selection.
- `scripts/evaluate.py` gained `--suffix` and applies configured `finetune.lead_max` to AIFS reference construction.

### 24.1 Checkpoint Verification

Phase D checkpoint byte identity was verified:

```text
outputs/finetune_best.ckpt
outputs/m4_phaseD_lead120_infield/finetune_best.ckpt
```

Both files have the same SHA256:

```text
DD4BB25F35A1D5B6AA06ED14DE64CE45D9E4348663DFA029AC5E05464732C2D2
```

Therefore E-a used the current `outputs/finetune_best.ckpt` through `configs/infer.yaml`.

### 24.2 Commands

Prediction:

```powershell
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split all --output outputs\predictions_2025.csv
```

Evaluation:

```powershell
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split all --predictions outputs\predictions_2025.csv --suffix _2025
```

Outputs:

```text
outputs/predictions_2025.csv
outputs/matched_metrics_2025.csv
outputs/metrics_by_lead_2025.csv
outputs/metrics_by_month_2025.csv
outputs/metrics_by_month_lead_2025.csv
outputs/precision_recall_2025.csv
outputs/cross_year_2024train_2025test.csv
```

PowerShell wrapped the `pynvml` FutureWarning as a `NativeCommandError`, but both scripts completed and wrote all expected outputs.

### 24.3 2025 Metrics

`outputs/metrics_by_lead_2025.csv`:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,619,0.6736672051696284,26.68899974260427,33.79127836731036,44.47339217184459
024-048,620,0.6064516129032258,27.79727413682221,54.98566689036615,78.30617448121407
048-096,1238,0.420032310177706,83.39860083071966,75.8525234917184,152.2679843391481
096-120,619,0.2455573505654281,192.99639210457883,77.52593579727674,250.42036881755487
```

`outputs/metrics_by_month_2025.csv`:

```text
year_month,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
2025-07,1597,0.43519098309329995,83.39971292076079,72.3471388583187,152.2679843391481
2025-08,1655,0.4827794561933535,55.60655685503367,67.57196262688257,118.38048103454257
```

`outputs/metrics_by_month_lead_2025.csv`:

```text
year_month,lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
2025-07,000-024,278,0.7194244604316546,26.078664670293428,37.36540698490772,44.29336210196678
2025-07,024-048,292,0.6472602739726028,27.795018453143882,59.742304093080385,72.88576563652262
2025-07,048-096,622,0.38263665594855306,98.64025191711735,77.38211585956243,168.22197501029135
2025-07,096-120,323,0.1826625386996904,299.55463130111536,76.8427715483368,329.0954198469375
2025-08,000-024,341,0.6363636363636364,26.811495251190035,30.77794554568176,44.47339217184459
2025-08,024-048,328,0.5701219512195121,27.805575299644474,53.5392397883598,83.55174001375545
2025-08,048-096,616,0.4577922077922078,60.507660291689135,73.93204881117899,124.73418110389579
2025-08,096-120,296,0.3141891891891892,131.90233324184953,77.85927277612325,200.29151320176965
```

`outputs/precision_recall_2025.csv`:

```text
conf_thresh,precision,recall
0.1,0.23601895734597156,0.45940959409594095
0.2,0.33647718582717306,0.4059040590405904
0.3,0.39531727180183235,0.3582410824108241
0.5,0.46225895316804405,0.2579950799507995
0.7,0.501532175689479,0.1509840098400984
```

### 24.4 Cross-Year Comparison

`outputs/cross_year_2024train_2025test.csv`:

```text
lead_bin,recall_2024_val,end2end_2024_val_km,recall_2025_test,end2end_2025_test_km,recall_delta_abs,recall_rel_change,end2end_delta_km,end2end_rel_change
000-024,0.6960526315789474,41.67780291017181,0.6736672051696284,44.47339217184459,-0.022385426409318954,-0.032160536996375055,2.7955892616727738,0.06707621483066438
024-048,0.65748031496063,69.93114192358084,0.6064516129032258,78.30617448121407,-0.051028702057404174,-0.0776125169016806,8.375032557633233,0.11976112969505458
048-096,0.4494750656167979,134.6083091745813,0.420032310177706,152.2679843391481,-0.029442755439091917,-0.06550475808638843,17.659675164566778,0.13119305392702707
096-120,0.300132802124834,201.875570731127,0.2455573505654281,250.42036881755487,-0.05457545155940591,-0.18183767709837453,48.544798086427875,0.240468908202288
```

Interpretation:

- Cross-year degradation is within the E-a `~25%` guideline for every lead bin.
- Recall relative decline ranges from `3.2%` to `18.2%`.
- End-to-end median relative degradation ranges from `6.7%` to `24.0%`.
- The weakest cell is `2025-07 / 096-120h`: recall `0.183`, end-to-end median `329.10 km`. This is materially weaker, but it is not a hard collapse under the document's examples: recall is not halved relative to the 2024 aggregate, and end-to-end error is not doubled.
- `2025-08` is close to or better than the 2024 aggregate at long lead (`096-120h end2end=200.29 km` vs 2024 val `201.88 km`).

E-a conclusion:

```text
PASS_MEASURED: Phase D checkpoint is cross-year usable on unlocked 2025-07/08, with mild to moderate degradation and a known weak long-lead July cell.
```

### 24.5 Verification

Commands:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall scripts tests tclocator
D:\study\envs\tc_loc\python.exe -m pytest -q
```

Results:

```text
compileall: passed
pytest: 11 passed in 10.71s
```

## 20. Step 4 Phase C Short-Lead Scaled Retrain

Phase C was run after Phase A/B using only the Phase A usable AIFS months:

```text
2024-05, 2024-06, 2024-07, 2024-08, 2024-09, 2024-10, 2024-11
```

Configuration for Phase C:

```text
labels.mode = ibtracs
finetune.lead_max = 24
split.group_by = year_month
split.val_groups_override = ["2024-08", "2024-10"]
finetune.freeze_encoder = true
```

Commands run:

```powershell
D:\study\envs\tc_loc\python.exe scripts\compute_norm_stats.py --config configs\finetune.yaml --domain aifs
D:\study\envs\tc_loc\python.exe scripts\build_label_cache.py --config configs\finetune.yaml --domain aifs
D:\study\envs\tc_loc\python.exe scripts\finetune.py --config configs\finetune.yaml
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split val
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split val --predictions outputs\predictions_val.csv
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split train
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split train --predictions outputs\predictions_train.csv
```

Training split and curve:

```text
AIFS split group_by=year_month train n=760 val n=310 val_groups=['2024-08', '2024-10']
epoch=1 train_loss=1.0928 val_center_mae_km=49.64
epoch=2 train_loss=0.9121 val_center_mae_km=54.49
epoch=3 train_loss=0.8621 val_center_mae_km=59.61
epoch=4 train_loss=0.8162 val_center_mae_km=57.72
epoch=5 train_loss=0.7748 val_center_mae_km=59.70
epoch=6 train_loss=0.7235 val_center_mae_km=68.41
epoch=7 train_loss=0.6847 val_center_mae_km=69.19
epoch=8 train_loss=0.6477 val_center_mae_km=71.50
epoch=9 train_loss=0.6071 val_center_mae_km=71.32
```

Phase C validation metrics (`outputs/m4_phaseC_lead24_baseline/metrics_by_lead_val.csv`):

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,760,0.6578947368421053,38.74313233075392,0.0,38.74313233075392
024-048,762,0.42650918635170604,68.27191034965425,0.0,68.27191034965425
048-096,1524,0.18110236220472442,126.62751666804043,0.0,126.62751666804043
096-120,753,0.0849933598937583,212.06335748378166,0.0,212.06335748378166
```

Phase C train metrics:

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,926,0.8034557235421166,33.2808307705589,0.0,33.2808307705589
024-048,924,0.5443722943722944,55.02913416701193,0.0,55.02913416701193
048-096,1848,0.1985930735930736,117.43209961139458,0.0,117.43209961139458
096-120,933,0.08467309753483387,213.0160730928539,0.0,213.0160730928539
```

Phase C gate conclusion:

```text
PASS
```

Rationale:

- Short-lead validation `000-024` median end-to-end error is `38.74 km`, below the 70 km gate used in this project.
- Short-lead validation recall is `0.658`, above the 0.5 gate.
- Held-out months do not collapse: August `40.69 km` / recall `0.627`, October `35.22 km` / recall `0.711`.

Phase C artifacts were copied to:

```text
outputs/m4_phaseC_lead24_baseline/
```

## 21. Step 4 Phase D Long-Lead In-Field Retrain

Phase D used the Phase B long-lead decision as the only intentional label/lead change from Phase C.

Configuration for Phase D:

```text
labels.mode = in_field
finetune.lead_max = 120
labels.search_radius_km = 100
split.group_by = year_month
split.val_groups_override = ["2024-08", "2024-10"]
finetune.freeze_encoder = true
train.num_workers = 0
train.log_every_batches = 400
```

Runtime notes:

- The first Phase D run with `train.num_workers=2` failed in a Windows DataLoader worker with `MemoryError` during `calc_vo850()`.
- This was a worker memory/process-pressure issue, not a CUDA OOM. The failing traceback showed a `numpy.core._exceptions._ArrayMemoryError` while allocating a small gradient temporary array.
- The fix was to set `train.num_workers=0`, keeping model, loss, decode, field center, and vorticity algorithms unchanged.
- `scripts/finetune.py` now saves the best checkpoint immediately after each validation improvement and supports `train.log_every_batches` for long real-data runs.

Commands run:

```powershell
D:\study\envs\tc_loc\python.exe scripts\build_label_cache.py --config configs\finetune.yaml --domain aifs
D:\study\envs\tc_loc\python.exe scripts\finetune.py --config configs\finetune.yaml
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split val
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split val --predictions outputs\predictions_val.csv
D:\study\envs\tc_loc\python.exe scripts\predict.py --config configs\infer.yaml --domain aifs --split train
D:\study\envs\tc_loc\python.exe scripts\evaluate.py --config configs\infer.yaml --split train --predictions outputs\predictions_train.csv
```

Training split and curve:

```text
AIFS split group_by=year_month train n=3192 val n=1302 val_groups=['2024-08', '2024-10']
epoch=1 train_loss=1.1047 val_center_mae_km=121.00
epoch=2 train_loss=0.9520 val_center_mae_km=113.31
epoch=3 train_loss=0.8597 val_center_mae_km=127.69
epoch=4 train_loss=0.7759 val_center_mae_km=111.72
epoch=5 train_loss=0.7019 val_center_mae_km=131.81
epoch=6 train_loss=0.6323 val_center_mae_km=141.63
epoch=7 train_loss=0.5768 val_center_mae_km=166.68
epoch=8 train_loss=0.5190 val_center_mae_km=182.88
epoch=9 train_loss=0.4652 val_center_mae_km=151.15
epoch=10 train_loss=0.4179 val_center_mae_km=147.48
epoch=11 train_loss=0.3800 val_center_mae_km=208.95
epoch=12 train_loss=0.3440 val_center_mae_km=176.09
```

Best checkpoint:

```text
outputs/finetune_best.ckpt
best epoch = 4
best val_center_mae_km = 111.72
```

Phase D validation metrics (`outputs/metrics_by_lead_val.csv`):

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,760,0.6960526315789474,26.901683822564014,30.893614518347572,41.67780291017181
024-048,762,0.65748031496063,27.79636372125148,55.45486552225859,69.93114192358084
048-096,1524,0.4494750656167979,69.03655479954406,77.55138916749934,134.6083091745813
096-120,753,0.300132802124834,138.99918690274026,79.99888350441337,201.875570731127
```

Phase D train metrics (`outputs/metrics_by_lead_train.csv`):

```text
lead_bin,n_ref,recall,loc_error_median_km,track_bias_median_km,end2end_median_km
000-024,926,0.908207343412527,25.345536823324803,37.908936914515884,38.87976152307055
024-048,924,0.8917748917748918,25.766761009923528,62.4667662782076,61.04620239653988
048-096,1848,0.7997835497835498,27.794947261061484,77.93900095000073,84.15285740266657
096-120,933,0.7009646302250804,37.776942862100356,78.6821913642163,94.64524627521425
```

Phase C vs Phase D validation comparison (`outputs/compare_phaseC_phaseD_by_lead_val.csv`):

```text
lead_bin,recall_phaseC,recall_phaseD,recall_delta,end2end_phaseC,end2end_phaseD,end2end_delta,end2end_rel_change
000-024,0.6579,0.6961,0.0382,38.7431,41.6778,2.9347,0.0757
024-048,0.4265,0.6575,0.2310,68.2719,69.9311,1.6592,0.0243
048-096,0.1811,0.4495,0.2684,126.6275,134.6083,7.9808,0.0630
096-120,0.0850,0.3001,0.2151,212.0634,201.8756,-10.1878,-0.0480
```

Phase D gate conclusion:

```text
PASS for Step 4 Phase D gate, but with clear overfitting warning.
```

Rationale:

- Short-lead validation `000-024` end-to-end median increased from `38.74 km` to `41.68 km`, a `7.6%` regression, under the Step 4 `15%` maximum regression gate.
- `024-048` end-to-end median changed from `68.27 km` to `69.93 km`, a `2.4%` regression.
- Long-lead recall improved materially: `096-120` recall increased from `0.085` to `0.300`, and `096-120` end-to-end median improved from `212.06 km` to `201.88 km`.
- Local detection error improved in every validation bin, but tracking bias appears in Phase D because the model now emits more long-lead detections and the current tracker links them with nonzero displacement error.
- The training curve overfits after epoch 4; keep the epoch-4 best checkpoint and do not use later weights.

Phase D artifacts were copied to:

```text
outputs/m4_phaseD_lead120_infield/
```

## 22. Current State After Step 4 Phase D

Current committed-or-to-commit code/config state:

- `configs/finetune.yaml` and `configs/infer.yaml` are currently set for the Phase D configuration: `labels.mode: "in_field"`, `finetune.lead_max: 120`, AIFS dir `G:/AIFS_PT`, usable months May-Nov 2024, and validation months August/October 2024.
- `scripts/finetune.py` now writes best checkpoints immediately and supports `train.log_every_batches`.
- `scripts/evaluate.py` writes `metrics_by_month_lead_{split}.csv` in addition to the earlier outputs.
- `scripts/predict.py` prints AIFS progress every 200 files.
- AIFS `.pt` channel reading now loads each tensor once per file for multi-channel reads.
- `build_label_cache.py` reads only `msl` when building AIFS labels, avoiding unnecessary channel/vorticity work during label-cache generation.

Recommended next technical step:

- Step 5 should focus on reducing the Phase D tracking bias and overfitting without changing D1-D6. The most direct candidates are tracker calibration by lead and/or the previously deferred `vo_850` hybrid field-center criterion. Do not introduce a block-detection paradigm.

Verification after Phase D:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts tests
D:\study\envs\tc_loc\python.exe -m pytest -q
```

Results:

```text
compileall: passed
pytest: 9 passed in 10.15s
```

Environment note:

- `pytest` is declared in `requirements.txt` but was missing from the local `tc_loc` environment. It was installed with `D:\study\envs\tc_loc\python.exe -m pip install pytest` before running tests.

## 19. Step 4 Phase B Aligned Displacement And Cap Diagnostics

Phase B is complete on the Phase A usable months only:

```text
2024-05, 2024-06, 2024-07, 2024-08, 2024-09, 2024-10, 2024-11
```

No blocked month was used.

Additional code changes made during Phase B:

- `scripts/phase0_consistency_and_displacement.py` accepts `--lead-max` and progress logging, and was run at `--lead-max 120`.
- `scripts/check_field_center.py` accepts `--lead-max` and progress logging, and was run at `--lead-max 120`.
- `tclocator/io_aifs.py` now uses `torch.load(..., mmap=True)` when supported, with fallback to normal `torch.load`. This is an I/O performance change only; it does not change `.pt` orientation, channel order, vorticity, normalization, labels, model, loss, or decode logic.
- `scripts/check_aifs_alignment.py` was rerun after the mmap change and still passed with the same Yagi lead-0 values.

### 19.1 Alignment Regression Check

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\check_aifs_alignment.py --config configs\finetune.yaml
```

Output:

```text
2024244N09137 valid=2024-09-07T12:00:00 truth=(21.00,106.00) msl_truth_hPa=982.95 min100_hPa=981.59 min100=(21.25,106.00) min100_dist_km=27.80 PASS
PASS
```

### 19.2 Displacement Vs Lead

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\phase0_consistency_and_displacement.py --config configs\finetune.yaml --lead-max 120 --progress-every 200
```

Progress completed through `processed_fields=3200 rows=8605`.

Output:

```text
Wrote F:\typhoon_loc\outputs\phase0\displacement_vs_lead.csv
Wrote F:\typhoon_loc\outputs\phase0\displacement_summary_by_lead.csv
Wrote F:\typhoon_loc\outputs\phase0\displacement_vs_lead.png
vo_850 consistency: {'status': 'precomputed_vo850_without_uv', 'message': 'ERA5 只有预计算 vo_850，缺少 u850/v850，无法自动验证 D5 口径一致性', 'error': "KeyError: 'ERA5 variable not found in dataset: u850'", 'pass': False}
建议 labels.mode = in_field
建议 finetune.lead_max = 48
建议 labels.search_radius_km = 100
```

`outputs/phase0/displacement_summary_by_lead.csv`:

```text
lead_bin,n,mean_km,median_km,p75_km,p90_km
000-024,1686,39.66374268977234,33.844046297660455,56.364362750610425,78.37871028789431
024-048,1686,57.14054762905627,58.68052638498072,79.88665521206742,90.20761000221628
048-072,1686,68.5063172508479,75.62983201911334,87.13710707079926,94.73458656574616
072-096,1686,73.87418069042869,79.05801236499545,88.95594131564711,95.13083888085572
096-120,1686,75.22451963059538,79.18061592992073,89.07203822180401,95.1121651594571
```

Interpretation:

- `0-24h` and `24-48h` remain under the ~75 km median threshold.
- `48-72h` is slightly above the threshold at `75.63 km`; later bins remain near `79 km`.
- Phase0 therefore recommends `labels.mode = in_field` and `finetune.lead_max = 48`.

### 19.3 Cap Statistics By Lead

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\check_field_center.py --config configs\finetune.yaml --lead-max 120 --max-samples 20000 --progress-every 200
```

The script processed all available matching samples under 120h:

```text
old_global_argmin: median_dist_to_truth=110.18 km (n=8854)
new_local_descent: median_dist_to_truth=71.94 km (n=8854)
stop_reason: local_min=5027 cap=3827 cap_fraction=0.432
median_dist local_min=46.18 km cap=82.38 km
```

The legacy overall PASS criterion returned `FAIL` because the old global-argmin median on aligned data is no longer more than 2x worse than local descent. For Step 4, the required evidence is the lead-binned cap table below, which was written successfully.

`outputs/diagnostics/field_center_by_lead.csv`:

```text
lead_bin,n,cap_fraction,old_global_argmin_median_km,new_local_descent_median_km,median_dist_local_min_km,median_dist_cap_km
000-024,1686,0.0771055753262159,38.77614337038442,33.844046297660455,31.63539091034072,81.9870435051914
024-048,1686,0.21233689205219455,67.1484549551124,58.68052638498072,47.850761593725366,82.73160590977032
048-072,1686,0.4543297746144721,112.31595337726954,75.62983201911334,56.549958653925806,83.28366894066943
072-096,1686,0.6287069988137604,160.2056485057286,79.05801236499545,61.077357986699596,83.0162008209699
096-120,1686,0.7147093712930012,223.8744177820961,79.18061592992073,61.762743868369384,81.4739899425099
```

Interpretation:

- Cap fraction is low for `0-24h` (`0.077`) and moderate for `24-48h` (`0.212`).
- Cap fraction exceeds `0.40` from `48-72h` onward and reaches `0.715` at `96-120h`.
- Per Step 4 rules, this marks long-lead in-field labeling as the baseline route, while `vo_850` hybrid center judging should be left as a separate Step 5 variable.

### 19.4 Phase B Decision

Phase B decision:

```text
labels.mode = in_field
finetune.lead_max = 48
labels.search_radius_km = 100
```

Rationale:

- `ibtracs` remains appropriate for short-lead Phase C because Step 3 already proved short-lead performance and Phase C must change only data scale.
- For long-lead extension, aligned displacement crosses the ~75 km median threshold immediately after 48h and cap_fraction becomes high, so direct IBTrACS labels should not be used for long lead.
- Because long-lead cap_fraction is high, Step 5 should consider a vo_850 hybrid center criterion, but Step 4 must not implement it.

## 18. Step 4 Phase A Data Coverage And Alignment Gate

Step 4 started from `D:\downloads\STEP4_SCALE_DATA.md`. Phase A-0/A is complete. No downstream retraining was run before the coverage and monthly alignment gate.

Code/config changes made for Phase A and later gates:

- `configs/finetune.yaml` and `configs/infer.yaml` now use `paths.aifs_dir: "G:/AIFS_PT"`.
- Both configs now include `data.usable_months`, currently set to `["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]`.
- `tclocator/common.py::iter_files` was already recursive via `rglob("*")`; no change was needed there.
- `tclocator/split.py` now filters AIFS files by configured usable initialization month, supports `split.val_groups_override`, and uses AIFS init month for `group_by: "year_month"` even when long leads cross into the next valid-time month.
- AIFS norm stats, label cache, fine-tuning, Phase0 displacement, prediction/evaluation split selection, and field-center diagnostics now share the same usable-month filtering path.
- `scripts/evaluate.py` now writes `metrics_by_month_{split}.csv`.
- `scripts/check_field_center.py` now writes lead-binned cap diagnostics to `outputs/diagnostics/field_center_by_lead.csv`.

### 18.1 Phase A Command

Command:

```powershell
D:\study\envs\tc_loc\python.exe scripts\audit_data_coverage.py --config configs\finetune.yaml --max-cases-per-month 200
```

Result:

```text
PASS_WITH_BLOCKED_MONTHS
```

The script wrote:

```text
outputs/audit/aifs_inventory.csv
outputs/audit/unparseable.csv
outputs/audit/ibtracs_coverage.csv
outputs/audit/aifs_truth_join.csv
outputs/audit/monthly_alignment_check.csv
outputs/audit/coverage_decision.json
```

### 18.2 AIFS And Truth Coverage

`outputs/audit/aifs_truth_join.csv`:

```text
year_month,n_files,n_inits,lead_min,lead_max,n_leads_per_init_median,missing_leads_examples,truth,n_records,n_sids
2024-04,1230,30,0,240,41.0,,MISSING,0,0
2024-05,1271,31,0,240,41.0,,OK,56,2
2024-06,1230,30,0,240,41.0,,OK,47,5
2024-07,1271,31,0,240,41.0,,OK,171,8
2024-08,1271,31,0,240,41.0,,OK,479,16
2024-09,1230,30,0,240,41.0,,OK,409,21
2024-10,1271,31,0,240,41.0,,OK,280,13
2024-11,1230,30,0,240,41.0,,OK,244,10
2024-12,1271,31,0,240,41.0,,OK,19,1
2025-01,1271,31,0,240,41.0,,MISSING,0,0
2025-02,1066,26,0,240,41.0,,MISSING,0,0
2025-07,1271,31,0,240,41.0,,MISSING,0,0
2025-08,1271,31,0,240,41.0,,MISSING,0,0
```

Notes:

- AIFS months present on `G:/AIFS_PT`: `2024-04` through `2025-08` except `2025-03` through `2025-06`.
- Each present month has full 6-hour lead coverage with median `41` leads per init (`0..240h`).
- Current `data/ibtracs/georef.csv` covers only through 2024 and has no truth for 2025 months.
- `2024-04` AIFS exists but has no truth in the configured IBTrACS CSV.

### 18.3 Monthly Alignment Decision

`outputs/audit/coverage_decision.json`:

```json
{
  "truth_usable_months": ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"],
  "truth_blocked_months": ["2024-04", "2025-01", "2025-02", "2025-07", "2025-08"],
  "alignment_pass_months": ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"],
  "alignment_blocked_months": ["2024-12"],
  "blocked_months": ["2024-04", "2024-12", "2025-01", "2025-02", "2025-07", "2025-08"],
  "recommended_data_usable_months": ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10", "2024-11"]
}
```

Interpretation:

- `2024-05` through `2024-11` pass the monthly AIFS `.pt` alignment gate.
- `2024-12` has truth, but the available storm is weak in AIFS: all checked short-lead candidates have `min100_hPa >= 1002.59 hPa`. It cannot serve as a reliable deep-low alignment gate month, so it is explicitly blocked from training/evaluation.
- 2025 months are blocked only because truth is missing. They can be enabled later only after adding matching `ISO_TIME,SID,LAT,LON` truth rows to `data/ibtracs`.

### 18.4 Alignment Evidence

The first passing row for each alignment-passed month:

```text
year_month,sid,valid_time,lead_hour,msl_truth_hPa,min100_hPa,min100_dist_km,status
2024-05,2024141N03142,2024-05-27T12:00:00+00:00,0,998.2,993.38,28.81,RELAXED_PASS
2024-06,2024141N03142,2024-06-01T12:00:00+00:00,0,993.0875,992.7675,27.33,RELAXED_PASS
2024-07,2024181N09320,2024-07-01T12:00:00+00:00,0,985.5591,985.5591,10.88,PASS
2024-08,2024213N14254,2024-08-02T12:00:00+00:00,0,999.6781,999.0781,16.79,RELAXED_PASS
2024-09,2024244N09137,2024-09-03T12:00:00+00:00,0,995.5358,993.8558,38.65,RELAXED_PASS
2024-10,2024269N14150,2024-10-01T12:00:00+00:00,0,991.6519,991.0119,28.15,RELAXED_PASS
2024-11,2024307N06143,2024-11-04T12:00:00+00:00,0,998.7762,998.7762,7.74,RELAXED_PASS
```

For `2024-12`, all checked short-lead candidates failed the pressure gate. The lowest `min100_hPa` was `1002.59 hPa`, so the month is blocked rather than used for training.

### 18.5 Phase A Gate Conclusion

Phase A passes with explicit blocked months.

Allowed for Phase B/C/D:

```text
2024-05, 2024-06, 2024-07, 2024-08, 2024-09, 2024-10, 2024-11
```

Explicitly excluded:

```text
2024-04, 2024-12, 2025-01, 2025-02, 2025-07, 2025-08
```

Do not train or evaluate on 2025 AIFS data until 2025 truth is added. Do not train/evaluate on `2024-12` unless a separate manual decision accepts weak-system alignment validation for that month.

### 18.6 Verification

Commands:

```powershell
D:\study\envs\tc_loc\python.exe -m compileall tclocator scripts tests
D:\study\envs\tc_loc\python.exe -c "import runpy; ns=runpy.run_path('tests/test_split.py'); ns['test_grouped_split_keeps_all_leads_of_init_together'](); ns['test_select_aifs_files_uses_same_deterministic_split'](); ns['test_year_month_group_uses_init_month'](); print('split tests ok')"
```

Results:

```text
compileall: passed
split tests: passed
```
