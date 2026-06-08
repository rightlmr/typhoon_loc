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
