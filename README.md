# TC Locator

## 前置条件

**AIFS 数据量是微调的硬瓶颈。** 如果 AIFS 样本只覆盖很短时段，只能做验证和流程冒烟，不足以有意义地微调。Phase 0 与 ERA5 预训练可以先做，AIFS 微调应等数据补齐后再启动；代码不会在缺数据时伪造训练样本。

**衰减信息存在下限。** 逐域归一化与 AIFS 微调可以缓解系统性均值/尺度偏移，但无法恢复预报场根本没有表征的弱信号。最弱或初生台风的召回会受限，评估应关注微调相对预训练直推的改进。

## 项目结构

代码按 `TC_LOCATOR_BUILD_SPEC.md` 的 §3 划分：

- `tclocator/common.py`: 网格、坐标、padding、haversine、配置与随机种子。
- `tclocator/io_era5.py`: ERA5 NetCDF 读取、裁剪、通道堆叠。
- `tclocator/io_aifs.py`: AIFS GRIB2 读取、文件名解析、裁剪、通道堆叠；文件顶部优先尝试 `import pygrib`。
- `tclocator/vorticity.py`: ERA5/AIFS 共用 `calc_vo850`。
- `tclocator/normalization.py`: ERA5 与 AIFS 逐域 z-score 统计与应用。
- `tclocator/labels.py`: IBTrACS 或场内 msl-min 中心标签、热力图、offset、mask。
- `tclocator/dataset.py`: 真实场 Dataset 与无数据 smoke synthetic Dataset。
- `tclocator/model.py`: 全场输入 U-Net，输出 heatmap 与 offset。
- `tclocator/losses.py`: CenterNet focal loss 与正样本 offset L1。
- `tclocator/decode.py`: 3x3 NMS、阈值、offset 修正与经纬度输出。
- `tclocator/tracking.py`: 跨时次最近邻轨迹关联。
- `tclocator/metrics.py`: `loc_error_km`、`track_bias_km`、`end2end_km` 分解。

默认配置为 Tier A：`channels: ["msl", "vo_850", "t_500"]`。`labels.mode` 与 `finetune.lead_max` 在配置中保持 `null`，必须由 Phase 0 结论填写。

## 运行顺序

1. 填充数据到 `F:/typhoon_loc/data/era5`、`F:/typhoon_loc/data/aifs`、`F:/typhoon_loc/data/ibtracs`。
2. 运行 Phase 0：
   ```powershell
   conda activate tc_loc
   python scripts/phase0_consistency_and_displacement.py --config configs/finetune.yaml
   ```
3. 将 Phase 0 打印的 `labels.mode`、`finetune.lead_max`、`labels.search_radius_km` 填入 `configs/pretrain.yaml`、`configs/finetune.yaml`、按需填入 `configs/infer.yaml`。
4. 计算逐域归一化统计：
   ```powershell
   python scripts/compute_norm_stats.py --config configs/pretrain.yaml --domain all
   ```
5. 预生成标签缓存：
   ```powershell
   python scripts/build_label_cache.py --config configs/pretrain.yaml --domain era5
   ```
6. ERA5 预训练：
   ```powershell
   python scripts/pretrain.py --config configs/pretrain.yaml
   ```
7. AIFS 直推与评估：
   ```powershell
   python scripts/predict.py --config configs/infer.yaml --domain aifs --checkpoint outputs/pretrain_best.ckpt
   python scripts/evaluate.py --config configs/infer.yaml
   ```
8. AIFS 数据补齐后再微调：
   ```powershell
   python scripts/finetune.py --config configs/finetune.yaml
   ```

`pretrain.py` 会在启动时断言 `labels.mode` 非空；`finetune.py` 会额外断言 `finetune.lead_max` 非空。失败时会打印：

```text
请先运行 scripts/phase0_consistency_and_displacement.py 并将结论填入 config
```

## 无真实数据 smoke test

这些命令只使用合成数据，适合 CI 验证代码路径：

```powershell
python scripts/phase0_consistency_and_displacement.py --config configs/finetune.yaml --smoke-synthetic
python scripts/compute_norm_stats.py --config configs/pretrain.yaml --domain all --smoke-synthetic
python scripts/build_label_cache.py --config configs/pretrain.yaml --smoke-synthetic
pytest
```

## 数据契约

ERA5 使用 6 小时一个时次的 NetCDF 文件，变量名通过 `era5.var_map` 映射。AIFS 使用 GRIB2，支持 `AIFS_YYYY_MM_DD_HH_FCST_XXXh.grib2` 与 `YYYYMMDDHHMMSS-Hh-oper-fc.grib2` 两种命名。ERA5 与 AIFS 的 `vo_850` 都通过同一 `calc_vo850(u850, v850, lat1d, lon1d)` 派生，除非 Phase 0 证明 ERA5 预计算涡度口径一致。

## 设计边界

本项目使用全场热力图关键点检测，不做分块各报中心。长 lead AIFS 不用原始最佳路径真位置直接做标签；标签目标是输入场内自洽中心。ERA5 与 AIFS 使用相同通道集、相同派生方式，并分别应用逐域归一化统计。

