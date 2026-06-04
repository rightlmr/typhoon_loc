# TC Locator — 项目实现规格(Codex)

> 目标读者:编码代理(Codex)。本文件是**实现规格**,不是讨论文档。
> 项目:从 ERA5/AIFS 全球气象场中检测并定位热带气旋(TC)中心,采用热力图关键点检测
> (CenterNet 风格 U-Net),ERA5 预训练 + AIFS 短 lead 微调。
> 约定:散文为中文;所有代码标识符、路径、配置键、变量名一律英文。
> **本规格自成体系,不依赖任何外部已有代码库;请完全从零实现。**
> **第 1 节"不可改动的设计决策(D1–D6)"经专门诊断得出,不得替换为"更通用"的方案。**

---

## 0. 一句话目标

输入全球气象场,输出 TC 中心 `(lat, lon, confidence)`。用 ERA5 预训练获得强特征,用 AIFS 预报场微调以适配其分布;**标签使用"场内中心"**,定位误差与预报场路径偏差分开评估。

---

## 1. 不可改动的设计决策(及理由)

| 决策 | 内容 | 理由(不要改) |
|---|---|---|
| **D1 范式** | CenterNet 风格热力图关键点检测:全局场 → 中心置信度热力图 + 亚网格偏移,峰值提取得中心 | 相比"把场切成小块、每块独立预测一个中心"的范式,全局热力图不会把一个空间上被抹平的台风切分到相邻块各报一个中心(从而产生虚假的"多台风"),并以置信度阈值 + 3×3 峰值 NMS 直接控制精度/召回 |
| **D2 迁移** | ERA5 全量预训练 → AIFS 微调(冻结 encoder,低 LR,仅短 lead) | AIFS 对 TC 的海平面气压/涡度信号存在**系统性衰减**(气压凹陷偏浅、涡度被削弱)。系统性偏移可学,但 AIFS 样本量小、直接训会过拟合,必须靠 ERA5 预训练打底 |
| **D3 标签** | 标签 = **TC 在输入场里的中心**(真值邻域内的海平面气压局地极小值),而非长 lead 下的原始最佳路径真位置 | 预报场在长 lead 下预报位置会偏离真位置数百公里。用原始真位置去标长 lead 的预报场,等于把预报场的路径误差注入定位标签,模型两头学不好。场内中心是自洽目标 |
| **D4 归一化** | **逐域** z-score:ERA5 用 ERA5 统计量,AIFS 用 AIFS 统计量,分别计算与存储 | 逐域归一化即一阶矩匹配的域适应,直接抵消系统性均值/尺度偏移,把衰减后的弱涡旋拉回与 ERA5 可比的相对异常 |
| **D5 通道一致** | ERA5 与 AIFS 必须使用**完全相同的通道集 + 完全相同的派生方式**(尤其涡度 vo_850) | 任何通道定义或计算口径不一致都会在迁移时变成额外域偏移。vo_850 在两域必须用同一函数从 u850/v850 派生(见 Phase 0 一致性检查) |
| **D6 输入** | 全场输入,不切块;281×881 padding 到 32 的倍数(288×896),U-Net 4–5 次下采样 | 保留全局空间关系,避免分块边界伪影 |

---

## 2. 数据契约

### 2.1 域与网格(所有数据统一裁剪到此域)
- 纬度:0–70°N;经度:100–320°E;分辨率:0.25°
- 形状:**281 (lat) × 881 (lon)**;lat 由北到南,lon 由西到东(0–360 体系)

### 2.2 数据根目录
- 项目根:`F:\typhoon_loc`
- 数据根:`F:\typhoon_loc\data`(代码生成完毕、开始训练前才会填充数据)
- 默认子目录布局(具体文件命名与变量名由 config 填写,代码必须可配置,不要写死):

```
F:\typhoon_loc\data\
  era5\        # ERA5 场, 6h 一个时次, .nc (xarray 可读)
  aifs\        # AIFS 预报场 GRIB2
  ibtracs\     # 最佳路径真值 CSV
```

### 2.3 ERA5(预训练数据)
- 6 小时一个时次的 `.nc` 文件,`xarray.open_dataset` 读取
- 需提供字段(经 config 的 `era5.var_map` 映射到内部名):
  - `msl`(海平面气压,单位 Pa)
  - `t_500`(500 hPa 温度,K)
  - 涡度来源二选一(见 D5):**(a) 提供 `u850`/`v850`,由代码统一派生 vo_850(首选)**;(b) 提供预先算好的 `vo_850`(则 Phase 0 §4.1 必须先验证口径)
- 时段须与 IBTrACS 覆盖区间一致(由数据决定,代码不假设具体年份)

### 2.4 AIFS(微调 + 推理数据)—— 自行实现 GRIB2 读取
GRIB2,`pygrib` 读取。**Windows 上 `import pygrib` 必须早于 `pandas`/`torch`**(规避 eccodes DLL 冲突)。

全球网格:721×1440,0.25°,lat = linspace(90, -90, 721),lon = linspace(0, 359.75, 1440)。

读取单变量:`grbs.select(shortName=<sn>, typeOfLevel=<tol>, level=<lvl>)[0].values`。变量→GRIB 键映射:

| 内部名 | shortName | typeOfLevel | level |
|---|---|---|---|
| mslp | msl | meanSea | 0 |
| u10 | 10u | heightAboveGround | 10 |
| v10 | 10v | heightAboveGround | 10 |
| t2 | 2t | heightAboveGround | 2 |
| u850 / v850 / q850 / t850 | u/v/q/t | isobaricInhPa | 850 |
| u700 / v700 / q700 / t700 | u/v/q/t | isobaricInhPa | 700 |
| u500 / v500 / q500 / t500 | u/v/q/t | isobaricInhPa | 500 |

- `mslp` 单位 Pa(与 ERA5 一致,无需换算)
- 气压层仅 850/700/500 hPa
- 文件名解析两种格式:`AIFS_YYYY_MM_DD_HH_FCST_XXXh.grib2` 与 `YYYYMMDDHHMMSS-Hh-oper-fc.grib2`;由文件名得 `(init_time, forecast_hour, valid_time = init_time + forecast_hour)`
- 裁剪:由 `(lat, lon)` 与全球网格 linspace 反查行列索引,切到 §2.1 域;近边界 `np.pad(mode='edge')`

### 2.5 涡度派生(vo_850,两域同一函数)
相对涡度 `vo = ∂v/∂x − ∂u/∂y`,在规则经纬网上:
- `dx = R * Δlon_rad * cos(lat)`,`dy = R * Δlat_rad`,`R = 6371000 m`
- `dv_dx = gradient(v, axis=lon) / dx`,`du_dy = gradient(u, axis=lat) / (−dy)`(注意 lat 由北到南,故取负)
- `vo = dv_dx − du_dy`
- 实现为单一函数 `calc_vo850(u, v, lat1d, lon1d)`,ERA5 与 AIFS 都调它,保证口径一致

### 2.6 IBTrACS(标签源)
- CSV,`pandas` 读取;需含列(经 config 的 `ibtracs.col_map` 映射):`ISO_TIME, SID, LAT, LON`
- 仅覆盖其自身年份区间,训练时段不得超出

### 2.7 通道集(可配置,默认 Tier A)
模型必须 channel-agnostic(通道从 config 读):

- **默认 Tier A:`["msl", "vo_850", "t_500"]`**(三者在 ERA5 与 AIFS 中均可直接取得/派生,零额外数据依赖)
- 升级路径(仅改 config 通道列表):如需加 `t_850`,要求 ERA5 端也提供 850 hPa 温度。代码不得对通道数目硬编码

---

## 3. 仓库结构(根 = `F:\typhoon_loc`)

```
F:\typhoon_loc\
  configs\
    pretrain.yaml
    finetune.yaml
    infer.yaml
  tclocator\
    __init__.py
    common.py          # 域/网格常量, latlon<->grid, padding, haversine
    io_era5.py         # ERA5 .nc 读取 + 裁剪 + 通道堆叠 (var_map 驱动)
    io_aifs.py         # AIFS GRIB2 读取 + 裁剪 + 通道堆叠 (自行实现, 见 §2.4)
    vorticity.py       # calc_vo850 (两域共用, 见 §2.5)
    normalization.py   # 逐域 z-score 统计量 计算/加载/应用 (D4)
    labels.py          # IBTrACS -> 场内中心 -> 高斯热力图 + offset (D3)
    dataset.py         # torch Dataset (ERA5/AIFS 共用, 以 domain 参数区分)
    model.py           # U-Net + heatmap head + offset head (D1)
    losses.py          # focal heatmap loss + L1 offset loss
    decode.py          # heatmap -> 峰值 NMS -> (lat, lon, conf)
    tracking.py        # 跨 lead 贪心最近邻关联 + 最短长度过滤
    metrics.py         # 三路误差分解
  scripts\
    phase0_consistency_and_displacement.py   # 必须先跑 (见 Phase 0)
    compute_norm_stats.py
    build_label_cache.py
    pretrain.py
    finetune.py
    predict.py
    evaluate.py
  data\                # 见 §2.2 (训练前才填充)
  README.md
  requirements.txt
```

---

## 4. Phase 0 — 前置诊断(运行顺序上的硬门禁)

脚本:`scripts/phase0_consistency_and_displacement.py`。其结论用于填写 `finetune.yaml` 的 `labels.mode` 与 `finetune.lead_max`。训练脚本须断言这两项已被填(非 null),否则报错退出并提示"先运行 Phase 0"。

### 4.1 vo_850 口径一致性检查(对应 D5)
- 若 ERA5 数据提供 u850/v850 → 用 `calc_vo850` 派生,与 AIFS 同函数派生天然一致,记录即可
- 若 ERA5 仅提供预算 vo_850 → 在同一时刻同一域上,把它与"用 ERA5 的 u850/v850 经 `calc_vo850` 派生"的结果对比(相关系数、峰值比、RMSE)
- **门禁**:若峰值比偏离 1 超过 ±20%,说明口径不一致 → 必须统一(首选:两域都用 `calc_vo850` 从 u/v 派生)。在确认前不要进入训练

### 4.2 真值 vs 场内 msl-min 偏移随 lead 曲线(决定 D3 的半径与 lead 上限)
对 AIFS 验证数据中的若干真台风(逐 SID、逐 valid_time、逐 lead):
1. 取 IBTrACS 真位置 `(lat_true, lon_true)`
2. 在 AIFS 场中,以真位置为中心、半径 R0(默认 500 km)内找 msl 局地极小值位置 `(lat_field, lon_field)`
3. 记录 `displacement_km = haversine(true, field)` 与 `lead_hour`
4. 输出:`displacement vs lead` 散点 + 分 lead 区间(0–24/24–48/48–96/96–120h)的中位/均值/分位表,存 CSV + PNG

**结论与决策(脚本须打印建议值)**:
- 各 lead 偏移均小(中位 < ~75 km)→ `labels.mode = ibtracs`
- 偏移随 lead 显著增长 → 设 `labels.mode = in_field`,且/或设 `finetune.lead_max`(只用偏移可接受的短 lead 段微调)
- 由偏移分布定 `labels.search_radius_km`(建议取 P90 量级,默认 300 km)

---

## 5. labels.py — 标签生成(对应 D3)

输入:某 valid_time 的场(ERA5 或 AIFS)+ 该时刻 IBTrACS 记录(0..N 个台风)。
输出:`heatmap [H, W]` float32 ∈ [0,1]、`offset [2, H, W]` float32、`mask [H, W]` uint8(正样本像素 = 1)。

每个台风:
1. 取参考中心:
   - `mode == "ibtracs"`:直接用 `(LAT, LON)`
   - `mode == "in_field"`:在真位置 `search_radius_km` 内找 msl 局地极小值作为参考中心
2. 参考中心 → grid 浮点坐标 `(cy_f, cx_f)`(`common.latlon_to_grid`)
3. 高斯泼溅:`heatmap[y,x] = max(heatmap[y,x], exp(-((x−cx_f)^2 + (y−cy_f)^2)/(2σ^2)))`,σ = `labels.sigma_px`(默认 3,约 0.75°)
4. offset:在 `(round(cy_f), round(cx_f))` 处写 `(cy_f − floor, cx_f − floor)`,该像素 mask = 1

无台风时刻 → 全 0 热力图(合法负样本,务必保留,用于压低虚检)。

`scripts/build_label_cache.py` 预生成标签缓存(.npz),避免训练时重复计算。

---

## 6. model.py + losses.py(对应 D1)

### 6.1 模型
- Backbone:U-Net(encoder 4–5 级下采样 + skip + decoder);输入通道 = `len(channels)`,输入 padding 到 (288, 896)
- 两个 head(输出 stride = 1,全分辨率):
  - `heatmap_head`:1 通道,sigmoid
  - `offset_head`:2 通道,亚网格偏移
- `freeze_encoder()` 供微调用
- 不引入外部预训练权重;backbone 从零在 ERA5 上训

### 6.2 损失
- heatmap:CenterNet penalty-reduced focal loss(对峰值正样本与周围高斯衰减加权),α=2、β=4(可配置)
- offset:仅在 mask=1 处的 L1
- 总损失:`L = L_heatmap + λ_off * L_offset`,`λ_off` 默认 1.0

---

## 7. decode.py + tracking.py — 推理后处理

### 7.1 decode.py
输入热力图 + offset,输出 `DataFrame[ISO_TIME, LAT, LON, CONF]`:
1. 3×3 max-pool NMS:`keep = (heatmap == maxpool3x3(heatmap))`
2. 阈值:`keep &= heatmap >= conf_thresh`(默认 0.3 —— 精度/召回旋钮)
3. 峰值取 offset 修正 → grid 浮点 → `grid_to_latlon` → `(lat, lon)`,`conf = heatmap` 值
4. 纬度过滤 `lat ∈ [lat_min, lat_max]`(默认 0–40)

### 7.2 tracking.py
- 跨 lead/时次贪心最近邻关联(`max_step_km` 默认 800,`expected_step_hours = 6`),最短长度过滤(`min_len` 默认 4)
- 检测端干净后无需额外的多步物理清洗;若实测仍有轨迹级虚检,再在 tracking 内加物理一致性(步长/转向)约束

---

## 8. 训练流程

### 8.1 数据划分(防泄漏)
- **按 SID/季节划分**,不得按时间步随机划分
- 预训练 ERA5:留出整季或若干 SID 作 val(年份由数据决定)
- 微调 AIFS:见 §10 数据量前置条件

### 8.2 compute_norm_stats.py(对应 D4)
- 分别统计 ERA5 train 集与 AIFS 集每通道 mean/std,产物 `norm_stats_era5.json`、`norm_stats_aifs.json`
- 对长尾量(vo_850)提供 `log1p+zscore` 选项(对非负化后取 log1p 再 z-score)

### 8.3 pretrain.py
- config `configs/pretrain.yaml`;用 ERA5 + `norm_stats_era5.json` + `labels.mode`
- AdamW + cosine LR,early stopping by val center-MAE;存 `pretrain_best.ckpt`

### 8.4 finetune.py
- config `configs/finetune.yaml`;载入 `pretrain_best.ckpt` → `freeze_encoder()` → 仅训 decoder + heads
- AIFS + `norm_stats_aifs.json` + Phase 0 决定的 `labels.mode` / `finetune.lead_max`
- 低 LR(默认预训练的 1/10),少 epoch,强 early stopping;存 `finetune_best.ckpt`
- **启动时断言** `labels.mode` 与 `finetune.lead_max` 非 null,否则退出并提示先跑 Phase 0

---

## 9. 评估(metrics.py)

对 AIFS 验证集,按 lead 分层报告三个量:

| 指标 | 定义 | 衡量谁 |
|---|---|---|
| `loc_error_km` | 预测中心 vs **场内参考中心**(msl-min) | 模型定位技巧(自洽) |
| `track_bias_km` | 场内参考中心 vs **IBTrACS 真值** | 预报场本身(非模型) |
| `end2end_km` | 预测中心 vs **IBTrACS 真值** | 端到端 = 上两者之和 |

另报:召回率(命中真台风数 / 应检数,命中阈值 60 km)、虚警率、按 `conf_thresh` 的 PR 曲线。轨迹级汇总:每条轨迹 match_ratio ≥ 0.6 且 median ≤ 300 km 计为有效。

---

## 10. 前置条件与已知瓶颈(README 顶部显著标注)

1. **AIFS 数据量是微调的硬瓶颈**:若 AIFS 样本仅覆盖很短时段,只够做验证/流程冒烟,不足以有意义地微调。Phase 0 + ERA5 预训练可先做;微调训练待 AIFS 数据补齐。`finetune.py` 在数据不足时应能跑通流程(smoke test)但不得伪造数据。
2. **衰减信息下限**:逐域归一化 + 微调能恢复系统性偏移,但预报场根本没表征的弱信号无法恢复;最弱/初生台风召回必然受限。预期目标应设为相对量(微调 vs 预训练直推),而非绝对值。

---

## 11. 里程碑与验收标准

| 里程碑 | 交付 | 通过标准 |
|---|---|---|
| M0 | Phase 0 脚本 + 两项结论(vo_850 一致性、displacement vs lead) | 输出 CSV/PNG;打印 `labels.mode`、`finetune.lead_max`、`labels.search_radius_km` 建议值 |
| M1 | labels/dataset/model/losses/decode + 单测 | 在一小批 ERA5 上能 overfit(loc_error 收敛 < 30 km);decode 在合成高斯热力图上峰值误差 < 1 px |
| M2 | ERA5 预训练 | val center-MAE 收敛;同一真台风不产生多余峰值(无分块劈裂式多检) |
| M3 | AIFS 推理(预训练权重直推) | 在 AIFS 验证集上跑出 §9 三路指标基线,定量给出 end2end 的成分(定位 vs 路径) |
| M4 | AIFS 微调(数据补齐后) | 相对 M3:召回提升、`loc_error_km` 中位下降 |

---

## 12. 可配置项(集中于 config,逻辑层不得硬编码)

```yaml
# 公共
channels: ["msl", "vo_850", "t_500"]   # 默认 Tier A
domain: {lat_min: 0.0, lat_max: 70.0, lon_min: 100.0, lon_max: 320.0, res: 0.25}
paths:
  era5_dir:    "F:/typhoon_loc/data/era5"
  aifs_dir:    "F:/typhoon_loc/data/aifs"
  ibtracs_csv: "F:/typhoon_loc/data/ibtracs/ibtracs.csv"
era5:
  var_map: {msl: "msl", t_500: "t_500", u850: "u850", v850: "v850"}  # 内部名 -> 文件变量名
  vo850_from_uv: true                # true=由 u850/v850 派生; false=直接读 vo_850
ibtracs:
  col_map: {time: "ISO_TIME", sid: "SID", lat: "LAT", lon: "LON"}
labels:
  mode: null                # "ibtracs" | "in_field"   <- Phase 0 决定 (训练前必填)
  search_radius_km: 300     # <- Phase 0 决定
  sigma_px: 3
decode:
  conf_thresh: 0.3
  lat_filter: [0.0, 40.0]
finetune:
  freeze_encoder: true
  lead_max: null            # <- Phase 0 决定 (训练前必填)
  lr_scale: 0.1
loss: {focal_alpha: 2, focal_beta: 4, lambda_offset: 1.0}
tracking: {max_step_km: 800, min_len: 4, expected_step_hours: 6}
norm: {method: "zscore"}    # vo_850 可单独 "log1p+zscore"
seed: 42
device: "auto"              # cuda 不可用回退 cpu
```

---

## 13. 执行顺序

1. 建 §3 仓库骨架 + `common.py`(域/网格/坐标转换/haversine)
2. 实现 io_era5 / io_aifs / vorticity / normalization / labels / dataset / model / losses / decode / tracking / metrics,过 M1 单测
3. 实现全部 scripts(含 Phase 0)
4. README 写清 §10 前置条件与"运行顺序":填充数据 → 跑 Phase 0 → 把结论填入 config → compute_norm_stats → pretrain → predict/evaluate → (数据补齐后)finetune
5. 训练脚本对 `labels.mode` / `finetune.lead_max` 做非空断言,强制先跑 Phase 0

> 任何与第 1 节 D1–D6 冲突的"简化"或"通用替换",须先在 commit 说明里写明理由并暂停等确认,不要静默改动。
