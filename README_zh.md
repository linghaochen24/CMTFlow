# 基于掩码 Transformer 流模型的晶体生成（CMTFlow）

使用掩码 Transformer 归一化流（CMTFlow）进行晶体结构生成的实现代码。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CMTFlow 是单卡实现：把 [TarFlow](http://arxiv.org/abs/2412.06329) 的掩码自回归 Transformer 流迁移到晶体任务上，用一维流把晶体张量映射到高斯先验，`train_cmtflow.py` / `eval_cmtflow.py` 负责训练与评测，`ablation_study/` 下是四组消融。

| 模式 | 开关 | 含义 | 指标 |
|------|------|------|------|
| CSP（条件生成 / 结构预测） | `--csp`（默认开） | 给定目标组分（原子类型 one-hot 条件），生成晶格与分数坐标 | `match_rate`、`sample_match_rate`、归一化 `rms_dist` |
| DNG（从头生成） | `--no-csp` | 不给定任何条件，自由生成晶体 | `wdist_density`、`wdist_num_elems`、`cov_*`、`amsd_*`、`amcd_*`、`novelty_rate`、`uniqueness_rate` |

### 依赖与安装

```
python>=3.10
torch>=2.1          # bfloat16 autocast + GradScaler，建议 CUDA
numpy pandas scipy
pymatgen            # CIF 解析、结构匹配、结构指纹
smact               # 组分（电中性）合法性
pyxtal              # 结构合法性
p_tqdm tqdm
matminer            # 可选，DNG 覆盖率指标需要
plotly kaleido mendeleev   # 可选，仅可视化使用
```

```
conda create -n cmtflow python=3.10 -y
conda activate cmtflow
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy pandas scipy pymatgen smact pyxtal p_tqdm tqdm
pip install matminer plotly kaleido mendeleev
```

下文所有命令均在仓库根目录执行。

### 数据准备

把 CSV 切分放到 `data/<dataset>/{train,val,test}.csv`。`cif` 列是必需项；建议保留 `material_id` 与 `formula`，属性列存在时会被读取。

| 数据集 | 属性列 | train | val | test | max_atom_num |
|--------|--------|-------|-----|------|--------------|
| `perov_5` | `heat_ref` | 11356 | 3787 | 3785 | 5 |
| `mp_20` | `formation_energy_per_atom` | 27136 | 9047 | 9046 | 20 |
| `carbon_24` | `energy_per_atom` | 6091 | 2032 | 2030 | 24 |
| `mpts_52` | `formation_energy_per_atom` | 27380 | 5000 | 8096 | 52 |

把每个切分转换成缓存张量文件 `data/<dataset>/{train,val,test}_ori.pkl`（`--num_workers` 控制 CIF 解析并行度）：

```
python prepare_data_tensor.py --data_path data --dataset mp_20 --num_workers 8
python prepare_data_tensor.py --data_path data --dataset perov_5 --num_workers 8
python prepare_data_tensor.py --data_path data --dataset carbon_24 --num_workers 8
python prepare_data_tensor.py --data_path data --dataset mpts_52 --num_workers 8
```

### 晶体张量表示

每个晶体由 `utils.create_tensor_from_structure` 编码成形状为 `(channels, max_atom_num)` 的张量：

| 切片 | 通道数 | 内容 |
|------|--------|------|
| 类型块 | 28 | 周期表 row(13) + col(15) 的 one-hot，归一化到 ±1 |
| 坐标块 | 90 | 分数坐标保留 3 位小数后逐位 one-hot，归一化到 ±1 |
| 晶格长度 | 3 | 按 `LATTICE_LENGTH_INTERVAL[dataset]` 归一化到 [-1, 1] |
| 晶格角度 | 3 | 角度 / 180 |
| 合计 | 124 | CSP 模式下模型只生成后 96 通道，类型块作为条件 `cond` |

`max_atom_num` 由数据集名后缀决定；`at_mask` 标记真实原子位，padding 位不参与 loss。

### 训练

```
python train_cmtflow.py --dataset perov_5 --data data --logdir runs_cmtflow \
  --transformer_channels 512 --transformer_blocks 8 --layers_per_block 4 \
  --batch_size 256 --epochs 100 --lr 1e-4 --noise_std 0.10 \
  --sample_freq 10 --num_samples 3785 --sample_batch_size 757 --csp
```

主要超参：

| 分组 | 参数 | 默认 | 说明 |
|------|------|------|------|
| 数据 | `--data` / `--dataset` | `data` / `perov_5` | 数据根目录与数据集名（决定 `max_atom_num` 与晶格归一化区间） |
| | `--logdir` | `runs_cmtflow` | checkpoint 与曲线输出目录 |
| | `--limit` | `0` | 只取前 N 条（0 = 全量），调试用 |
| 模型 | `--transformer_channels` | `512` | 模型宽度 |
| | `--transformer_blocks` | `8` | 流 block 数 |
| | `--layers_per_block` | `4` | 每个 block 内部层数 |
| | `--nvp` / `--no-nvp` | `True` | 是否使用 NVP 耦合 |
| | `--noise_std` | `0.10` | 训练输入加性高斯噪声标准差 |
| 优化 | `--batch_size` / `--epochs` / `--lr` | `256` / `100` / `1e-4` | AdamW + cosine 衰减到 `1e-6` |
| 采样 | `--sample_freq` | `10` | 每 N 个 epoch 做一次验证采样 |
| | `--num_samples` / `--sample_batch_size` | `3785` / `757` | 配对的验证记录数；`num_samples` 不能超过 val 切分大小 |
| | `--csp` / `--no-csp` | `True` | CSP 条件生成 / DNG 从头生成 |
| | `--cfg` | `0` | 引导权重（0 = 不用引导） |
| 其它 | `--update_prior` | `False` | 在线更新先验方差 `var`（`--no-nvp` 时使用） |
| | `--dry_run` | `False` | 单批次冒烟测试 |

实现要点：

- 训练目标：`loss = 0.5 * mean(z²) - mean(logdet)`，两项均只在真实原子位上统计；`--update_prior` 打开时按 `var.lerp_(z², 0.1)` 在线更新先验。
- 混合精度 `torch.autocast(bfloat16)` + `GradScaler`，梯度裁剪 `max_norm=1.0`。
- 学习率走 `utils.CosineLRSchedule`：warmup = 1 个 epoch 的步数，峰值 `--lr`，余弦衰减到 `1e-6`。
- 验证采样：从高斯先验反向采样后，再做一步梯度去噪修正，步长 `n_atoms * noise_std²`。
- 产出物：checkpoint 每个 epoch 覆盖保存为 `{logdir}/{dataset}_anewflow_{channels}_{blocks}_{layers}_{csp|dng}.pth`；RMS 曲线写到 `{同名}_rms.png`；采样阶段打印 `comp_valid` / `struct_valid` / `valid` 通过率以及失败原因。

冒烟测试：

```
python train_cmtflow.py --dataset perov_5 --dry_run --limit 64 --epochs 1
```

### 评测

#### CSP

```
python eval_cmtflow.py --dataset perov_5 --data ./data --logdir runs_cmtflow --outdir runs_cmtflow \
  --num_samples 0 --top_k 1 --sample_batch_size 250 \
  --transformer_channels 512 --transformer_blocks 8 --layers_per_block 4 --csp
```

`--num_samples 0` 表示全测试集，给 `N` 时会自动截断到测试集大小；每个组分采样 `--top_k` 次。

#### DNG

```
python eval_cmtflow.py --dataset perov_5 --num_samples 10000 --no-csp --outdir runs_cmtflow
```

模型参数（`--transformer_channels`、`--transformer_blocks`、`--layers_per_block`、`--nvp`）必须与 checkpoint 一致；checkpoint 默认按训练时的命名规则在 `--logdir` 下拼接，也可用 `--checkpoint` 直接指定路径。

评测流程：构造 `CrystalTensorDataset` → 从先验采样 + 一步梯度去噪 → `utils.decode_tensor` 解码成 pymatgen `Structure` 并做有效性检查 → 计算指标 → 保存采样结果。

指标口径：

- CSP —— `comp_valid` / `struct_valid` / `valid`：组分（SMACT）与结构（最小原子间距等）合法性的通过率。
- CSP —— `match_rate`：**top-k 组内命中率**，同一组分的 `top_k` 个样本只要有 1 个匹配成功即算该组命中；`sample_match_rate` 是逐样本匹配率。
- CSP —— `rms_dist`：组内命中样本的最小归一化 RMSD（`StructureMatcher.get_rms_dist`，`stol=0.5`、`angle_tol=10`、`ltol=0.3`；按 `(V/N)^(1/3)` 归一化，因 `scale=True` 不惩罚体积误差）。未匹配记为 `None`，全部失败时为 `nan`。`match_rate` 与 `rms_dist` 要成对读：前者说明能否找到，后者说明找到后有多准。
- DNG —— `wdist_density` / `wdist_num_elems`：密度、元素个数的 Wasserstein 距离。
- DNG —— `cov_*` / `amsd_*` / `amcd_*`：覆盖率与平均距离（CrystalNN 结构指纹 + Magpie 组分指纹，阈值见 `constants.COV_Cutoffs`）；缺 `matminer` 时返回 `nan`。
- DNG —— `novelty_rate` / `uniqueness_rate`：S.U.N. 口径的新颖率与唯一率。

样本张量、`at_mask`、解码后的原子序数与有效位保存到 `{outdir}/{dataset}_{csp|dng}_samples.pt`，同时打印 `sampling_time` 与 `receval_time`。

### 消融实验

`ablation_study/` 下每个目录都是自包含的实验，自带模型（`cmtflow_s*.py`）、`utils.py`、`constants.py` 与 `train_s*.py` / `eval_s*.py`。

| 编号 | 目录 | 入口 | 相对基线（仓库根目录的 28 通道 one-hot 版）的改动 |
|------|------|------|--------------------------------------------------|
| S1 | `ablation_study/s1_sinusoidal_pe/` | `train_s1.py` / `eval_s1.py` | 在相对位置偏置之外，额外加固定 sinusoidal **绝对**位置编码（随输入一起置换） |
| S2 | `ablation_study/s2_binary_atom_type/` | `train_s2.py` / `eval_s2.py` | 原子类型改为 **7 位二进制**（±1）编码，`TYPE_CHANNELS=7`、`TOTAL_CHANNELS=103` |
| S3 | `ablation_study/s3_original_atom_type/` | `train_s3.py` / `eval_s3.py` | 原子类型回到**原版 (N, N) 归一化原子序数块**，去掉周期表 row/col one-hot 归纳偏置 |
| S4 | `ablation_study/s4_no_relative_position/` | `train_s4.py` / `eval_s4.py` | 移除 `RelativePositionBias`（`rel_bias = None`） |

示例：

```
python ablation_study/s1_sinusoidal_pe/train_s1.py --dataset perov_5 --data data --logdir runs_cmtflow \
  --transformer_channels 512 --transformer_blocks 4 --layers_per_block 4 \
  --epochs 100 --batch_size 256 --num_samples 3785 --sample_batch_size 757 --csp
python ablation_study/s1_sinusoidal_pe/eval_s1.py --dataset perov_5 --logdir runs_cmtflow --num_samples 0 --top_k 1
```

消融前需要知道的几件事情：

- `--logdir` 是相对当前工作目录解析的，而消融脚本默认 `--data ../../data/`，这只在对应子目录内成立。训练与评测请保持同一个 cwd，或直接给绝对路径 / `--checkpoint <path/to/model.pth>`。
- checkpoint 前缀不同：根目录 `train_cmtflow.py` 用 `{dataset}_anewflow_{...}.pth`，`train_s*.py` 用 `{dataset}_cmtflow_{...}.pth`。
- 缓存 `*_ori.pkl` 是 124 通道；S2 / S3 会在 `utils.get_data` 里即时重编码类型块，因此可以复用共享缓存。但不要用消融目录里的 `prepare_data_tensor.py` 覆盖共享缓存。

加载前可用下面这段自检 checkpoint 属于哪套模型（看类型块宽度与 `pos_embed` / `rel_bias` 是否存在）：

```python
import torch
sd = torch.load('<your>.pth', map_location='cpu', weights_only=False)
token_num = 5                                     # perov_5；换数据集时按名字后缀改
w = sd['blocks.0.csp_embed.weight'].shape[1]
print('类型通道数 =', w // token_num,
      '| pos_embed 个数 =', sum('pos_embed' in k for k in sd),
      '| rel_bias 个数 =', sum('rel_bias' in k for k in sd))
```

`token_num=5` 时的对照表：

| 类型通道 | pos_embed | rel_bias | 对应模型 |
|----------|-----------|----------|----------|
| 28 | 8 | 48 | S1（sinusoidal 绝对位置编码） |
| 28 | 0 | 48 | 根目录基线（28 通道 one-hot + RPE） |
| 28 | 0 | 0 | S4（无相对位置偏置） |
| 7 | 0 | 48 | S2（二进制原子类型） |
| 5（= token_num） | 0 | 48 | S3（原版原子序数块） |

### 注意事项

- `prepare_data_tensor.py` 会跳过已存在的 `*_ori.pkl`，需要重建请先删除该文件。
- 该脚本里有 `if __debug__: args.num_workers = 1`，而普通 `python` 调用下 `__debug__` 为 `True`，所以 CIF 解析默认是串行的；需要并行请用 `python -O` 或删掉这一行。
- 大文件建议不入库：忽略 `*_ori.pkl`、`runs*/**/*.pth`、`*_samples.pt`，保留 `data/<dataset>/*.csv` 切分与 `figures/*.png`。

### 致谢

流模型框架参考 [TarFlow](http://arxiv.org/abs/2412.06329)。Perov-5、Carbon-24、MP-20 来自 [CDVAE](https://github.com/txie-93/cdvae)（[DiffCSP](https://github.com/jiaor17/DiffCSP) 亦沿用该套数据），MPTS-52 取自 [MP time-split 代码库](https://github.com/sparks-baird/mp-time-split)。

### 联系方式

问题与建议请提交 GitHub Issue。
