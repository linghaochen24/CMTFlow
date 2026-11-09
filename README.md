# Crystal Structure Prediction by Transformer Autoregressive Normalizing Flow (CMTFlow)

Implementation codes for crystal structure generation with a masked Transformer normalizing flow (CMTFlow).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

CMT-Flow is a single-GPU implementation that ports the masked autoregressive Transformer flow of [TarFlow](http://arxiv.org/abs/2412.06329) to crystalline materials: a one-dimensional flow maps a crystal tensor to a Gaussian prior, `train_cmtflow.py` / `eval_cmtflow.py` handle training and evaluation, and four variants live under `ablation_study/`.

| Mode | Flag | Description | Metrics |
|------|------|-------------|---------|
| CSP (conditional generation) | `--csp` (default) | Generate lattice and fractional coordinates conditioned on the target composition (atom-type one-hot) | `match_rate`, `sample_match_rate`, normalized `rms_dist` |
| DNG (de novo generation) | `--no-csp` | Generate crystals freely, without any condition | `wdist_density`, `wdist_num_elems`, `cov_*`, `amsd_*`, `amcd_*`, `novelty_rate`, `uniqueness_rate` |

### Dependencies and Setup

```
python>=3.10
torch>=2.1          # bfloat16 autocast + GradScaler, CUDA recommended
numpy pandas scipy
pymatgen            # CIF parsing, structure matching, structural fingerprints
smact               # composition (charge-neutrality) validity
pyxtal              # structure validity
p_tqdm tqdm
matminer            # optional, required by the DNG coverage metrics
plotly kaleido mendeleev   # optional, visualization only
```

```
conda create -n cmtflow python=3.10 -y
conda activate cmtflow
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy pandas scipy pymatgen smact pyxtal p_tqdm tqdm
pip install matminer plotly kaleido mendeleev
```

All commands below are run from the repository root.

### Data Preparation

Place the CSV splits at `data/<dataset>/{train,val,test}.csv`. The `cif` column is mandatory; `material_id` and `formula` are recommended, and the property column is read when present.

| Dataset | Property column | train | val | test | max_atom_num |
|---------|-----------------|-------|-----|------|--------------|
| `perov_5` | `heat_ref` | 11356 | 3787 | 3785 | 5 |
| `mp_20` | `formation_energy_per_atom` | 27136 | 9047 | 9046 | 20 |
| `carbon_24` | `energy_per_atom` | 6091 | 2032 | 2030 | 24 |
| `mpts_52` | `formation_energy_per_atom` | 27380 | 5000 | 8096 | 52 |

Convert each split into a cached tensor file `data/<dataset>/{train,val,test}_ori.pkl` (`--num_workers` controls the CIF-parsing parallelism):

```
python prepare_data_tensor.py --data_path data --dataset mp_20 --num_workers 8
python prepare_data_tensor.py --data_path data --dataset perov_5 --num_workers 8
python prepare_data_tensor.py --data_path data --dataset carbon_24 --num_workers 8
python prepare_data_tensor.py --data_path data --dataset mpts_52 --num_workers 8
```

### Crystal Tensor Representation

Each crystal is encoded by `utils.create_tensor_from_structure` into a `(channels, max_atom_num)` tensor:

| Slice | Channels | Content |
|-------|----------|---------|
| Atom type | 28 | one-hot of periodic-table row (13) + column (15), scaled to ±1 |
| Coordinates | 90 | fractional coordinates rounded to 3 decimals, digit-wise one-hot, scaled to ±1 |
| Lattice length | 3 | normalized to [-1, 1] by `LATTICE_LENGTH_INTERVAL[dataset]` |
| Lattice angle | 3 | angle / 180 |
| Total | 124 | under `--csp` the model only generates the last 96 channels and takes the type block as `cond` |

`max_atom_num` is derived from the dataset name suffix, and `at_mask` marks the real atoms so that padded positions are excluded from the loss.

### Training

```
python train_cmtflow.py --dataset perov_5 --data data --logdir runs_cmtflow \
  --transformer_channels 512 --transformer_blocks 8 --layers_per_block 4 \
  --batch_size 256 --epochs 100 --lr 1e-4 --noise_std 0.10 \
  --sample_freq 10 --num_samples 3785 --sample_batch_size 757 --csp
```

Main hyper-parameters:

| Group | Argument | Default | Description |
|-------|----------|---------|-------------|
| Data | `--data` / `--dataset` | `data` / `perov_5` | data root and dataset name (sets `max_atom_num` and lattice normalization) |
| | `--logdir` | `runs_cmtflow` | checkpoint and curve output directory |
| | `--limit` | `0` | use only the first N records (0 = all), for debugging |
| Model | `--transformer_channels` | `512` | model width |
| | `--transformer_blocks` | `8` | number of flow blocks |
| | `--layers_per_block` | `4` | depth per flow block |
| | `--nvp` / `--no-nvp` | `True` | use NVP coupling |
| | `--noise_std` | `0.10` | standard deviation of input additive Gaussian noise |
| Optimization | `--batch_size` / `--epochs` / `--lr` | `256` / `100` / `1e-4` | AdamW with cosine decay down to `1e-6` |
| Sampling | `--sample_freq` | `10` | validate-sample every N epochs |
| | `--num_samples` / `--sample_batch_size` | `3785` / `757` | paired validation records; `num_samples` must not exceed the val split |
| | `--csp` / `--no-csp` | `True` | CSP conditional generation vs. DNG de novo generation |
| | `--cfg` | `0` | guidance weight (0 = no guidance) |
| Other | `--update_prior` | `False` | update the prior variance `var` online (used when `--no-nvp`) |
| | `--dry_run` | `False` | smoke test on a single batch |

Implementation notes:

- Objective: `loss = 0.5 * mean(z²) - mean(logdet)`, where both terms are averaged over real atoms only; with `--update_prior` the prior is updated online as `var.lerp_(z², 0.1)`.
- Mixed precision `torch.autocast(bfloat16)` with `GradScaler` and gradient clipping at `max_norm=1.0`.
- Learning rate follows `utils.CosineLRSchedule`: one epoch of warmup, peak `--lr`, cosine decay to `1e-6`.
- Validation sampling reverse-samples from the Gaussian prior and then applies one gradient-denoising step with step size `n_atoms * noise_std²`.
- Outputs: the checkpoint is overwritten every epoch as `{logdir}/{dataset}_anewflow_{channels}_{blocks}_{layers}_{csp|dng}.pth`, the RMS curve is written to `{same_name}_rms.png`, and the sampling stage prints the `comp_valid` / `struct_valid` / `valid` pass rates together with the failure reasons.

Smoke test:

```
python train_cmtflow.py --dataset perov_5 --dry_run --limit 64 --epochs 1
```

### Evaluation

#### Crystal structure prediction (CSP)

```
python eval_cmtflow.py --dataset perov_5 --data ./data --logdir runs_cmtflow --outdir runs_cmtflow \
  --num_samples 0 --top_k 1 --sample_batch_size 250 \
  --transformer_channels 512 --transformer_blocks 8 --layers_per_block 4 --csp
```

`--num_samples 0` evaluates the full test split and `--num_samples N` is truncated to the split size. Each composition is sampled `--top_k` times.

#### De novo generation (DNG)

```
python eval_cmtflow.py --dataset perov_5 --num_samples 10000 --no-csp --outdir runs_cmtflow
```

The model arguments (`--transformer_channels`, `--transformer_blocks`, `--layers_per_block`, `--nvp`) must match the checkpoint, which is resolved from `--logdir` by the training naming rule unless `--checkpoint` is given explicitly.

Evaluation pipeline: build `CrystalTensorDataset` → sample from the prior plus one gradient-denoising step → `utils.decode_tensor` into pymatgen `Structure` objects with validity checks → compute metrics → dump the samples.

Metric definitions:

- CSP — `comp_valid` / `struct_valid` / `valid`: pass rates of composition (SMACT) and structure (minimum interatomic distance) validity.
- CSP — `match_rate`: hit rate **within a top-k group**, a group counts as a hit if any of its `top_k` samples matches the ground truth; `sample_match_rate` is the per-sample rate.
- CSP — `rms_dist`: the smallest normalized RMSD among the hits of a group (`StructureMatcher.get_rms_dist`, `stol=0.5`, `angle_tol=10`, `ltol=0.3`; normalized by `(V/N)^(1/3)` because `scale=True` does not penalize volume error). Unmatched entries are `None`, and the value is `nan` when everything fails. Read `match_rate` and `rms_dist` as a pair: the first says whether a match was found, the second how accurate it is.
- DNG — `wdist_density` / `wdist_num_elems`: Wasserstein distances of density and element count.
- DNG — `cov_*` / `amsd_*` / `amcd_*`: coverage and average distances (CrystalNN structure fingerprints + Magpie composition fingerprints, cutoffs in `constants.COV_Cutoffs`); these return `nan` when `matminer` is missing.
- DNG — `novelty_rate` / `uniqueness_rate`: S.U.N. novelty and uniqueness.

Samples (tensors, `at_mask`, decoded atomic numbers and validity flags) are saved to `{outdir}/{dataset}_{csp|dng}_samples.pt`, and `sampling_time` / `receval_time` are reported.

### Ablation Studies

Each directory under `ablation_study/` is self-contained, shipping its own model (`cmtflow_s*.py`), `utils.py`, `constants.py` and `train_s*.py` / `eval_s*.py`.

| ID | Directory | Entry point | Change w.r.t. the baseline (28-channel one-hot at the repository root) |
|----|-----------|-------------|-----------------------------------------------------------------------|
| S1 | `ablation_study/s1_sinusoidal_pe/` | `train_s1.py` / `eval_s1.py` | add a fixed sinusoidal **absolute** positional encoding on top of the relative-position bias (permuted together with the input) |
| S2 | `ablation_study/s2_binary_atom_type/` | `train_s2.py` / `eval_s2.py` | atom type encoded as **7-bit binary** (±1), `TYPE_CHANNELS=7`, `TOTAL_CHANNELS=103` |
| S3 | `ablation_study/s3_original_atom_type/` | `train_s3.py` / `eval_s3.py` | atom type reverted to the **original (N, N) normalized atomic-number block**, dropping the row/column one-hot inductive bias |
| S4 | `ablation_study/s4_no_relative_position/` | `train_s4.py` / `eval_s4.py` | remove `RelativePositionBias` (`rel_bias = None`) |

Example:

```
python ablation_study/s1_sinusoidal_pe/train_s1.py --dataset perov_5 --data data --logdir runs_cmtflow \
  --transformer_channels 512 --transformer_blocks 4 --layers_per_block 4 \
  --epochs 100 --batch_size 256 --num_samples 3785 --sample_batch_size 757 --csp
python ablation_study/s1_sinusoidal_pe/eval_s1.py --dataset perov_5 --logdir runs_cmtflow --num_samples 0 --top_k 1
```

Things worth knowing before running the ablations:

- `--logdir` is resolved against the current working directory, and the ablation scripts default to `--data ../../data/`, which only holds inside the corresponding subdirectory. Either keep the same cwd for training and evaluation, or pass absolute paths / `--checkpoint <path/to/model.pth>`.
- Checkpoint prefixes differ: the root `train_cmtflow.py` writes `{dataset}_anewflow_{...}.pth` while `train_s*.py` writes `{dataset}_cmtflow_{...}.pth`.
- The cached `*_ori.pkl` files store 124 channels; S2 and S3 re-encode the type block on the fly (`utils.get_data`), so the shared cache can be reused. Do not run the `prepare_data_tensor.py` of an ablation directory over the shared cache.

To identify which model a checkpoint belongs to, inspect the type-block width and the presence of `pos_embed` / `rel_bias` before loading:

```python
import torch
sd = torch.load('<your>.pth', map_location='cpu', weights_only=False)
token_num = 5                                     # perov_5; change with the dataset suffix
w = sd['blocks.0.csp_embed.weight'].shape[1]
print('type channels =', w // token_num,
      '| pos_embed =', sum('pos_embed' in k for k in sd),
      '| rel_bias =', sum('rel_bias' in k for k in sd))
```

With `token_num=5`:

| Type channels | pos_embed | rel_bias | Model |
|---------------|-----------|----------|-------|
| 28 | 8 | 48 | S1 (sinusoidal absolute PE) |
| 28 | 0 | 48 | root baseline (28-channel one-hot + RPE) |
| 28 | 0 | 0 | S4 (no relative-position bias) |
| 7 | 0 | 48 | S2 (binary atom type) |
| 5 (`= token_num`) | 0 | 48 | S3 (original atomic-number block) |

### Notes

- `prepare_data_tensor.py` skips splits whose `*_ori.pkl` already exists; delete the file to rebuild.
- That script contains `if __debug__: args.num_workers = 1`, and `__debug__` is `True` for a normal `python` invocation, so CIF parsing is serial unless you run `python -O` or remove the line.
- To cache the large artifacts: ignore `*_ori.pkl`, `runs*/**/*.pth` and `*_samples.pt`, and keep the `data/<dataset>/*.csv` splits and `figures/*.png`.

### Acknowledgments

The flow framework follows [TarFlow](http://arxiv.org/abs/2412.06329). Perov-5, Carbon-24 and MP-20 come from [CDVAE](https://github.com/txie-93/cdvae) (as used in [DiffCSP](https://github.com/jiaor17/DiffCSP)), and MPTS-52 is collected from the [MP time-split codebase](https://github.com/sparks-baird/mp-time-split).

### Contact

Please open an issue on GitHub for questions and bug reports.
