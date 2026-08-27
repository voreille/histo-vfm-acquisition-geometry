# Histo VFM Acquisition Geometry

Code for the paper **"Post-hoc Shrinkage of Acquisition-Induced Variability in Histopathology Vision Foundation Model Embeddings"**.

We characterize and mitigate scanner- and stain-induced variability in histopathology vision foundation model (VFM) embeddings using post-hoc geometric correction on frozen embeddings.

The main workflow is:
1. Prepare the SCORPION dataset (`prepare-scorpion`)
2. Run a cross-validated erasure grid (`run-experiment`) with `multi_deltas_*` configs
3. Fit the selected eraser on all data (`fit-chained-eraser`)

---
> [!NOTE]
> **Pre-release status.** This repository currently contains a pre-release implementation. Final reproducibility validation against the experiments reported in the paper is ongoing. Minor changes to the code, configurations, and documentation may occur before the validated `v1.0` release.

## Installation

### 1. Create a conda environment

```bash
conda create -n histovfmgeom python=3.12 pip -y
conda activate histovfmgeom
```

### 2. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch

Install PyTorch 2.13.0 and TorchVision 0.28.0 with the CUDA 12.6 runtime:

```bash
python -m pip install \
  torch==2.13.0 \
  torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

The CUDA runtime bundled with PyTorch does not have to match the version shown by `nvidia-smi`. The NVIDIA driver must support the selected runtime.

### 4. Install this package

```bash
pip install -e ".[histoaug]"
```

- `histoaug` pulls in `tiatoolbox` + `h5py` (stain simulation).

---

## Data

### Download and prepare SCORPION

Download from [Zenodo 16517924](https://zenodo.org/records/16517924), extract, tile, and write metadata in one command:

```bash
prepare-scorpion --download
```

This creates:
- `data/raw/SCORPION_dataset/` — raw images
- `data/processed/SCORPION_tiles_224px_0p5mpp/` — 224 px tiles at 0.5 mpp
- `data/processed/SCORPION_tiles_224px_0p5mpp/metadata.csv`

If you already have the raw data extracted:

```bash
prepare-scorpion --raw-dir /path/to/SCORPION_dataset
```

---

## Usage

### 1. Run the cross-validated erasure grid

```bash
run-experiment --config configs/experiments/scorpion/multi_deltas_grid_soft_h0mini.yaml
```

Key configs under `configs/experiments/scorpion/`:

| Config | Model | Eraser |
|---|---|---|
| `multi_deltas_grid_pca_h0mini.yaml` | H-Optimus-0-mini | PCA |
| `multi_deltas_grid_pca_hoptimus.yaml` | H-Optimus-1 | PCA |
| `multi_deltas_grid_soft_h0mini.yaml` | H-Optimus-0-mini | Soft |
| `multi_deltas_grid_soft_hoptimus1.yaml` | H-Optimus-1 | Soft |

Add `--dry-run` to validate the config without running. Add `--run-only-one-fold` for a quick smoke test.

### 2. Fit the final eraser on all data

After selecting the best hyperparameters from the grid, fit on all SCORPION data:

```bash
fit-chained-eraser configs/fitting/chained_soft_h0mini_sweep.yaml
```

Fitting configs are under `configs/fitting/`.

---

## Project structure

```
configs/
  experiments/scorpion/   # multi_deltas_* CV configs
  fitting/                # full-data fitting configs
histovfmgeom/
  cli/                    # entry points: prepare_scorpion, run_experiment, fit_chained_eraser_cli
  concept_erasure/        # eraser classes and fitter (multi_paired_delta_erasers, leace, ...)
  data/                   # embedding loading and tile dataset
  deltas/                 # delta construction (scanner, stain, domain)
  evaluation/             # probe and erasure metrics
  experiments/            # experiment runner (sequential_delta_grid)
  models/                 # VFM encoder wrapper
  projections/            # linear projection utilities
```

---

## Acknowledgements

### Third-party code

The covariance-shrinkage implementation is adapted from
[EleutherAI/concept-erasure](https://github.com/EleutherAI/concept-erasure/blob/main/concept_erasure/shrinkage.py),
released under the MIT License. The original copyright and license notice
are provided in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

### Data

This work uses the [SCORPION dataset](https://doi.org/10.5281/zenodo.16517924),
introduced by Ryu et al. in
[*SCORPION: Addressing Scanner-Induced Variability in Histopathology*](https://arxiv.org/abs/2507.20907).
We thank the authors for making this scanner-paired histopathology dataset publicly available.

---

## Citation

If you use this code, please cite:

```
[BibTeX entry]
```

