# Battery Capacity-Curve Generation with Delay-Embedding DDPM

> Official code release accompanying the paper *(title TBD)*.
> Generates full lithium-ion battery capacity-degradation trajectories from
> early-cycle observations using a conditional Delay-Embedding diffusion model,
> and uses the synthetic trajectories to augment downstream State-of-Health (SOH)
> prediction.

## Overview

The Proposed method is a **conditional 2-D diffusion model** that operates on
**Delay-Embedding (DE) images** of capacity-degradation curves. Compared to
fixed-length resampling approaches, it preserves the natural length of each
battery's trajectory through an adaptive embedding, and uses:

- **`UpBlock2D`** (bilinear resize + conv) in place of `ConvTranspose2d`,
  removing checkerboard artefacts in the generated images.
- **Masked Loss** restricted to the valid (non-padded) columns of the DE image.
- **Padding Constraint** that overwrites the padded columns with the last
  valid column after every denoising step, keeping the diffusion process from
  drifting in regions where there is no data.

The repository contains, alongside the Proposed method, four baseline
generative models and two classical augmentation techniques, plus a full
downstream evaluation pipeline with eight predictor architectures.

| Family | Methods |
|---|---|
| **Proposed** | Adaptive Delay-Embedding DDPM (UpBlock2D + Masked Loss + Padding Constraint) |
| Generative baselines | DE-DDPM (resampled DE), 1D-DDPM, Vanilla GAN, Vanilla β-VAE |
| Classical augmentations | Gaussian noise (Fan et al., 2022), time warping (Kim et al., 2020) |
| Downstream predictors | MLP, CNN, GRU, LSTM, Transformer, PatchTST, Informer, iTransformer |

### Note on classical augmentation baselines

We compare against two well-established classical augmentation methods commonly
cited in the battery SOH literature: **Gaussian Noise** (Fan et al., 2022) and
**Time Warping** (Kim et al., 2020). A third method, **amplitude scaling**, was
also implemented and observed during development but **excluded from the final
comparison tables** — with two well-cited representatives already covering the
classical augmentation category, adding a third would not change the
methodological narrative. The Scaling experiment is retained in our internal
logs as a sanity check but is not reported in the main results.

## Reproducibility

The refactored codebase is validated against the original research code through:

- **Architecture-equivalence verifiers** (`scripts/verify_*_arch.py`) — bit-identical
  model initialization and forward outputs given the same seed.
- **Unit test suite** (`tests/`) — 114 tests covering data prep, model architectures,
  augmentation, and downstream evaluation.
- **End-to-end pipeline reproduction** — see `outputs/eval_downstream/metrics.csv`
  vs `outputs/reference_metrics.csv` for side-by-side comparison with the
  paper's reported numbers.

Run the full verification suite:

```bash
bash verify_all.sh    # or verify_all.bat on Windows
pytest tests/
```

Note on bit-identical reproduction: training the generators from scratch will
not produce checkpoints bit-identical to the originals because of CUDA kernel
non-determinism (cuDNN LSTM/Conv kernels) and hardware differences between
training runs. Method rankings and downstream MAE are preserved within typical
CUDA noise (≤ 5% relative drift on most rows; see `reference_metrics.csv` for
the reference numbers).

### Pre-trained checkpoints

Pre-trained checkpoints for all five generators (Proposed, DE-DDPM, 1D-DDPM,
Vanilla GAN, Vanilla VAE) are available from the
[GitHub Releases page](../../releases) as `checkpoints_v1.0.tar.gz` (~130 MB).
Extract into the project root, then `python scripts/evaluate_downstream.py`.

## Requirements

- Python ≥ 3.9 (tested on 3.10)
- PyTorch ≥ 2.0 (CUDA recommended for diffusion training)
- A 64-bit OS (Windows / Linux / macOS)

See [`requirements.txt`](requirements.txt) for the full list. For
GPU support, install a PyTorch build matching your CUDA version from
<https://pytorch.org/get-started/locally/>.

## Installation

```bash
git clone <this-repo>
cd battery-de-ddpm

# Recommended: create a fresh environment
conda create -n battery-soh python=3.10
conda activate battery-soh

# Install dependencies (install GPU torch separately if needed)
python -m pip install -r requirements.txt
```

## Data

The experiments use the **XJTU battery dataset** (Q_nominal = 2.0 Ah).
*(Add download link here.)* After downloading, place the `.mat` files under
`data/raw/`:

```
data/raw/
├── Batch-1/   *.mat files for batch 1
├── Batch-2/
├── Batch-3/
├── Batch-4/
└── Batch-5/
```

Then run the preparation pipeline:

```bash
python scripts/prepare_data.py
# → outputs/prepared_data.pkl
```

This loads every `.mat` file, builds the 13-feature physics-condition vector,
performs a stratified train/val/test split (0.7 / 0.15 / 0.15 per batch with
seed 42), and writes a single `prepared_data.pkl` consumed by every training
and evaluation script.

## End-to-end reproduction

### One-command full pipeline

```cmd
:: Windows
run_all.bat

:: Linux / macOS
bash run_all.sh
```

This runs the full Block 1 → 2 → 3 chain. Training all five generative
models is the GPU-heavy part (each takes ≈ 1-2 h on a single modern GPU,
much less on V100 / A100 / L40S).

### Just the verification suite

```cmd
:: Windows
verify_all.bat

:: Linux / macOS
bash verify_all.sh
```

This runs only `pytest` + the architecture-equivalence verifiers (fast —
under a minute).

### Step-by-step

```bash
# 1. Prepare data (Block 1)
python scripts/prepare_data.py

# 2. Train the five generative models (Block 2)
python scripts/train_1d_ddpm.py          # outputs/ckpt/1d_ddpm/final.pt
python scripts/train_resampled_de.py     # outputs/ckpt/resampled_de/final.pt
python scripts/train_proposed.py         # outputs/ckpt/proposed/final.pt
python scripts/train_vanilla_gan.py      # outputs/ckpt/vanilla_gan/vanilla_gan.pt
python scripts/train_vanilla_vae.py      # outputs/ckpt/vanilla_vae/vanilla_vae.pt

# 3. Downstream evaluation (Block 3)
python scripts/evaluate_downstream.py
# → outputs/eval_downstream/metrics.csv     (per task × method × predictor)
# → outputs/eval_downstream/results.pkl     (last-run prediction details)
# → Console rankings by mean MAE per method, per task
```

Generated synthetic curves are cached at `outputs/curves/<method>_curves.pkl`,
so repeated evaluations skip the (expensive) diffusion-sampling phase.

## Configuration

All hyperparameters live in YAML files under [`config/`](config/) and are
loaded via [`src/utils/config.py`](src/utils/config.py).
Each script reads its own config:

| Script | Config | Purpose |
|---|---|---|
| `prepare_data.py` | `default.yaml` | Seed, splits, paths |
| `train_1d_ddpm.py` | `train_1d_ddpm.yaml` | SEQ_LEN=384, T=1000, EPOCHS=500 |
| `train_resampled_de.py` | `train_resampled_de.yaml` | IMG=32, SEQ_LEN=500, EPOCHS=400 |
| `train_proposed.py` | `train_proposed.yaml` | Same + `use_upblock`, masked loss, padding constraint |
| `train_vanilla_gan.py` | `train_vanilla_gan.yaml` | SEQ=384, LATENT=128, label smoothing 0.9/0.1 |
| `train_vanilla_vae.py` | `train_vanilla_vae.yaml` | SEQ=384, LATENT=32, KL_W=5e-4 |
| `evaluate_downstream.py` | `eval_downstream.yaml` | 3 tasks × 8 methods × 8 predictors × 5 runs |

**Do not edit hyperparameter values** if you wish to reproduce paper results
exactly — they are pinned to match the originals.

## Project structure

```
battery-de-ddpm/
├── config/                       # YAML hyperparameters (one per script)
├── data/raw/Batch-{1..5}/        # ← put .mat files here
├── outputs/                      # generated artefacts (mostly gitignored)
│   ├── prepared_data.pkl
│   ├── ckpt/{model}/             # trained model weights
│   ├── curves/                   # cached synthetic curves (per method)
│   ├── metrics/                  # CSV metrics from training scripts
│   ├── figures/{model}/          # loss curves + per-battery visualisations
│   └── eval_downstream/
│       ├── metrics.csv           # this reproduction's results
│       └── reference_metrics.csv # original paper's reference numbers
├── src/                          # importable Python package
│   ├── data/
│   │   ├── battery.py            # BatteryRaw + ensure_1d_float
│   │   ├── split.py              # gather_mat_files + stratified split
│   │   └── datasets.py           # 4 PyTorch datasets (1D/2D × cond/no-cond)
│   ├── delay_embedding.py        # DelayEmbedding (fixed + adaptive modes)
│   ├── models/
│   │   ├── common.py             # SinusoidalEmbedding, ResBlock1D/2D, UpBlock2D
│   │   ├── unet1d.py             # 1-D conditional UNet
│   │   ├── unet2d.py             # 2-D conditional UNet (toggle use_upblock)
│   │   ├── gan.py                # Generator + Discriminator (spectral norm)
│   │   └── vae.py                # Encoder + Decoder + VAE + vae_loss
│   ├── diffusion/
│   │   ├── ddpm.py               # StandardDDPM (1D + 2D, masked-loss aware)
│   │   └── proposed_constraints.py  # build_mask + enforce_padding_constraint
│   ├── evaluation/
│   │   ├── predictors.py         # 8 downstream architectures
│   │   ├── datasets.py           # SOHCurveDataset (input_ratio split)
│   │   ├── scaler.py             # SOHScaler (Min-Max with buffer)
│   │   ├── metrics.py            # MAE/RMSE/MAPE/R²/Pearson/MaxAE/MedAE
│   │   └── runner.py             # train_eval_loop with early stopping
│   ├── augmentation/
│   │   ├── classical.py          # Gaussian noise + Time warping
│   │   └── generative.py         # generate_* for each generative method
│   └── utils/
│       ├── seed.py               # set_seed (torch + numpy + random)
│       └── config.py             # YAML loader
├── scripts/                      # CLI entry points
│   ├── prepare_data.py
│   ├── compare_pkl.py            # diff two prepared_data.pkl files
│   ├── train_*.py                # 5 training scripts
│   ├── evaluate_downstream.py    # main Block-3 evaluation
│   └── verify_*_arch.py          # architecture-equivalence verifiers
├── tests/                        # 114 unit tests
├── run_all.bat / run_all.sh      # one-command reproduction
├── verify_all.bat / verify_all.sh   # one-command verification
├── requirements.txt
├── LICENSE                       # MIT
├── CITATION.cff
└── README.md
```

## Method details

### Delay-Embedding (DE)

A capacity curve `y[1:L]` is unrolled into a 2-D matrix where column *j* is a
length-*n* sliding window starting at position `(j-1)·m+1`, with *m* the hop.
Two modes are implemented in [`src/delay_embedding.py`](src/delay_embedding.py):

- **`forward_to_square`** (adaptive): picks *n* and *m* so the resulting
  matrix has at most *n* × *n* columns; zero-pads where necessary.
  *Used by the Proposed method.*
- **`forward_fixed`** (fixed length): resamples `y` to `seq_len` first, then
  computes a fixed *n* × *n* embedding.
  *Used by DE-DDPM (the resampling baseline).*

Both modes provide bit-exact inverses (`inverse_from_square`, `inverse_fixed`)
so curves recovered from DE images are identical to within the resampling
that the forward direction applies.

### DDPM

[`StandardDDPM`](src/diffusion/ddpm.py) is shape-agnostic (1-D or 2-D) and
supports two Proposed-specific extras:
- An optional **mask** for `train_step`, restricting MSE to valid columns.
- An optional **padding_fn** for `train_step` and `sample`, applied to the
  noisy `x_t` before each forward pass / after each reverse step.

### Downstream evaluation

For each of `{Baseline, Proposed, DE-DDPM, DDPM, GAN, VAE, GaussianNoise,
TimeWarping} × {SOH 30→70, SOH 40→60, SOH 50→50} × {8 predictors} × 5 seeds`,
`evaluate_downstream.py` fits a downstream model on the (real + augmented)
training curves, early-stops on validation MSE, and computes seven metrics on
test predictions inverse-transformed back to the raw SOH scale.

## Results

Generated by `python scripts/evaluate_downstream.py`.

The full per-row results live in
[`outputs/eval_downstream/metrics.csv`](outputs/eval_downstream/metrics.csv).
The original paper's reference numbers are at
[`outputs/reference_metrics.csv`](outputs/reference_metrics.csv) for direct
comparison.

*(Optional: add a summary table here once you finalise which numbers go in the paper.)*

## Citation

If you use this code, please cite:

```bibtex
@article{TBD,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {2026},
  note    = {Code available at \url{https://github.com/walterpupu2001-source/LADE-Diff}},
}
```

## Acknowledgements

This work uses the **XJTU battery dataset** *(add reference here)*.

The downstream predictors implement architectures from:
- **PatchTST** — Nie *et al.*, "A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers", ICLR 2023.
- **Informer** — Zhou *et al.*, "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021.
- **iTransformer** — Liu *et al.*, "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024.

## License

This project is released under the [MIT License](LICENSE).
