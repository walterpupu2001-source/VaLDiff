#!/usr/bin/env bash
# ============================================================
# run_all.sh — full reproduction pipeline
#
# Runs the entire experiment end-to-end:
#   1. Block 1: prepare_data.py            (~1 min, CPU)
#   2. Block 2: train all 5 generative     (~5-10 h, single GPU)
#      models (1D-DDPM, Resampled-DE,
#      Proposed, GAN, VAE)
#   3. Block 3: evaluate_downstream.py     (~2-4 h, single GPU)
#
# Requires:
#   * data/raw/Batch-{1..5}/*.mat present
#   * conda env or venv with requirements.txt installed
#   * A CUDA-capable GPU is strongly recommended for steps 2-3.
#
# To skip individual training steps (e.g. if a checkpoint is
# already present), comment them out below.
# ============================================================

set -e
cd "$(dirname "$0")"

echo
echo "============================================================"
echo "BLOCK 1: prepare data"
echo "============================================================"
python scripts/prepare_data.py

echo
echo "============================================================"
echo "BLOCK 2: train generative models (long-running)"
echo "============================================================"

echo "--- 1D-DDPM ---"
python scripts/train_1d_ddpm.py

echo "--- Resampled-DE (DE-DDPM) ---"
python scripts/train_resampled_de.py

echo "--- Proposed ---"
python scripts/train_proposed.py

echo "--- Vanilla GAN ---"
python scripts/train_vanilla_gan.py

echo "--- Vanilla VAE ---"
python scripts/train_vanilla_vae.py

echo
echo "============================================================"
echo "BLOCK 3: downstream evaluation"
echo "============================================================"
python scripts/evaluate_downstream.py

echo
echo "============================================================"
echo "ALL DONE."
echo "See outputs/eval_downstream/ for the final metrics CSV/PKL."
echo "============================================================"
