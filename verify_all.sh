#!/usr/bin/env bash
# ============================================================
# verify_all.sh
#
# Runs the full verification suite:
#   1. pytest (114 unit tests)
#   2. 5 architecture-equivalence verifiers
#
# Total runtime: ~1 minute. No data or GPU required.
# verify_step1_with_data.py auto-discovers .mat files in
# data/raw/Batch-1/ and gracefully skips if none are present,
# so this wrapper works whether or not data is set up yet.
# ============================================================

set -e
cd "$(dirname "$0")"

echo
echo "============================================================"
echo "Step 1: pytest (114 unit tests)"
echo "============================================================"
python -m pytest tests/ -v

echo
echo "============================================================"
echo "Step 2: Architecture-equivalence verifiers"
echo "============================================================"

echo
echo "--- verify_step1_with_data.py (BatteryRaw + DelayEmbedding) ---"
python scripts/verify_step1_with_data.py

echo
echo "--- verify_step3_arch.py (1D-DDPM) ---"
python scripts/verify_step3_arch.py

echo
echo "--- verify_step3b_arch.py (2D DDPM + Proposed extras) ---"
python scripts/verify_step3b_arch.py

echo
echo "--- verify_step3c_arch.py (GAN + VAE) ---"
python scripts/verify_step3c_arch.py

echo
echo "--- verify_step3d_arch.py (8 predictors + SOHScaler + classical aug) ---"
python scripts/verify_step3d_arch.py

echo
echo "============================================================"
echo "ALL VERIFICATIONS PASSED - refactored code is bit-identical"
echo "to the original."
echo "============================================================"
