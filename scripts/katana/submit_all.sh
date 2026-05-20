#!/bin/bash
# ============================================================
# submit_all.sh
# One-click submit of the entire pipeline on Katana.
# Uses PBS -W depend=afterok:JOBID to enforce the right order:
#
#   00_prepare  ─┬─►  01_train_proposed     ─┐
#                ├─►  01_train_resampled_de  ┤
#                ├─►  01_train_1d_ddpm       ├─►  02_evaluate
#                ├─►  01_train_vanilla_gan   ┤
#                └─►  01_train_vanilla_vae  ─┘
#
# All 5 training jobs run in parallel after prepare; evaluate
# waits until all 5 finish successfully.
#
# Usage:
#   bash scripts/katana/submit_all.sh
# ============================================================

set -e

cd "$(dirname "$0")/../.."   # go to project root regardless of where it was called from

KATANA=scripts/katana
mkdir -p logs

echo "============================================================"
echo "Submitting full pipeline to Katana"
echo "Project root: $(pwd)"
echo "============================================================"

# 1. Prepare
JOB_PREP=$(qsub "$KATANA/00_prepare.pbs")
echo "  [submitted]  00_prepare           → $JOB_PREP"

# 2. Five trainings, all depending on prep
JOB_PROP=$(qsub -W depend=afterok:$JOB_PREP "$KATANA/01_train_proposed.pbs")
echo "  [submitted]  01_train_proposed    → $JOB_PROP    (after $JOB_PREP)"

JOB_RDE=$(qsub -W depend=afterok:$JOB_PREP "$KATANA/01_train_resampled_de.pbs")
echo "  [submitted]  01_train_resampled_de→ $JOB_RDE   (after $JOB_PREP)"

JOB_1D=$(qsub -W depend=afterok:$JOB_PREP "$KATANA/01_train_1d_ddpm.pbs")
echo "  [submitted]  01_train_1d_ddpm     → $JOB_1D    (after $JOB_PREP)"

JOB_GAN=$(qsub -W depend=afterok:$JOB_PREP "$KATANA/01_train_vanilla_gan.pbs")
echo "  [submitted]  01_train_vanilla_gan → $JOB_GAN   (after $JOB_PREP)"

JOB_VAE=$(qsub -W depend=afterok:$JOB_PREP "$KATANA/01_train_vanilla_vae.pbs")
echo "  [submitted]  01_train_vanilla_vae → $JOB_VAE   (after $JOB_PREP)"

# 3. Evaluate after all 5
DEPS="afterok:$JOB_PROP:$JOB_RDE:$JOB_1D:$JOB_GAN:$JOB_VAE"
JOB_EVAL=$(qsub -W depend=$DEPS "$KATANA/02_evaluate.pbs")
echo "  [submitted]  02_evaluate          → $JOB_EVAL  (after all 5 trains)"

echo ""
echo "============================================================"
echo "All 7 jobs submitted."
echo ""
echo "Monitor with:"
echo "  qstat -u \$USER"
echo "  watch -n 30 'qstat -u \$USER'"
echo ""
echo "Inspect logs:"
echo "  tail -f logs/01_train_proposed.log"
echo "  ls -la logs/"
echo ""
echo "When 02_evaluate finishes, compare with paper:"
echo "  python scripts/katana/compare_with_paper.py"
echo "============================================================"
