# ============================================================
# Source this (do NOT execute) to activate the env.
#   source scripts/katana/activate_env.sh
# Used at the top of every PBS job script.
# ============================================================
ENV_NAME="battery-soh"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "ERROR: miniconda3 not found at \$HOME/miniconda3"
    echo "Run scripts/katana/setup_env.sh first."
    exit 1
fi

# Sanity check
python -c "import torch; print(f'[env] PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed in env '$ENV_NAME'."
    exit 1
}
