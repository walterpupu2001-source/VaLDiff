#!/bin/bash
# ============================================================
# Katana 一次性环境设置
# ============================================================
# 用法（在你 SSH 进 Katana 之后，cd 到项目目录然后跑）：
#   bash scripts/katana/setup_env.sh
#
# 之后每次提交作业 / 进 interactive session 前，先：
#   source scripts/katana/activate_env.sh
# ============================================================

set -e

ENV_NAME="battery-soh"
PYTHON_VERSION="3.10"

echo "===================================================="
echo "Setting up conda environment '$ENV_NAME' on Katana"
echo "===================================================="

# 检查 Miniconda 是否安装
if [ ! -d "$HOME/miniconda3" ] && ! command -v conda &> /dev/null; then
    echo "[+] Miniconda 未安装，正在下载..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    cd -
    echo "[+] Miniconda 已安装到 $HOME/miniconda3"
fi

# 激活 conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 创建 env
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[+] Env '$ENV_NAME' already exists; skipping creation"
else
    echo "[+] Creating conda env '$ENV_NAME' (python $PYTHON_VERSION)..."
    conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION
fi

conda activate "$ENV_NAME"

# 装 PyTorch (CUDA 12.1，匹配 Katana 的 V100/A100 + 你本地 torch 2.5.1)
echo "[+] Installing PyTorch 2.5.1 with CUDA 12.1 support..."
pip install --quiet torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
echo "[+] Installing other dependencies..."
pip install --quiet \
    "numpy>=1.24,<2.0" \
    "scipy>=1.10,<2.0" \
    "matplotlib>=3.7" \
    "pyyaml>=6.0" \
    "pytest>=7.0"

# 验证
echo ""
echo "===================================================="
echo "Verification:"
echo "===================================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA build: {torch.version.cuda}')"
python -c "import numpy, scipy, matplotlib, yaml; print('numpy / scipy / matplotlib / pyyaml: OK')"
echo ""
echo "✓ Environment '$ENV_NAME' is ready."
echo ""
echo "Note: torch.cuda.is_available() will return False on the login node."
echo "      That is normal — GPU is only available inside GPU jobs."
