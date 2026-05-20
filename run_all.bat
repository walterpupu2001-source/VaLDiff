@echo off
REM ============================================================
REM run_all.bat - full reproduction pipeline
REM
REM Runs the entire experiment end-to-end:
REM   1. Block 1: prepare_data.py            (~1 min, CPU)
REM   2. Block 2: train all 5 generative     (~5-10 h, single GPU)
REM      models (1D-DDPM, Resampled-DE,
REM      Proposed, GAN, VAE)
REM   3. Block 3: evaluate_downstream.py     (~2-4 h, single GPU)
REM
REM Requires:
REM   * data\raw\Batch-{1..5}\*.mat present
REM   * conda env or venv with requirements.txt installed
REM   * A CUDA-capable GPU is strongly recommended for steps 2-3.
REM
REM To skip individual training steps (e.g. if a checkpoint is
REM already present), comment them out below.
REM ============================================================

setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo ============================================================
echo BLOCK 1: prepare data
echo ============================================================
python scripts/prepare_data.py
if errorlevel 1 ( exit /b 1 )

echo.
echo ============================================================
echo BLOCK 2: train generative models (long-running)
echo ============================================================

echo --- 1D-DDPM ---
python scripts/train_1d_ddpm.py
if errorlevel 1 ( exit /b 1 )

echo --- Resampled-DE (DE-DDPM) ---
python scripts/train_resampled_de.py
if errorlevel 1 ( exit /b 1 )

echo --- Proposed ---
python scripts/train_proposed.py
if errorlevel 1 ( exit /b 1 )

echo --- Vanilla GAN ---
python scripts/train_vanilla_gan.py
if errorlevel 1 ( exit /b 1 )

echo --- Vanilla VAE ---
python scripts/train_vanilla_vae.py
if errorlevel 1 ( exit /b 1 )

echo.
echo ============================================================
echo BLOCK 3: downstream evaluation
echo ============================================================
python scripts/evaluate_downstream.py
if errorlevel 1 ( exit /b 1 )

echo.
echo ============================================================
echo ALL DONE.
echo See outputs\eval_downstream\ for the final metrics CSV/PKL.
echo ============================================================
endlocal
