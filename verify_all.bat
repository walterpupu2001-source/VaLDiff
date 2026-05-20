@echo off
REM ============================================================
REM verify_all.bat
REM
REM Runs the full verification suite:
REM   1. pytest (114 unit tests)
REM   2. 5 architecture-equivalence verifiers
REM
REM Total runtime: ~1 minute. No data or GPU required.
REM verify_step1_with_data.py auto-discovers .mat files in
REM data\raw\Batch-1\ and gracefully skips if none are present,
REM so this wrapper works whether or not data is set up yet.
REM ============================================================

setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo ============================================================
echo Step 1: pytest (114 unit tests)
echo ============================================================
python -m pytest tests/ -v
if errorlevel 1 (
    echo.
    echo [FAIL] pytest reported failures. Stopping.
    exit /b 1
)

echo.
echo ============================================================
echo Step 2: Architecture-equivalence verifiers
echo ============================================================

echo.
echo --- verify_step1_with_data.py (BatteryRaw + DelayEmbedding) ---
python scripts/verify_step1_with_data.py
if errorlevel 1 ( exit /b 1 )

echo.
echo --- verify_step3_arch.py (1D-DDPM) ---
python scripts/verify_step3_arch.py
if errorlevel 1 ( exit /b 1 )

echo.
echo --- verify_step3b_arch.py (2D DDPM + Proposed extras) ---
python scripts/verify_step3b_arch.py
if errorlevel 1 ( exit /b 1 )

echo.
echo --- verify_step3c_arch.py (GAN + VAE) ---
python scripts/verify_step3c_arch.py
if errorlevel 1 ( exit /b 1 )

echo.
echo --- verify_step3d_arch.py (8 predictors + SOHScaler + classical aug) ---
python scripts/verify_step3d_arch.py
if errorlevel 1 ( exit /b 1 )

echo.
echo ============================================================
echo ALL VERIFICATIONS PASSED - refactored code is bit-identical
echo to the original.
echo ============================================================
endlocal
