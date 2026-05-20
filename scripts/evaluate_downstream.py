"""
Block 3 downstream evaluation — main entry point.

Mirrors the original Block 3 driver in:
- Loading prepared_data.pkl
- Converting to SOH at the right moment (only AT TRAINING-DATA ASSEMBLY,
  the generated curves stay on capacity scale until that moment)
- Augmented training data = train_curves (SOH) + capacity_list_to_soh(syn)
- Fitting SOHScaler on TRAIN SOH curves only
- Looping over 3 tasks × 9 methods × 8 predictors × NUM_RUNS seeds
- Using set_seed(BASE_SEED + run*100) before each predictor instantiation
- Saving the LAST-RUN prediction details for each (task, method, model)
- Producing the same CSV column set as Block 3
"""

import csv
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from src.augmentation.classical import (
    aug_gaussian_noise,
    aug_time_warping,
)
from src.augmentation.generative import (
    generate_de_ddpm,
    generate_ddpm_1d,
    generate_gan,
    generate_resampled_de,
    generate_vae,
)
from src.data.battery import BatteryRaw  # noqa: F401 — needed for unpickling
from src.evaluation.datasets import SOHCurveDataset
from src.evaluation.predictors import PREDICTOR_REGISTRY
from src.evaluation.runner import train_eval_loop
from src.evaluation.scaler import (
    SOHScaler,
    capacity_list_to_soh,
    capacity_to_soh_curve,
    ensure_1d_float,
)
from src.utils.config import load_config
from src.utils.seed import set_seed


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_path(out_dir, filename):
    return os.path.join(out_dir, filename)


def main(config_path: str = "config/eval_downstream.yaml"):
    cfg = load_config(config_path)
    BASE_SEED = int(cfg["seed"])
    set_seed(BASE_SEED)

    cwd = Path.cwd()
    prepared_pkl = cwd / cfg["paths"]["prepared_pkl"]
    cache_dir = cwd / cfg["paths"]["curves_cache_dir"]
    results_dir = cwd / cfg["paths"]["results_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BLOCK 3: SOH Trajectory Prediction with Generative Augmentation")
    print("=" * 70)

    # ----- Load prepared data -----
    with open(prepared_pkl, "rb") as f:
        data = pickle.load(f)
    train_batts = data["train_batts"]
    val_batts = data["val_batts"]
    test_batts = data["test_batts"]
    cond_num_mean = data["cond_num_mean"]
    cond_num_std = data["cond_num_std"]

    # Capacity (Ah) curves — used for generation and gmin/gmax computation.
    train_capacity_curves = [ensure_1d_float(b.capacity_curve) for b in train_batts]
    val_capacity_curves = [ensure_1d_float(b.capacity_curve) for b in val_batts]
    test_capacity_curves = [ensure_1d_float(b.capacity_curve) for b in test_batts]

    # SOH curves — used for downstream prediction.
    Q_NOM = float(cfg["soh"]["q_nominal_ah"])
    train_curves = capacity_list_to_soh(train_capacity_curves, q_nominal=Q_NOM)
    val_curves = capacity_list_to_soh(val_capacity_curves, q_nominal=Q_NOM)
    test_curves = capacity_list_to_soh(test_capacity_curves, q_nominal=Q_NOM)

    print(f"Train: {len(train_batts)}, Val: {len(val_batts)}, Test: {len(test_batts)}")
    print(f"SOH definition: SOH(k) = Q(k) / {Q_NOM:.1f}Ah")

    # ----- Fit SOHScaler on train SOH only -----
    SOH_SCALER = SOHScaler(buffer=float(cfg["soh"]["buffer"]))
    SOH_SCALER.fit(train_curves)
    print(f"SOHScaler fitted on train: "
          f"raw=[{SOH_SCALER.raw_min:.6f}, {SOH_SCALER.raw_max:.6f}], "
          f"buffered=[{SOH_SCALER.min_val:.6f}, {SOH_SCALER.max_val:.6f}]")

    device = get_device()
    print(f"Device: {device}")

    # ============================================================
    # Phase 1: Generate augmented data (with caching)
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 1: Data Generation")
    print("=" * 70)

    NUM_AUG = int(cfg["augmentation"]["num_aug_per_battery"])
    IMG_SIZE = int(cfg["geometry"]["img_size_2d"])
    SEQ_1D = int(cfg["geometry"]["seq_len_1d"])
    SEQ_RES = int(cfg["geometry"]["seq_len_resampled"])
    NUM_TOTAL_SYN = len(train_batts) * NUM_AUG

    def save_cached_curves(curves, name):
        path = cache_dir / f"{name}_curves.pkl"
        with open(path, "wb") as f:
            pickle.dump(curves, f)
        print(f"    -> cached to {path}")

    def load_cached_curves(name, alt_names=None):
        """Load cached synthetic curves.

        Tries multiple filenames and handles BOTH the Block-3 ``curve_cache``
        format (flat list of arrays) AND the Block-3 ``generated_curves``
        viz format (dict with 'synthetic_curves' key).
        """
        candidates = [name] + (alt_names or [])
        for n in candidates:
            path = cache_dir / f"{n}_curves.pkl"
            if not path.exists():
                continue
            with open(path, "rb") as f:
                obj = pickle.load(f)
            # Flat list of arrays?
            if isinstance(obj, list):
                curves = obj
            # Viz-format dict with 'synthetic_curves' key?
            elif isinstance(obj, dict) and "synthetic_curves" in obj:
                curves = obj["synthetic_curves"]
            else:
                print(f"  [cache] {n}: file exists but has unrecognised format, skipping")
                continue
            if curves is None or len(curves) == 0:
                print(f"  [cache] {n}: file exists but contains 0 curves, skipping")
                continue
            print(f"  [cache] {name}: loaded {len(curves)} curves from {path}")
            return list(curves)
        return None

    # ``augmented_data[method]`` holds the FULL list of SOH curves used as
    # training data for that method (real + synthetic, the latter converted
    # from capacity to SOH at the moment of appending — verbatim with Block 3).
    augmented_data = {"Baseline": train_curves}
    ckpt_paths = cfg["paths"]["ckpt_paths"]

    # Proposed (UpBlock2D + DE)
    if Path(ckpt_paths["Proposed"]).exists():
        syn = load_cached_curves("Proposed")
        if syn is None:
            syn = generate_de_ddpm(
                ckpt_paths["Proposed"], train_batts,
                cond_num_mean, cond_num_std,
                use_upblock=True, num_aug_per_battery=NUM_AUG,
                img_size=IMG_SIZE, device=device,
            )
            save_cached_curves(syn, "Proposed")
        augmented_data["Proposed"] = train_curves + capacity_list_to_soh(syn, q_nominal=Q_NOM)
    else:
        print(f"  [skip] Proposed: ckpt not found ({ckpt_paths['Proposed']})")

    # DE-DDPM (Resampled-DE checkpoint, ConvTranspose2d)
    if Path(ckpt_paths["DE-DDPM"]).exists():
        # Block 3 saved this under two different filenames:
        #   curve_cache:        Resampled-DE_curves.pkl   (flat list)
        #   generated_curves:   DE-DDPM_curves.pkl         (viz dict)
        syn = load_cached_curves("Resampled-DE", alt_names=["DE-DDPM"])
        if syn is None:
            syn = generate_resampled_de(
                ckpt_paths["DE-DDPM"], train_batts,
                cond_num_mean, cond_num_std,
                num_aug_per_battery=NUM_AUG,
                img_size=IMG_SIZE, seq_len_resampled=SEQ_RES, device=device,
            )
            save_cached_curves(syn, "Resampled-DE")
        augmented_data["DE-DDPM"] = train_curves + capacity_list_to_soh(syn, q_nominal=Q_NOM)
    else:
        print(f"  [skip] DE-DDPM: ckpt not found ({ckpt_paths['DE-DDPM']})")

    # 1D-DDPM
    if Path(ckpt_paths["DDPM"]).exists():
        syn = load_cached_curves("DDPM")
        if syn is None:
            syn = generate_ddpm_1d(
                ckpt_paths["DDPM"], train_batts,
                cond_num_mean, cond_num_std,
                num_aug_per_battery=NUM_AUG,
                seq_len=SEQ_1D, device=device,
            )
            save_cached_curves(syn, "DDPM")
        augmented_data["DDPM"] = train_curves + capacity_list_to_soh(syn, q_nominal=Q_NOM)
    else:
        print(f"  [skip] DDPM: ckpt not found ({ckpt_paths['DDPM']})")

    # GAN (unconditional)
    if Path(ckpt_paths["GAN"]).exists():
        syn = load_cached_curves("GAN")
        if syn is None:
            syn = generate_gan(
                ckpt_paths["GAN"], NUM_TOTAL_SYN, train_capacity_curves,
                seq_len=SEQ_1D, device=device,
            )
            save_cached_curves(syn, "GAN")
        augmented_data["GAN"] = train_curves + capacity_list_to_soh(syn, q_nominal=Q_NOM)
    else:
        print(f"  [skip] GAN: ckpt not found ({ckpt_paths['GAN']})")

    # VAE (unconditional)
    if Path(ckpt_paths["VAE"]).exists():
        syn = load_cached_curves("VAE")
        if syn is None:
            syn = generate_vae(
                ckpt_paths["VAE"], NUM_TOTAL_SYN, train_capacity_curves,
                seq_len=SEQ_1D, device=device,
            )
            save_cached_curves(syn, "VAE")
        augmented_data["VAE"] = train_curves + capacity_list_to_soh(syn, q_nominal=Q_NOM)
    else:
        print(f"  [skip] VAE: ckpt not found ({ckpt_paths['VAE']})")

    # Classical augmentations (computed on capacity curves, then converted to SOH).
    print("\n[Classical augmentation methods]")
    syn_gauss = aug_gaussian_noise(train_capacity_curves, num_aug_per_battery=NUM_AUG, base_seed=BASE_SEED)
    augmented_data["GaussianNoise"] = train_curves + capacity_list_to_soh(syn_gauss, q_nominal=Q_NOM)
    print(f"  GaussianNoise: {len(syn_gauss)} curves")

    syn_warp = aug_time_warping(train_capacity_curves, num_aug_per_battery=NUM_AUG, base_seed=BASE_SEED)
    augmented_data["TimeWarping"] = train_curves + capacity_list_to_soh(syn_warp, q_nominal=Q_NOM)
    print(f"  TimeWarping:   {len(syn_warp)} curves")

    print(f"\n  Total methods ready: {len(augmented_data)}")

    # ============================================================
    # Phase 2: Downstream evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("Phase 2: Downstream SOH Trajectory Evaluation")
    print("=" * 70)

    NUM_RUNS = int(cfg["num_runs"])
    DS_EPOCHS = int(cfg["downstream"]["epochs"])
    DS_BS = int(cfg["downstream"]["batch_size"])
    DS_LR = float(cfg["downstream"]["learning_rate"])
    METHODS = list(cfg["method_order"])
    PREDICTORS = list(cfg["predictors"])
    TASKS = cfg["tasks"]

    # Storage for last-run prediction details (Block 3 saves them).
    preds_dir = results_dir / "predictions_soh"
    preds_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for task_name, task_cfg in TASKS.items():
        in_dim = int(task_cfg["input_size"])
        out_dim = int(task_cfg["output_size"])
        fixed_len = int(task_cfg["fixed_len"])
        input_ratio = float(task_cfg["input_ratio"])

        print(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")

        # Val / test datasets shared across methods.
        val_ds = SOHCurveDataset(val_curves, None, None,
                                 fixed_len=fixed_len, input_ratio=input_ratio, scaler=SOH_SCALER)
        test_ds = SOHCurveDataset(test_curves, None, None,
                                  fixed_len=fixed_len, input_ratio=input_ratio, scaler=SOH_SCALER)
        val_loader = DataLoader(val_ds, batch_size=DS_BS)
        test_loader = DataLoader(test_ds, batch_size=DS_BS)

        for method in METHODS:
            if method not in augmented_data:
                continue
            curves = augmented_data[method]
            train_ds = SOHCurveDataset(curves, None, None,
                                       fixed_len=fixed_len, input_ratio=input_ratio, scaler=SOH_SCALER)
            train_loader = DataLoader(train_ds, batch_size=DS_BS, shuffle=True)

            print(f"\n  {method} (train samples: {len(train_ds)})")

            for predictor_name in PREDICTORS:
                Predictor = PREDICTOR_REGISTRY[predictor_name]
                run_metrics = {k: [] for k in
                               ['MAE', 'RMSE', 'MAPE', 'R2', 'Pearson', 'MaxAE', 'MedAE']}

                for run in range(NUM_RUNS):
                    set_seed(BASE_SEED + run * 100)
                    model = Predictor(in_dim, out_dim)
                    last_run = run == NUM_RUNS - 1

                    if last_run:
                        m, details = train_eval_loop(
                            model, train_loader, val_loader, test_loader,
                            epochs=DS_EPOCHS, lr=DS_LR,
                            return_details=True, scaler=SOH_SCALER, device=device,
                        )
                        details["task"] = task_name
                        details["method"] = method
                        details["model"] = predictor_name
                        details_path = preds_dir / f"{task_name}_{method}_{predictor_name}_predictions.pkl"
                        with open(details_path, "wb") as f:
                            pickle.dump(details, f)
                    else:
                        m = train_eval_loop(
                            model, train_loader, val_loader, test_loader,
                            epochs=DS_EPOCHS, lr=DS_LR,
                            return_details=False, scaler=SOH_SCALER, device=device,
                        )
                    for k, v in m.items():
                        run_metrics[k].append(v)

                avg_metrics = {k: float(np.mean(v)) for k, v in run_metrics.items()}
                std_metrics = {k: float(np.std(v)) for k, v in run_metrics.items()}

                print(f"    {predictor_name:<12} MAE: {avg_metrics['MAE']:.4f}±{std_metrics['MAE']:.4f} | "
                      f"RMSE: {avg_metrics['RMSE']:.4f}±{std_metrics['RMSE']:.4f} | "
                      f"R²: {avg_metrics['R2']:.4f}±{std_metrics['R2']:.4f}")

                results.append({
                    "Task": task_name, "Method": method, "Model": predictor_name,
                    "MAE_mean": avg_metrics["MAE"], "MAE_std": std_metrics["MAE"],
                    "RMSE_mean": avg_metrics["RMSE"], "RMSE_std": std_metrics["RMSE"],
                    "MAPE_mean": avg_metrics["MAPE"], "MAPE_std": std_metrics["MAPE"],
                    "R2_mean": avg_metrics["R2"], "R2_std": std_metrics["R2"],
                    "Pearson_mean": avg_metrics["Pearson"], "Pearson_std": std_metrics["Pearson"],
                    "MaxAE_mean": avg_metrics["MaxAE"], "MaxAE_std": std_metrics["MaxAE"],
                    "MedAE_mean": avg_metrics["MedAE"], "MedAE_std": std_metrics["MedAE"],
                })

    # ----- Save CSV & PKL -----
    csv_path = results_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow(r)
    pkl_path = results_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"results": results}, f)

    print(f"\nResults CSV: {csv_path}")
    print(f"Results PKL: {pkl_path}")

    # ----- Print ranking summary (Block 3 style) -----
    print("\n" + "=" * 70 + "\nRANKING SUMMARY\n" + "=" * 70)
    for task_name in TASKS:
        print(f"\n--- {task_name} ---")
        for predictor_name in PREDICTORS:
            task_model = [r for r in results if r["Task"] == task_name and r["Model"] == predictor_name]
            task_model.sort(key=lambda x: x["MAE_mean"])
            print(f"  {predictor_name}:")
            for i, r in enumerate(task_model[:3]):
                mark = "1st" if i == 0 else ("2nd" if i == 1 else "3rd")
                star = " *" if r["Method"] == "Proposed" else ""
                print(f"    {mark}. {r['Method']}{star} (MAE={r['MAE_mean']:.4f})")


if __name__ == "__main__":
    main()
