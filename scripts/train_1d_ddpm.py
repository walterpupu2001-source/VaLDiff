"""
1-D Conditional DDPM training (Block-2 baseline).

End-to-end equivalent of the original Block-2 1D-DDPM script. The
business logic — model architecture, DDPM noise schedule, training
loop, sampling routine, evaluation metrics — is preserved exactly.
Only the *organisation* changes: configuration moves to YAML, modules
move into ``src/``, and outputs (checkpoints / metrics / figures)
land in dedicated ``outputs/`` sub-directories.

Usage:
    cd <project root>
    python scripts/train_1d_ddpm.py
"""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

# Force UTF-8 stdout/stderr (Windows-friendly).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # headless, savefig only — no plt.show()
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.data.battery import BatteryRaw  # noqa: F401  needed for pickle load
from src.data.datasets import CurveDataset1D, collate_keep_meta, interp_to_len
from src.diffusion.ddpm import StandardDDPM
from src.models.unet1d import UNet1D
from src.utils.config import load_config
from src.utils.seed import set_seed


# ============================================================
# Helpers (verbatim from original Block-2 1D-DDPM)
# ============================================================
def get_device():
    """Choose CUDA if available and operational; fall back to CPU on failure."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        return "cuda"
    except RuntimeError as e:
        print(f"⚠ CUDA 不兼容，回退到 CPU: {e}")
        return "cpu"


def mae_rmse(y_true, y_pred):
    L = min(len(y_true), len(y_pred))
    diff = (y_true[:L] - y_pred[:L]).astype(np.float64)
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))


# ============================================================
# Main
# ============================================================
def main(config_path: str = "config/train_1d_ddpm.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    cwd = Path.cwd()
    prepared_pkl = cwd / cfg["paths"]["prepared_pkl"]
    ckpt_dir = cwd / cfg["paths"]["ckpt_dir"]
    metrics_csv = cwd / cfg["paths"]["metrics_csv"]
    figures_dir = cwd / cfg["paths"]["figures_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BLOCK 2: 1D Conditional DDPM (对比组，无延迟嵌入)")
    print("=" * 60)

    with open(prepared_pkl, "rb") as f:
        data = pickle.load(f)
    train_batts = data["train_batts"]
    val_batts = data["val_batts"]
    test_batts = data["test_batts"]
    cond_num_mean = data["cond_num_mean"]
    cond_num_std = data["cond_num_std"]
    print(
        f"已加载: 训练集={len(train_batts)}, 验证集={len(val_batts)}, 测试集={len(test_batts)}"
    )

    # ------------------------------------------------------------------
    # Dataset & DataLoader
    # ------------------------------------------------------------------
    SEQ_LEN = int(cfg["dataset"]["seq_len"])
    BATCH_SZ = int(cfg["training"]["batch_size"])

    train_ds = CurveDataset1D(
        train_batts, SEQ_LEN, cond_num_mean, cond_num_std, tag="train"
    )
    gmin, gmax = train_ds.global_min, train_ds.global_max
    test_ds = CurveDataset1D(
        test_batts, SEQ_LEN, cond_num_mean, cond_num_std, gmin, gmax, tag="test"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SZ,
        shuffle=True,
        collate_fn=collate_keep_meta,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Model & DDPM
    # ------------------------------------------------------------------
    device = get_device()
    model = UNet1D(
        img_ch=1,
        base=int(cfg["model"]["base_ch"]),
        cond_dim=int(cfg["model"]["cond_dim"]),
        t_dim=int(cfg["model"]["t_dim"]),
    ).to(device)

    ddpm = StandardDDPM(
        model,
        T=int(cfg["ddpm"]["t_steps"]),
        device=device,
        beta_start=float(cfg["ddpm"]["beta_start"]),
        beta_end=float(cfg["ddpm"]["beta_end"]),
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n设备: {device}")
    print(f"参数量: {num_params/1e6:.2f} M (BASE_CH={cfg['model']['base_ch']})")
    print(
        f"配置: T={cfg['ddpm']['t_steps']}, seq_len={SEQ_LEN}, "
        f"epochs={cfg['training']['epochs']}, bs={BATCH_SZ}"
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    EPOCHS = int(cfg["training"]["epochs"])
    LR = float(cfg["training"]["learning_rate"])
    SAVE_EVERY = int(cfg["training"]["save_every"])
    CLIP = float(cfg["training"]["grad_clip_norm"])

    opt = torch.optim.Adam(ddpm.model.parameters(), lr=LR)
    losses = []
    for ep in range(EPOCHS):
        ddpm.model.train()
        run = 0.0
        for xs, conds, _ in train_loader:
            xs, conds = xs.to(device), conds.to(device)
            loss = ddpm.train_step(xs, conds)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), CLIP)
            opt.step()
            run += loss.item()
        avg = run / max(1, len(train_loader))
        losses.append(avg)
        print(f"Epoch {ep+1}/{EPOCHS} | Loss={avg:.6f}")
        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(ddpm.model.state_dict(), str(ckpt_dir / f"ep_{ep+1}.pt"))
    torch.save(ddpm.model.state_dict(), str(ckpt_dir / "final.pt"))
    print("✓ 训练完成")

    # Loss curve
    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.grid(alpha=0.3)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training Loss (1D-DDPM, CH={cfg['model']['base_ch']})")
    plt.tight_layout()
    plt.savefig(str(figures_dir / "train_loss.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # Evaluation on test set
    # ------------------------------------------------------------------
    NUM_SAMPLES = int(cfg["evaluation"]["num_samples_to_gen"])
    all_metrics = []

    ddpm.model.eval()
    for idx in range(len(test_ds)):
        tb = test_batts[idx]
        x, cond, meta = test_ds[idx]
        cond_rep = cond.unsqueeze(0).repeat(NUM_SAMPLES, 1).to(device)
        seq_len = x.shape[-1]

        # Sample
        xs = ddpm.sample(cond_rep, shape=(1, seq_len)).cpu().numpy()

        real = tb.capacity_curve
        real_len = len(real)
        pred_curves = []
        for k in range(NUM_SAMPLES):
            y_den = CurveDataset1D.denorm_curve(xs[k, 0], meta)
            y_interp = interp_to_len(y_den, real_len)
            pred_curves.append(y_interp)

        print(f"\n=== {tb.battery_name} (len={real_len}) ===")
        for k, pred in enumerate(pred_curves, start=1):
            mae, rmse = mae_rmse(real, pred)
            print(f"Gen_{k} | MAE={mae:.6f} | RMSE={rmse:.6f}")
            all_metrics.append((tb.battery_name, f"Gen_{k}", mae, rmse))

        ens_mean = np.mean(pred_curves, axis=0)
        mae_m, rmse_m = mae_rmse(real, ens_mean)
        print(f"Ensemble | MAE={mae_m:.6f} | RMSE={rmse_m:.6f}")
        all_metrics.append((tb.battery_name, "Ensemble", mae_m, rmse_m))

        # Visualisation: same 1-panel layout as original
        plt.figure(figsize=(10, 5))
        plt.plot(real, "b-", lw=2.5, label="Real")
        for i, c in enumerate(pred_curves, 1):
            plt.plot(c, "--", alpha=0.7, label=f"Gen {i}")
        plt.plot(ens_mean, "r-", lw=2, label="Ensemble")
        plt.xlabel("Cycle"); plt.ylabel("Capacity (Ah)")
        plt.title(f"{tb.battery_name} - 1D DDPM")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(figures_dir / f"{tb.battery_name}.png"), dpi=300)
        plt.close()

    # Metrics CSV
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["battery", "sample", "mae", "rmse"])
        for r in all_metrics:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}"])
    print(f"✓ 完成（1D-DDPM）")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Metrics:     {metrics_csv}")
    print(f"  Figures:     {figures_dir}")


if __name__ == "__main__":
    main()
