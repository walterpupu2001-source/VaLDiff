"""
Vanilla β-VAE training (Block-2 unconditional baseline).

End-to-end equivalent of the original Block-2 Vanilla-VAE script.
"""

import csv
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from src.data.battery import BatteryRaw  # noqa: F401
from src.data.datasets import (
    CurveDataset1D_NoCond,
    no_cond_collate,
    interp_to_len,
)
from src.models.vae import VAE, vae_loss
from src.utils.config import load_config
from src.utils.seed import set_seed


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mae_rmse(y1, y2):
    L = min(len(y1), len(y2))
    d = np.asarray(y1[:L]) - np.asarray(y2[:L])
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def main(config_path: str = "config/train_vanilla_vae.yaml"):
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
    print("BLOCK 2-Vanilla-VAE: 1D Vanilla VAE（无条件）")
    print("=" * 60)

    with open(prepared_pkl, "rb") as f:
        data = pickle.load(f)
    train_batts = data["train_batts"]
    test_batts = data["test_batts"]
    print(f"已加载: 训练={len(train_batts)}, 测试={len(test_batts)}")

    SEQ_LEN = int(cfg["dataset"]["seq_len"])
    BATCH_SZ = int(cfg["training"]["batch_size"])
    LATENT = int(cfg["model"]["latent_dim"])
    HIDDEN_CH = int(cfg["model"]["hidden_ch"])
    EPOCHS = int(cfg["training"]["epochs"])
    LR = float(cfg["training"]["learning_rate"])
    KL_W = float(cfg["training"]["kl_weight"])
    PATIENCE = int(cfg["training"]["scheduler_patience"])
    FACTOR = float(cfg["training"]["scheduler_factor"])
    NUM = int(cfg["evaluation"]["num_samples_to_gen"])
    SAVE_EVERY = int(cfg["training"]["save_every"])

    train_ds = CurveDataset1D_NoCond(train_batts, SEQ_LEN, tag="train")
    gmin, gmax = train_ds.global_min, train_ds.global_max
    test_ds = CurveDataset1D_NoCond(test_batts, SEQ_LEN, gmin, gmax, tag="test")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SZ, shuffle=True,
        drop_last=False, collate_fn=no_cond_collate,
    )

    device = get_device()
    print(f"\n设备: {device}")

    model = VAE(seq_len=SEQ_LEN, latent_dim=LATENT, ch=HIDDEN_CH).to(device)
    print(f"VAE 参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"配置: latent_dim={LATENT}, kl_weight={KL_W}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=PATIENCE, factor=FACTOR,
    )

    losses, recon_losses, kl_losses = [], [], []
    start_time = time.time()

    for ep in range(EPOCHS):
        model.train()
        total_loss, total_recon, total_kl, n = 0.0, 0.0, 0.0, 0

        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, recon, kl = vae_loss(x, x_recon, mu, logvar, kl_weight=KL_W)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            n += 1

        avg_loss = total_loss / n
        avg_recon = total_recon / n
        avg_kl = total_kl / n
        losses.append(avg_loss); recon_losses.append(avg_recon); kl_losses.append(avg_kl)

        scheduler.step(avg_loss)

        if (ep + 1) % 20 == 0:
            print(
                f"Epoch {ep+1:3d}/{EPOCHS} | Loss={avg_loss:.6f} | "
                f"Recon={avg_recon:.6f} | KL={avg_kl:.6f}"
            )

        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), str(ckpt_dir / f"ep_{ep+1}.pt"))

    print(f"训练耗时: {time.time() - start_time:.2f}s")

    torch.save(model.state_dict(), str(ckpt_dir / "vanilla_vae.pt"))
    print(f"✓ 保存: {ckpt_dir/'vanilla_vae.pt'}")

    # ----- Loss curves -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(losses);       axes[0].set_title("Total Loss");        axes[0].grid(alpha=0.3)
    axes[1].plot(recon_losses); axes[1].set_title("Reconstruction Loss"); axes[1].grid(alpha=0.3)
    axes[2].plot(kl_losses);    axes[2].set_title("KL Divergence");     axes[2].grid(alpha=0.3)
    for ax in axes:
        ax.set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(str(figures_dir / "train_loss.png"), dpi=150)
    plt.close()

    # ----- Evaluation -----
    model.eval()
    all_metrics = []
    global_meta = {"gmin": gmin, "gmax": gmax, "seq_len": SEQ_LEN}

    for idx in range(len(test_ds)):
        tb = test_batts[idx]
        real = np.asarray(tb.capacity_curve)

        with torch.no_grad():
            samples = model.sample(NUM, device).cpu().numpy()

        preds = []
        for s in samples:
            y_den = CurveDataset1D_NoCond.denorm_curve(s, global_meta)
            preds.append(interp_to_len(y_den, len(real)))

        print(f"\n=== {tb.battery_name} ===")
        for k, p in enumerate(preds, 1):
            mae, rmse = mae_rmse(real, p)
            print(f"Gen_{k} | MAE={mae:.6f} | RMSE={rmse:.6f}")
            all_metrics.append((tb.battery_name, f"Gen_{k}", mae, rmse))
        ens = np.mean(preds, axis=0)
        mae, rmse = mae_rmse(real, ens)
        print(f"Ensemble | MAE={mae:.6f} | RMSE={rmse:.6f}")
        all_metrics.append((tb.battery_name, "Ensemble", mae, rmse))

        plt.figure(figsize=(10, 5))
        plt.plot(real, "b-", lw=2.5, label="Real")
        for i, c in enumerate(preds, 1):
            plt.plot(c, "--", alpha=0.7, label=f"Gen {i}")
        plt.plot(ens, "r-", lw=2, label="Ensemble")
        plt.xlabel("Cycle"); plt.ylabel("Capacity (Ah)")
        plt.title(f"{tb.battery_name} - Vanilla VAE (Unconditional)")
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(figures_dir / f"{tb.battery_name}.png"), dpi=300)
        plt.close()

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["battery_name", "sample", "mae", "rmse"])
        for r in all_metrics:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}"])
    print(f"\n✓ 指标: {metrics_csv}")


if __name__ == "__main__":
    main()
