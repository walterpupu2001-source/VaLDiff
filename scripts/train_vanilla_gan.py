"""
Vanilla GAN training (Block-2 unconditional baseline).

End-to-end equivalent of the original Block-2 Vanilla-GAN script.
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
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.battery import BatteryRaw  # noqa: F401
from src.data.datasets import (
    CurveDataset1D_NoCond,
    no_cond_collate,
    interp_to_len,
)
from src.models.gan import DiscriminatorGAN, GeneratorGAN
from src.utils.config import load_config
from src.utils.seed import set_seed


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def mae_rmse(y1, y2):
    L = min(len(y1), len(y2))
    d = np.asarray(y1[:L]) - np.asarray(y2[:L])
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def main(config_path: str = "config/train_vanilla_gan.yaml"):
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
    print("BLOCK 2-Vanilla-GAN: 1D Vanilla GAN（无条件）")
    print("=" * 60)

    with open(prepared_pkl, "rb") as f:
        data = pickle.load(f)
    train_batts = data["train_batts"]
    test_batts = data["test_batts"]
    print(f"已加载: 训练={len(train_batts)}, 测试={len(test_batts)}")

    SEQ_LEN = int(cfg["dataset"]["seq_len"])
    BATCH_SZ = int(cfg["training"]["batch_size"])
    LATENT = int(cfg["model"]["latent_dim"])
    G_CH = int(cfg["model"]["g_ch"])
    D_CH = int(cfg["model"]["d_ch"])
    EPOCHS = int(cfg["training"]["epochs"])
    LR_G = float(cfg["training"]["lr_g"])
    LR_D = float(cfg["training"]["lr_d"])
    BETAS = tuple(cfg["training"]["adam_betas"])
    LBL_R = float(cfg["training"]["label_smooth_real"])
    LBL_F = float(cfg["training"]["label_smooth_fake"])
    NOISE = float(cfg["training"]["noise_std"])
    NUM = int(cfg["evaluation"]["num_samples_to_gen"])
    SAVE_EVERY = int(cfg["training"]["save_every"])

    train_ds = CurveDataset1D_NoCond(train_batts, SEQ_LEN, tag="train")
    gmin, gmax = train_ds.global_min, train_ds.global_max
    test_ds = CurveDataset1D_NoCond(test_batts, SEQ_LEN, gmin, gmax, tag="test")

    # NOTE: drop_last=True is REQUIRED because BatchNorm1d misbehaves on
    # batch size 1. Preserved from the original.
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SZ, shuffle=True,
        drop_last=True, collate_fn=no_cond_collate,
    )

    device = get_device()
    print(f"\n设备: {device}")

    G = GeneratorGAN(latent_dim=LATENT, seq_len=SEQ_LEN, ch=G_CH).to(device)
    D = DiscriminatorGAN(seq_len=SEQ_LEN, ch=D_CH).to(device)
    print(f"Generator 参数: {sum(p.numel() for p in G.parameters())/1e6:.2f}M")
    print(f"Discriminator 参数: {sum(p.numel() for p in D.parameters())/1e6:.2f}M")

    opt_G = torch.optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)
    criterion = nn.BCELoss()

    g_losses, d_losses = [], []
    start_time = time.time()

    for ep in range(EPOCHS):
        G.train(); D.train()
        g_sum, d_sum, n = 0.0, 0.0, 0

        for real, _ in train_loader:
            real = real.to(device)
            bs = real.size(0)
            real_lbl = torch.full((bs,), LBL_R, device=device)
            fake_lbl = torch.full((bs,), LBL_F, device=device)

            # ----- Train D -----
            opt_D.zero_grad()
            real_out = D(real + NOISE * torch.randn_like(real))
            z = torch.randn(bs, LATENT, device=device)
            fake = G(z).detach()
            fake_out = D(fake + NOISE * torch.randn_like(fake))
            d_loss = (criterion(real_out, real_lbl) + criterion(fake_out, fake_lbl)) / 2
            d_loss.backward()
            opt_D.step()

            # ----- Train G -----
            opt_G.zero_grad()
            z = torch.randn(bs, LATENT, device=device)
            fake_out = D(G(z))
            g_loss = criterion(fake_out, torch.ones(bs, device=device))
            g_loss.backward()
            opt_G.step()

            g_sum += g_loss.item(); d_sum += d_loss.item(); n += 1

        g_losses.append(g_sum / n); d_losses.append(d_sum / n)

        if (ep + 1) % 20 == 0:
            print(f"Epoch {ep+1:3d}/{EPOCHS} | G={g_sum/n:.4f} | D={d_sum/n:.4f}")

        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(
                {"generator": G.state_dict(), "discriminator": D.state_dict()},
                str(ckpt_dir / f"ep_{ep+1}.pt"),
            )

    print(f"训练耗时: {time.time() - start_time:.2f}s")

    torch.save(
        {"generator": G.state_dict(), "discriminator": D.state_dict()},
        str(ckpt_dir / "vanilla_gan.pt"),
    )
    print(f"✓ 保存: {ckpt_dir/'vanilla_gan.pt'}")

    # ----- Loss curves -----
    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Vanilla GAN Training Loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(str(figures_dir / "train_loss.png"), dpi=150)
    plt.close()

    # ----- Evaluation -----
    G.eval()
    all_metrics = []
    global_meta = {"gmin": gmin, "gmax": gmax, "seq_len": SEQ_LEN}

    for idx in range(len(test_ds)):
        tb = test_batts[idx]
        real = np.asarray(tb.capacity_curve)

        with torch.no_grad():
            z = torch.randn(NUM, LATENT, device=device)
            fakes = G(z).cpu().numpy()

        preds = []
        for f in fakes:
            y_den = CurveDataset1D_NoCond.denorm_curve(f, global_meta)
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
        plt.title(f"{tb.battery_name} - Vanilla GAN (Unconditional)")
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
