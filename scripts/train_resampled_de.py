"""
Resampled-DE / DE-DDPM training (Block-2 comparison method).

End-to-end equivalent of the original Block-2 Resampled-DE script.
Uses fixed-mode delay embedding (resample → 32×32 DE) and standard
DDPM (no masked loss, no padding constraint).
"""

import csv
import pickle
import sys
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

from src.data.battery import BatteryRaw  # noqa: F401  needed for pickle
from src.data.datasets import (
    CurveDataset2D_ResampledDE,
    collate_keep_meta,
    interp_to_len,
)
from src.delay_embedding import DelayEmbedding
from src.diffusion.ddpm import StandardDDPM
from src.models.unet2d import UNet2D
from src.utils.config import load_config
from src.utils.seed import set_seed


def get_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except RuntimeError as e:
        print(f"⚠ CUDA 不兼容，回退到 CPU: {e}")
        return "cpu"


def mae_rmse(y_true, y_pred):
    L = min(len(y_true), len(y_pred))
    diff = (y_true[:L] - y_pred[:L]).astype(np.float64)
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.mean(diff ** 2)))


def main(config_path: str = "config/train_resampled_de.yaml"):
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
    print("BLOCK 2: Resampled-DE (★ 重采样到固定长度)")
    print("=" * 60)

    with open(prepared_pkl, "rb") as f:
        data = pickle.load(f)
    train_batts = data["train_batts"]
    test_batts = data["test_batts"]
    cond_num_mean = data["cond_num_mean"]
    cond_num_std = data["cond_num_std"]
    print(f"已加载: 训练={len(train_batts)}, 测试={len(test_batts)}")

    IMG_SIZE = int(cfg["dataset"]["img_size"])
    SEQ_LEN = int(cfg["dataset"]["seq_len"])
    BATCH_SZ = int(cfg["training"]["batch_size"])

    train_ds = CurveDataset2D_ResampledDE(
        train_batts, IMG_SIZE, SEQ_LEN, cond_num_mean, cond_num_std, tag="train"
    )
    gmin, gmax = train_ds.global_min, train_ds.global_max
    test_ds = CurveDataset2D_ResampledDE(
        test_batts, IMG_SIZE, SEQ_LEN, cond_num_mean, cond_num_std,
        global_min=gmin, global_max=gmax, tag="test",
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SZ, shuffle=True,
        collate_fn=collate_keep_meta, num_workers=0,
    )

    device = get_device()
    model = UNet2D(
        img_ch=1,
        base=int(cfg["model"]["base_ch"]),
        cond_dim=int(cfg["model"]["cond_dim"]),
        t_dim=int(cfg["model"]["t_dim"]),
        use_upblock=bool(cfg["model"]["use_upblock"]),
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
    print(f"参数量: {num_params/1e6:.2f} M | 条件维度: {train_ds.cond_dim}")
    print(f"★ 重采样长度: SEQ_LEN={SEQ_LEN}")

    # ----------------- Training -----------------
    EPOCHS = int(cfg["training"]["epochs"])
    LR = float(cfg["training"]["learning_rate"])
    CLIP = float(cfg["training"]["grad_clip_norm"])
    SAVE_EVERY = int(cfg["training"]["save_every"])

    opt = torch.optim.Adam(ddpm.model.parameters(), lr=LR)
    losses = []
    for ep in range(EPOCHS):
        ddpm.model.train()
        run = 0.0
        for xs, conds, _ in train_loader:
            xs, conds = xs.to(device), conds.to(device)
            loss = ddpm.train_step(xs, conds)  # no mask / no padding_fn
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), CLIP)
            opt.step(); run += loss.item()
        avg = run / max(1, len(train_loader))
        losses.append(avg)
        print(f"Epoch {ep+1}/{EPOCHS} | Loss={avg:.6f}")
        if (ep + 1) % SAVE_EVERY == 0:
            torch.save(ddpm.model.state_dict(), str(ckpt_dir / f"ep_{ep+1}.pt"))
    torch.save(ddpm.model.state_dict(), str(ckpt_dir / "final.pt"))
    print("✓ 训练完成")

    plt.figure(figsize=(7, 4))
    plt.plot(losses); plt.grid(alpha=0.3)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training Loss (Resampled-DE, SEQ_LEN={SEQ_LEN}, CH={cfg['model']['base_ch']})")
    plt.tight_layout(); plt.savefig(str(figures_dir / "train_loss.png"), dpi=150); plt.close()

    # ----------------- Evaluation -----------------
    NUM = int(cfg["evaluation"]["num_samples_to_gen"])
    all_metrics = []
    ddpm.model.eval()

    for idx in range(len(test_ds)):
        tb = test_batts[idx]
        _, cond, meta = test_ds[idx]
        cond_rep = cond.unsqueeze(0).repeat(NUM, 1).to(device)
        real = test_batts[idx].capacity_curve
        real_len = len(real)

        imgs = ddpm.sample(cond_rep, shape=(1, IMG_SIZE, IMG_SIZE)).cpu().numpy()

        pred_curves = []
        for k in range(NUM):
            img_denorm = CurveDataset2D_ResampledDE.denorm_image(imgs[k, 0], meta)
            curve_seq = CurveDataset2D_ResampledDE.image_to_curve(img_denorm, meta)
            pred_curves.append(interp_to_len(curve_seq, real_len))

        print(f"\n=== {tb.battery_name} (orig_len={real_len}, SEQ_LEN={SEQ_LEN}) ===")
        for k, pred in enumerate(pred_curves, start=1):
            mae, rmse = mae_rmse(real, pred)
            print(f"Gen_{k} | MAE={mae:.6f} | RMSE={rmse:.6f}")
            all_metrics.append((tb.battery_name, f"Gen_{k}", mae, rmse))
        ens_mean = np.mean(pred_curves, axis=0)
        mae_m, rmse_m = mae_rmse(real, ens_mean)
        print(f"Ensemble | MAE={mae_m:.6f} | RMSE={rmse_m:.6f}")
        all_metrics.append((tb.battery_name, "Ensemble", mae_m, rmse_m))

        # 3-panel visualisation matching the original
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ax = axes[0]
        ax.plot(real, "b-", lw=2.5, label="Real")
        for i, c in enumerate(pred_curves, 1):
            ax.plot(c, "--", alpha=0.7, label=f"Gen {i}")
        ax.plot(ens_mean, "r-", lw=2, label="Ensemble")
        ax.set_xlabel("Cycle"); ax.set_ylabel("Capacity (Ah)")
        ax.set_title(f"{tb.battery_name} (Resampled-DE)")
        ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1]
        real_resampled = interp_to_len(real, SEQ_LEN)
        img_real, _ = DelayEmbedding.forward_fixed(real_resampled, IMG_SIZE, SEQ_LEN)
        im = ax.imshow(img_real, aspect='auto', cmap='viridis')
        ax.set_title(f'Real DE (resampled to {SEQ_LEN})'); plt.colorbar(im, ax=ax)

        ax = axes[2]
        raw_img_gen = CurveDataset2D_ResampledDE.denorm_image(imgs[0, 0], meta)
        im = ax.imshow(raw_img_gen, aspect='auto', cmap='viridis')
        ax.set_title('Gen DE (raw)'); plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.savefig(str(figures_dir / f"{tb.battery_name}.png"), dpi=300)
        plt.close()

    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["battery", "sample", "mae", "rmse"])
        for r in all_metrics:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}"])
    print(f"✓ 完成（Resampled-DE，SEQ_LEN={SEQ_LEN}）")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Metrics:     {metrics_csv}")
    print(f"  Figures:     {figures_dir}")


if __name__ == "__main__":
    main()
