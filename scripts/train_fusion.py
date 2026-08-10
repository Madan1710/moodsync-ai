"""Train the learned-fusion MLP on synthetic data.

Loss = KL(predicted_distribution || target_distribution)
     + λ * BCE(predicted_mismatch_logit, target_mismatch)

We jointly optimise both heads — the fused distribution AND the mismatch
flag — because the fused distribution should *also* respond to mismatch
(e.g. lean more neutral / hedged when modalities disagree).

Usage:
    # Generate the dataset first:
    python -m scripts.generate_synthetic
    # Then train:
    python -m scripts.train_fusion --epochs 30 --batch-size 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from moodsync.config import PATH_CONFIG
from moodsync.models.fusion import FusionMLP


def train(
    data_path: Path,
    out_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_mismatch: float,
    val_split: float,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    blob = torch.load(data_path, map_location="cpu")
    X, y_dist, y_mis = blob["X"], blob["y_dist"], blob["y_mismatch"]
    print(f"Loaded {X.shape[0]:,} samples from {data_path}")

    dataset = TensorDataset(X, y_dist, y_mis)
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionMLP().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_val = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        total = 0.0
        n = 0
        for x, yd, ym in train_loader:
            x, yd, ym = x.to(device), yd.to(device), ym.to(device)
            out = model(x)
            fused_logits = out[:, :7]
            mismatch_logit = out[:, 7]

            log_p = F.log_softmax(fused_logits, dim=-1)
            kl = F.kl_div(log_p, yd, reduction="batchmean")
            bce = F.binary_cross_entropy_with_logits(mismatch_logit, ym)
            loss = kl + lambda_mismatch * bce

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()
        train_loss = total / n

        # ---- validate ----
        model.eval()
        v_total = 0.0
        v_correct_mis = 0
        v_n = 0
        with torch.no_grad():
            for x, yd, ym in val_loader:
                x, yd, ym = x.to(device), yd.to(device), ym.to(device)
                out = model(x)
                fused_logits = out[:, :7]
                mismatch_logit = out[:, 7]
                log_p = F.log_softmax(fused_logits, dim=-1)
                kl = F.kl_div(log_p, yd, reduction="batchmean")
                bce = F.binary_cross_entropy_with_logits(mismatch_logit, ym)
                loss = kl + lambda_mismatch * bce
                v_total += loss.item() * x.size(0)
                preds_mis = (torch.sigmoid(mismatch_logit) > 0.5).float()
                v_correct_mis += (preds_mis == ym).sum().item()
                v_n += x.size(0)
        val_loss = v_total / v_n
        val_acc_mis = v_correct_mis / v_n

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc_mis": val_acc_mis})
        print(
            f"epoch {epoch:>3d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"mismatch_acc={val_acc_mis:.3f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            print(f"  ↳ best so far — saved to {out_path}")

    print(f"\n✅ Training complete. Best val loss: {best_val:.4f}")
    print(f"   Weights: {out_path}")
    print(f"   Set MODEL_CONFIG.fusion_weights to this file (already wired).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=PATH_CONFIG.synthetic_data)
    p.add_argument("--out", type=Path, default=PATH_CONFIG.fusion_weights)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-mismatch", type=float, default=0.5)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Synthetic dataset not found at {args.data}. "
            "Run `python -m scripts.generate_synthetic` first."
        )
    train(
        data_path=args.data,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_mismatch=args.lambda_mismatch,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
