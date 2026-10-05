import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
 
ORGAN_COLS = ["spleen_mm", "kidney_right_mm", "kidney_left_mm", "gallbladder_mm", "liver_mm"]
 
 
def smooth(values, weight=0.6):
    """Exponential moving average smoothing."""
    smoothed, last = [], values.iloc[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed
 
 
def plot_loss(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.plot(df["epoch"], df["train_loss"], color="#b0b8c8", linewidth=1, alpha=0.5, label="_raw_train")
    ax.plot(df["epoch"], df["val_loss"],   color="#f0a080", linewidth=1, alpha=0.5, label="_raw_val")
    ax.plot(df["epoch"], smooth(df["train_loss"]), color="#4a7fc1", linewidth=2.2, label="Train Loss")
    ax.plot(df["epoch"], smooth(df["val_loss"]),   color="#e05c20", linewidth=2.2, label="Val Loss")
 
    best_epoch = df.loc[df["val_loss"].idxmin(), "epoch"]
    best_loss  = df["val_loss"].min()
    ax.axvline(best_epoch, color="#e05c20", linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate(f"best val\nepoch {best_epoch}", xy=(best_epoch, best_loss),
                xytext=(best_epoch + max(1, len(df)*0.03), best_loss),
                fontsize=8, color="#e05c20",
                arrowprops=dict(arrowstyle="->", color="#e05c20", lw=1))
 
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
 
    path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)
 
 
def plot_avg_error(df: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.plot(df["epoch"], df["avg_error_mm"], color="#b8c8b0", linewidth=1, alpha=0.5, label="_raw")
    ax.plot(df["epoch"], smooth(df["avg_error_mm"]), color="#2e8b57", linewidth=2.5,
            label="Avg Centroid Error")
 
    best_epoch = df.loc[df["avg_error_mm"].idxmin(), "epoch"]
    best_err   = df["avg_error_mm"].min()
    ax.axvline(best_epoch, color="#2e8b57", linestyle="--", linewidth=1, alpha=0.6)
    ax.annotate(f"{best_err:.2f} mm\nepoch {best_epoch}", xy=(best_epoch, best_err),
                xytext=(best_epoch + max(1, len(df)*0.03), best_err),
                fontsize=8, color="#2e8b57",
                arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1))
 
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Error (mm)", fontsize=11)
    ax.set_title("Average Centroid Localization Error", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
 
    path = os.path.join(out_dir, "avg_error_curve.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)
 
 
def plot_per_organ(df: pd.DataFrame, out_dir: str):
    available = [c for c in ORGAN_COLS if c in df.columns]
    if not available:
        print("No per-organ columns found in CSV, skipping.")
        return
 
    colors = ["#e05c20", "#4a7fc1", "#2e8b57", "#c77dff", "#f4a261"]
    fig, ax = plt.subplots(figsize=(10, 6))
 
    for col, color in zip(available, colors):
        label = col.replace("_mm", "").replace("_", " ").title()
        ax.plot(df["epoch"], df[col], color=color, linewidth=1, alpha=0.3, label="_raw")
        ax.plot(df["epoch"], smooth(df[col]), color=color, linewidth=2, label=label)
 
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Error (mm)", fontsize=11)
    ax.set_title("Per-Organ Centroid Error", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
 
    path = os.path.join(out_dir, "per_organ_error.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)
 
 
def plot_lr(df: pd.DataFrame, out_dir: str):
    if "lr" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["lr"].astype(float), color="#888", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Learning Rate", fontsize=11)
    ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
 
    path = os.path.join(out_dir, "lr_schedule.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close(fig)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/logs/metrics.csv")
    parser.add_argument("--out", default="outputs/plots")
    args = parser.parse_args()
 
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} epochs from {args.csv}")
 
    plot_loss(df, args.out)
    plot_avg_error(df, args.out)
    plot_per_organ(df, args.out)
    plot_lr(df, args.out)
    print(f"\nAll plots saved to: {args.out}/")
 
 
if __name__ == "__main__":
    main()