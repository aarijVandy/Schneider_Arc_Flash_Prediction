"""Train a stratified regression model with a toggle between MLP and ResNet.

Default is the original ResNet regressor; set --model mlp to mirror the
`new_improved_model.py` MLP. Stratified splits are applied to both so the class
imbalance handling stays consistent across architectures.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATHS = {
    "epri": "mv-data/epri.csv",
    "hcb": "mv-data/hcb.csv",
    "ieeeall": "mv-data/ieeeall.csv",
}

FEATURE_COLS = ["t", "D_mm", "gap_mm", "Voc", "Ibf"]
TARGET_COL = "IEmeas"


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------
def load_and_clean(paths, feature_cols, target_col):
    frames = []
    for source_name, path in paths.items():
        df = pd.read_csv(path)
        df["source"] = source_name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"Total raw samples: {len(data)}")

    data = data.dropna(subset=feature_cols + [target_col, "source"])
    print(f"After dropping NaN: {len(data)}")

    data = data[data[target_col] > 0]
    for col in feature_cols:
        data = data[data[col] > 0]
    print(f"After removing non-positive rows: {len(data)}")

    q_low = data[target_col].quantile(0.01)
    q_high = data[target_col].quantile(0.99)
    data = data[(data[target_col] >= q_low) & (data[target_col] <= q_high)]
    print(f"After target outlier trim: {len(data)}")

    return data.reset_index(drop=True)


def make_strat_bins(y, n_bins=5):
    """Create stratification bins for a continuous target using quantiles."""

    y_flat = np.asarray(y).reshape(-1)
    bins = pd.qcut(y_flat, q=n_bins, labels=False, duplicates="drop")

    if pd.Series(bins).nunique() < 2:
        bins = pd.cut(y_flat, bins=n_bins, labels=False, duplicates="drop")

    return np.asarray(bins, dtype=int)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class ArcFlashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class ResNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]

        layers = []
        prev_dim = input_dim

        # Input stem
        layers.extend([
            nn.Linear(prev_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev_dim = hidden_dims[0]

        # Residual blocks
        for h in hidden_dims:
            if prev_dim != h:
                layers.extend([
                    nn.Linear(prev_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                ])
            layers.append(ResidualBlock(h, h, dropout=dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# -----------------------------------------------------------------------------
# Training / Evaluation helpers
# -----------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            running_loss += loss.item()
    return running_loss / len(loader)


def plot_diagnostics(train_losses, val_losses, test_targets, test_preds, rmsle, r2, out_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses, label="Train MSE", alpha=0.8)
    axes[0].plot(val_losses, label="Val MSE", alpha=0.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training History")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(test_targets, test_preds, alpha=0.6, s=30)
    min_val = min(test_targets.min(), test_preds.min())
    max_val = max(test_targets.max(), test_preds.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect prediction")
    axes[1].set_xlabel("Actual Incident Energy (cal/cm²)")
    axes[1].set_ylabel("Predicted Incident Energy (cal/cm²)")
    axes[1].set_title(f"Predicted vs Actual\nTest RMSLE: {rmsle:.4f}, R²: {r2:.4f}")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    residuals = test_preds - test_targets
    axes[2].scatter(test_targets, residuals, alpha=0.6, s=30)
    axes[2].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[2].set_xlabel("Actual Incident Energy (cal/cm²)")
    axes[2].set_ylabel("Residual (Predicted - Actual)")
    axes[2].set_title("Residual Plot")
    axes[2].grid(True)

    plt.tight_layout()
    out_path = f"{out_prefix}_diagnostics.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved: {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    print("Loading and cleaning data...")
    data_clean = load_and_clean(DATA_PATHS, FEATURE_COLS, TARGET_COL)

    X = data_clean[FEATURE_COLS].values
    y = data_clean[TARGET_COL].values.reshape(-1, 1)

    print("\nTarget statistics:")
    print(f"  Min:    {y.min():.4f}")
    print(f"  Max:    {y.max():.4f}")
    print(f"  Mean:   {y.mean():.4f}")
    print(f"  Median: {np.median(y):.4f}")

    print("\nCreating target bins for stratification...")
    strat_bins = make_strat_bins(y, n_bins=5)
    print(f"Number of stratification bins used: {len(np.unique(strat_bins))}")

    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw, bins_train, bins_temp = train_test_split(
        X,
        y,
        strat_bins,
        test_size=0.30,
        random_state=42,
        stratify=strat_bins,
    )

    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw,
        y_temp_raw,
        test_size=0.50,
        random_state=42,
        stratify=bins_temp,
    )

    print("\nSplit sizes:")
    print(f"  Train: {len(X_train_raw)}")
    print(f"  Val:   {len(X_val_raw)}")
    print(f"  Test:  {len(X_test_raw)}")

    print("\nFitting scalers on train split only...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train_raw)
    X_val = scaler_X.transform(X_val_raw)
    X_test = scaler_X.transform(X_test_raw)

    y_train = scaler_y.fit_transform(y_train_raw)
    y_val = scaler_y.transform(y_val_raw)
    y_test = scaler_y.transform(y_test_raw)

    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")
    print("Saved scalers: scaler_X.pkl, scaler_y.pkl")

    train_loader = DataLoader(ArcFlashDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(ArcFlashDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(ArcFlashDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    if args.model == "mlp":
        model = MLPRegressor(input_dim=len(FEATURE_COLS), hidden_dim=args.hidden_dim, dropout=args.dropout)
        model_name = "mlp"
    else:
        model = ResNetRegressor(input_dim=len(FEATURE_COLS), dropout=args.dropout)
        model_name = "resnet"

    model = model.to(device)
    print(f"Model: {model_name} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    train_losses, val_losses = [], []

    ckpt_path = f"{model_name}_best_model.pth"

    print("\n" + "=" * 60)
    print(f"Training {model_name.upper()} Regressor...")
    print("=" * 60)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "feature_cols": FEATURE_COLS,
                    "target_col": TARGET_COL,
                },
                ckpt_path,
            )
        else:
            epochs_without_improve += 1

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{args.epochs}] | "
                f"Train MSE: {train_loss:.4f} | "
                f"Val MSE: {val_loss:.4f} | "
                f"LR: {lr:.6f}"
            )

        if epochs_without_improve >= args.patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best epoch: {best_epoch + 1}")
    print(f"Best val MSE: {best_val_loss:.4f}")
    print(f"Saved model: {ckpt_path}")

    print("\nLoading best checkpoint for evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    test_preds_scaled, test_targets_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            test_preds_scaled.append(preds.cpu().numpy())
            test_targets_scaled.append(y_batch.numpy())

    test_preds_scaled = np.vstack(test_preds_scaled)
    test_targets_scaled = np.vstack(test_targets_scaled)

    test_preds_orig = scaler_y.inverse_transform(test_preds_scaled)
    test_targets_orig = scaler_y.inverse_transform(test_targets_scaled)

    test_preds_safe = np.clip(test_preds_orig, 0, None)
    test_targets_safe = np.clip(test_targets_orig, 0, None)

    rmsle = np.sqrt(np.mean((np.log1p(test_preds_safe) - np.log1p(test_targets_safe)) ** 2))
    r2 = r2_score(test_targets_orig, test_preds_orig)
    mae = mean_absolute_error(test_targets_orig, test_preds_orig)
    rmse = np.sqrt(mean_squared_error(test_targets_orig, test_preds_orig))

    print("\n" + "=" * 60)
    print("Test Set Performance")
    print("=" * 60)
    print(f"RMSLE: {rmsle:.4f}")
    print(f"R2:    {r2:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print("=" * 60)

    plot_diagnostics(train_losses, val_losses, test_targets_orig, test_preds_orig, rmsle, r2, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified regression with MLP or ResNet.")
    parser.add_argument("--model", choices=["resnet", "mlp"], default="resnet", help="Model architecture to train (default: resnet)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for Adam")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dim for MLP (ignored for ResNet)")
    parser.add_argument("--dropout", type=float, default=0.10, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")

    main(parser.parse_args())
