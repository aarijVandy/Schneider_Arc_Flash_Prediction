import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

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
data_paths = {
    "epri": "mv-data/epri.csv",
    "hcb": "mv-data/hcb.csv",
    "ieeeall": "mv-data/ieeeall.csv",
}

feature_cols = ["t", "D_mm", "gap_mm", "Voc", "Ibf"]
target_col = "IEmeas"

batch_size = 32
num_epochs = 200
learning_rate = 1e-3
weight_decay = 1e-5
hidden_dim = 64
dropout = 0.10
patience = 20  # early stopping patience

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

    # Drop missing values
    data = data.dropna(subset=feature_cols + [target_col, "source"])
    print(f"After dropping NaN: {len(data)}")

    # Remove non-positive values
    data = data[data[target_col] > 0]
    for col in feature_cols:
        data = data[data[col] > 0]
    print(f"After removing non-positive rows: {len(data)}")

    # Trim extreme target outliers
    q_low = data[target_col].quantile(0.01)
    q_high = data[target_col].quantile(0.99)
    data = data[(data[target_col] >= q_low) & (data[target_col] <= q_high)]
    print(f"After target outlier trim: {len(data)}")

    return data.reset_index(drop=True)


def make_strat_bins(y, n_bins=5):
    """
    Create stratification bins for a continuous regression target.
    Uses quantile bins when possible.
    """
    y_flat = np.asarray(y).reshape(-1)

    # duplicates='drop' avoids failures when repeated target values exist
    bins = pd.qcut(y_flat, q=n_bins, labels=False, duplicates="drop")

    # If too few unique bins are created, fall back to equal-width bins
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
# Model
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


# -----------------------------------------------------------------------------
# Load + clean
# -----------------------------------------------------------------------------
print("Loading and cleaning data...")
data_clean = load_and_clean(data_paths, feature_cols, target_col)

X = data_clean[feature_cols].values
y = data_clean[target_col].values.reshape(-1, 1)

print("\nTarget statistics:")
print(f"  Min:    {y.min():.4f}")
print(f"  Max:    {y.max():.4f}")
print(f"  Mean:   {y.mean():.4f}")
print(f"  Median: {np.median(y):.4f}")

# -----------------------------------------------------------------------------
# Stratified split for regression
# -----------------------------------------------------------------------------
print("\nCreating target bins for stratification...")
strat_bins = make_strat_bins(y, n_bins=5)
print(f"Number of stratification bins used: {len(np.unique(strat_bins))}")

# First split: train vs temp
X_train_raw, X_temp_raw, y_train_raw, y_temp_raw, bins_train, bins_temp = train_test_split(
    X,
    y,
    strat_bins,
    test_size=0.30,
    random_state=42,
    stratify=strat_bins,
)

# Second split: val vs test
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

# -----------------------------------------------------------------------------
# Scale using TRAIN ONLY
# -----------------------------------------------------------------------------
print("\nFitting scalers on train split only...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_val = scaler_X.transform(X_val_raw)
X_test = scaler_X.transform(X_test_raw)

y_train = scaler_y.fit_transform(y_train_raw)
y_val = scaler_y.transform(y_val_raw)
y_test = scaler_y.transform(y_test_raw)

print(f"Scaled train feature shape: {X_train.shape}")
print(f"Scaled train target shape:  {y_train.shape}")

# Save scalers
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("Saved scalers: scaler_X.pkl, scaler_y.pkl")

# -----------------------------------------------------------------------------
# Dataloaders
# -----------------------------------------------------------------------------
train_dataset = ArcFlashDataset(X_train, y_train)
val_dataset = ArcFlashDataset(X_val, y_val)
test_dataset = ArcFlashDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

model = MLPRegressor(
    input_dim=len(feature_cols),
    hidden_dim=hidden_dim,
    dropout=dropout,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

best_val_loss = float("inf")
best_epoch = -1
epochs_without_improvement = 0

train_losses = []
val_losses = []

print("\n" + "=" * 60)
print("Training MLP Regressor...")
print("=" * 60)

for epoch in range(num_epochs):
    # -------------------------
    # Train
    # -------------------------
    model.train()
    running_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    # -------------------------
    # Validate
    # -------------------------
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            running_val_loss += loss.item()

    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    # Save best checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "feature_cols": feature_cols,
                "target_col": target_col,
            },
            "mlp_best_model.pth",
        )
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {val_loss:.4f} | "
            f"LR: {lr:.6f}"
        )

    # Early stopping
    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
        break

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
print(f"Best epoch: {best_epoch + 1}")
print(f"Best val MSE: {best_val_loss:.4f}")
print("Saved model: mlp_best_model.pth")

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
print("\nLoading best checkpoint for evaluation...")
checkpoint = torch.load("mlp_best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
test_preds_scaled = []
test_targets_scaled = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        preds = model(X_batch)

        test_preds_scaled.append(preds.cpu().numpy())
        test_targets_scaled.append(y_batch.numpy())

test_preds_scaled = np.vstack(test_preds_scaled)
test_targets_scaled = np.vstack(test_targets_scaled)

# Inverse-transform back to original IE units
test_preds_orig = scaler_y.inverse_transform(test_preds_scaled)
test_targets_orig = scaler_y.inverse_transform(test_targets_scaled)

# Clip only for RMSLE safety
test_preds_safe = np.clip(test_preds_orig, 0, None)
test_targets_safe = np.clip(test_targets_orig, 0, None)

rmsle = np.sqrt(
    np.mean((np.log1p(test_preds_safe) - np.log1p(test_targets_safe)) ** 2)
)
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

# -----------------------------------------------------------------------------
# Diagnostics by target regime
# -----------------------------------------------------------------------------
high_threshold = np.quantile(test_targets_orig, 0.90)

high_mask = test_targets_orig.flatten() >= high_threshold
low_mask = ~high_mask

if high_mask.sum() > 0:
    high_r2 = r2_score(test_targets_orig[high_mask], test_preds_orig[high_mask])
    high_mae = mean_absolute_error(test_targets_orig[high_mask], test_preds_orig[high_mask])
    print("\nHigh-IE subset diagnostics (top 10% of TEST targets):")
    print(f"  Threshold: {high_threshold:.4f}")
    print(f"  Count:     {int(high_mask.sum())}")
    print(f"  High R2:   {high_r2:.4f}")
    print(f"  High MAE:  {high_mae:.4f}")

if low_mask.sum() > 0:
    low_r2 = r2_score(test_targets_orig[low_mask], test_preds_orig[low_mask])
    low_mae = mean_absolute_error(test_targets_orig[low_mask], test_preds_orig[low_mask])
    print("\nLow/medium-IE subset diagnostics (bottom 90% of TEST targets):")
    print(f"  Count:     {int(low_mask.sum())}")
    print(f"  Low R2:    {low_r2:.4f}")
    print(f"  Low MAE:   {low_mae:.4f}")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
print("\nGenerating plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training history
axes[0].plot(train_losses, label="Train MSE", alpha=0.8)
axes[0].plot(val_losses, label="Val MSE", alpha=0.8)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training History (MLP)")
axes[0].legend()
axes[0].grid(True)

# Predicted vs actual (log scale)
axes[1].scatter(test_targets_orig, test_preds_orig, alpha=0.6, s=30)
min_val = min(test_targets_orig.min(), test_preds_orig.min())
max_val = max(test_targets_orig.max(), test_preds_orig.max())
axes[1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
axes[1].set_xlabel("Actual IE (cal/cm²)")
axes[1].set_ylabel("Predicted IE (cal/cm²)")
axes[1].set_title(f"Pred vs Actual (log)\nRMSLE={rmsle:.4f}, R²={r2:.4f}")
axes[1].legend()
axes[1].grid(True)
axes[1].set_xscale("log")
axes[1].set_yscale("log")

# Residual plot
residuals = test_preds_orig - test_targets_orig
axes[2].scatter(test_targets_orig, residuals, alpha=0.6, s=30)
axes[2].axhline(y=0, color="r", linestyle="--", lw=2)
axes[2].set_xlabel("Actual IE (cal/cm²)")
axes[2].set_ylabel("Residual (Pred - Actual)")
axes[2].set_title("Residual Plot")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("mlp_predictions.png", dpi=300, bbox_inches="tight")
print("Saved plot: mlp_predictions.png")

# Linear scatter
fig2, ax = plt.subplots(figsize=(8, 8))
ax.scatter(
    test_targets_orig,
    test_preds_orig,
    alpha=0.6,
    s=30,
    c="blue",
    edgecolors="k",
    linewidth=0.5,
)
ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect")
ax.set_xlabel("Actual IE (cal/cm²)", fontsize=12)
ax.set_ylabel("Predicted IE (cal/cm²)", fontsize=12)
ax.set_title(
    f"MLP: Pred vs Actual (linear)\nRMSLE={rmsle:.4f}, R²={r2:.4f}",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlp_predictions_linear.png", dpi=300, bbox_inches="tight")
print("Saved plot: mlp_predictions_linear.png")

plt.show()

print("\nArtifacts:")
print("  - mlp_best_model.pth")
print("  - scaler_X.pkl")
print("  - scaler_y.pkl")
print("  - mlp_predictions.png")
print("  - mlp_predictions_linear.png")