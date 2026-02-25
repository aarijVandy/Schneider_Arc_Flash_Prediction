# dashboard.py  (Demo Mode UI)
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Arc Flash IE Predictor", layout="wide")

REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_DIR / "mv-data"

# Select relevant features for prediction
# Ones we use: Iameas (arc current), t (time), D_mm (distance), gap_mm, Voc
FEATURE_COLS = ["Iameas", "t", "D_mm", "gap_mm", "Voc", "Ibf"]
TARGET_COL = "IEmeas" # Incident energy measured

DEFAULT_HIDDEN_DIMS = [128, 256, 256, 128]
DEFAULT_DROPOUT = 0.30

# -----------------------------
# Model definition (ResNet Block)
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity # Residual connection
        out = self.relu(out)
        return out

# ResNet Regressor
class ResNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, dropout=0.2):
        super().__init__()
        hidden_dims = hidden_dims or DEFAULT_HIDDEN_DIMS

        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        prev_dim = hidden_dims[0]
        for h in hidden_dims:
            layers.append(ResidualBlock(prev_dim, h, dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Custom Dataset
class ArcFlashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def mse_loss(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)


# -----------------------------
# Data functions
# -----------------------------
def load_mv_data(use_combined=True):
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"mv-data folder not found at: {DATA_DIR}")

    def read_if_exists(name):
        p = DATA_DIR / name
        return pd.read_csv(p) if p.exists() else None

    ieeeall = read_if_exists("ieeeall.csv")
    if ieeeall is None:
        raise FileNotFoundError("mv-data/ieeeall.csv not found")

    if not use_combined:
        return ieeeall

    epri = read_if_exists("epri.csv")
    hcb = read_if_exists("hcb.csv")
    dfs = [df for df in [epri, hcb, ieeeall] if df is not None]
    return pd.concat(dfs, ignore_index=True)


def clean_data(df: pd.DataFrame):
    required = FEATURE_COLS + [TARGET_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    d = df.dropna(subset=required).copy()

    d = d[d[TARGET_COL] > 0]
    for c in FEATURE_COLS:
        d = d[d[c] > 0]

    # Remove outliers (helps with learning)
    q1 = d[TARGET_COL].quantile(0.01)
    q3 = d[TARGET_COL].quantile(0.99)
    d = d[(d[TARGET_COL] >= q1) & (d[TARGET_COL] <= q3)]
    return d


@st.cache_data
def cached_load_and_clean(use_combined: bool):
    raw = load_mv_data(use_combined=use_combined)
    clean = clean_data(raw)
    return raw, clean


def split_and_scale(df: pd.DataFrame, test_frac=0.15, val_frac=0.15, seed=SEED):
    X = df[FEATURE_COLS].astype(float).values
    y = df[TARGET_COL].astype(float).values.reshape(-1, 1)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed
    )
    val_size_adjusted = val_frac / (1 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size_adjusted, random_state=seed
    )

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_s = scaler_X.transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    return (X_train_s, y_train_s), (X_val_s, y_val_s), (X_test_s, y_test_s), scaler_X, scaler_y


def r2_score_np(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def metrics_np(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    y_true_c = np.clip(y_true, 0, None)
    y_pred_c = np.clip(y_pred, 0, None)
    mask = (y_true_c > 0) & (y_pred_c > 0)
    if np.any(mask):
        rmsle = float(np.sqrt(np.mean((np.log1p(y_pred_c[mask]) - np.log1p(y_true_c[mask])) ** 2)))
    else:
        rmsle = float("nan")

    r2 = r2_score_np(y_true, y_pred)
    return mae, rmse, rmsle, r2


def validate_inputs(inp: dict):
    bad = []
    for k, v in inp.items():
        try:
            vv = float(v)
        except Exception:
            bad.append(k)
            continue
        if (not np.isfinite(vv)) or vv <= 0:
            bad.append(k)
    return bad


# -----------------------------
# Artifacts
# -----------------------------
def artifact_paths(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "model": out_dir / "resnet_best_model.pth",
        "sx": out_dir / "scaler_X.pkl",
        "sy": out_dir / "scaler_y.pkl",
    }


def try_load_artifacts(search_dirs):
    for d in search_dirs:
        d = Path(d)
        model_p = d / "resnet_best_model.pth"
        sx_p = d / "scaler_X.pkl"
        sy_p = d / "scaler_y.pkl"
        if model_p.exists() and sx_p.exists() and sy_p.exists():
            return model_p, sx_p, sy_p
    return None, None, None


def load_model_and_scalers(model_path: Path, sx_path: Path, sy_path: Path, device):
    scaler_X = joblib.load(sx_path)
    scaler_y = joblib.load(sy_path)

    ckpt = torch.load(model_path, map_location=device)

    # support both: checkpoint dict OR raw state_dict
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    arch = ckpt.get("arch", {}) if isinstance(ckpt, dict) else {}
    input_dim = int(arch.get("input_dim", len(FEATURE_COLS)))
    hidden_dims = arch.get("hidden_dims", DEFAULT_HIDDEN_DIMS)
    dropout = float(arch.get("dropout", DEFAULT_DROPOUT))

    model = ResNetRegressor(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    meta = {
        "epoch": int(ckpt.get("epoch", 0)) + 1 if isinstance(ckpt, dict) else None,
        "val_loss": float(ckpt.get("val_loss", float("nan"))) if isinstance(ckpt, dict) else float("nan"),
        "paths": {"model": str(model_path), "sx": str(sx_path), "sy": str(sy_path)},
        "device": str(device),
        "arch": {"input_dim": input_dim, "hidden_dims": hidden_dims, "dropout": dropout},
        "schema": ckpt.get("schema", {}) if isinstance(ckpt, dict) else {},
    }
    return model, scaler_X, scaler_y, meta


def predict_with_artifacts(model, scaler_X, scaler_y, df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS].astype(float).values
    Xs = scaler_X.transform(X)

    dev = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
        yp_s = model(xt).cpu().numpy()

    yp = scaler_y.inverse_transform(yp_s).reshape(-1)
    yp = np.clip(yp, 0, None)  # physical clamp

    out = df.copy()
    out["pred_IEmeas"] = yp
    return out

def enable_dropout_only(model: nn.Module):
    """
    Turn on Dropout layers ONLY (train mode), keep everything else (esp BatchNorm) in eval mode.
    This allows MC dropout with batch size 1.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

def predict_mc_dropout(model, scaler_X, scaler_y, df: pd.DataFrame, n=30):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS].astype(float).values
    Xs = scaler_X.transform(X)

    dev = next(model.parameters()).device
    xt = torch.tensor(Xs, dtype=torch.float32, device=dev)

    # ✅ keep BN in eval; enable only dropout
    enable_dropout_only(model)

    preds = []
    with torch.no_grad():
        for _ in range(int(n)):
            yp_s = model(xt).cpu().numpy()
            yp = scaler_y.inverse_transform(yp_s).reshape(-1)
            preds.append(np.clip(yp, 0, None))

    # restore full eval mode
    model.eval()

    preds = np.vstack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

def plot_training_target_hist(clean_df: pd.DataFrame, pred_now: float):
    y = clean_df[TARGET_COL].astype(float).values
    y = y[np.isfinite(y)]
    y = y[y > 0]

    p01, p50, p99 = np.percentile(y, [1, 50, 99])

    fig, ax = plt.subplots(figsize=(6.2, 2.6))

    # log bins for skewed data
    lo = max(y.min(), 1e-6)
    hi = y.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 35)

    ax.hist(y, bins=bins, alpha=0.45, edgecolor="black", linewidth=0.4)

    ax.set_xscale("log")

    ax.axvline(p01, linestyle="dotted", linewidth=2, label="p01")
    ax.axvline(p50, linestyle="--", linewidth=2, label="median")
    ax.axvline(p99, linestyle="dotted", linewidth=2, label="p99")
    ax.axvline(max(pred_now, 1e-6), linewidth=3, label="prediction")

    ax.set_title("Training IEmeas distribution (log scale)")
    ax.set_xlabel("IEmeas (cal/cm²)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="upper right")

    return fig, (p01, p50, p99)

# -----------------------------
# Streamlit state
# -----------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler_X = None
    st.session_state.scaler_y = None
    st.session_state.meta = None

if "train_history" not in st.session_state:
    st.session_state.train_history = {"train": [], "val": []}

if "live_running" not in st.session_state:
    st.session_state.live_running = False
    st.session_state.live_idx = 0
    st.session_state.live_log = []


# -----------------------------
# Demo Mode UI helpers
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_loaded():
    if st.session_state.model is not None:
        return True

    search_dirs = [REPO_DIR, REPO_DIR / "Models"]
    found_model, found_sx, found_sy = try_load_artifacts(search_dirs)
    if not found_model:
        return False

    model, sx, sy, meta = load_model_and_scalers(found_model, found_sx, found_sy, device)
    st.session_state.model = model
    st.session_state.scaler_X = sx
    st.session_state.scaler_y = sy
    st.session_state.meta = meta
    return True


@st.cache_data
def training_bounds(use_combined: bool):
    _, clean_df = cached_load_and_clean(use_combined)
    bounds = {}
    for c in FEATURE_COLS:
        lo = float(clean_df[c].quantile(0.01))
        hi = float(clean_df[c].quantile(0.99))
        bounds[c] = (lo, hi)
    # target distribution for a quick reference plot
    t_lo = float(clean_df[TARGET_COL].quantile(0.01))
    t_med = float(clean_df[TARGET_COL].quantile(0.50))
    t_hi = float(clean_df[TARGET_COL].quantile(0.99))
    return bounds, (t_lo, t_med, t_hi)


# -----------------------------
# PAGE: Header
# -----------------------------
st.title("Arc Flash Incident Energy Predictor")

loaded = ensure_loaded()
if loaded:
    meta = st.session_state.meta or {}
    msg = "Model loaded ✅"
    if meta.get("epoch") is not None:
        msg += f" (best epoch: {meta['epoch']})"
    st.success(msg, icon="✅")
else:
    st.error("No saved model artifacts found. Train a model in Advanced → Training.", icon="⚠️")

# -----------------------------
# Sidebar: Inputs only (Demo Mode)
# -----------------------------
with st.sidebar:
    st.header("Inputs")

    st.caption("All values must be positive. Units shown.")

    # (light guardrails via min_value)
    Iameas = st.number_input("Arc Current Iameas (kA)", value=15.0, min_value=0.0)
    t = st.number_input("Arc Duration t (s)", value=0.5, min_value=0.0)
    D_mm = st.number_input("Working Distance D (mm)", value=914.0, min_value=0.0)
    gap_mm = st.number_input("Electrode Gap (mm)", value=32.0, min_value=0.0)
    Voc = st.number_input("Open Circuit Voltage Voc (kV)", value=13.8, min_value=0.0)
    Ibf = st.number_input("Bolted Fault Current Ibf (kA)", value=20.0, min_value=0.0)

    show_unc = st.checkbox("Show uncertainty band", value=True)
    mc_n = st.slider("Uncertainty samples", 10, 80, 30, 10, disabled=not show_unc)

    st.divider()
    run_btn = st.button("Run prediction", use_container_width=True)

    with st.expander("Model details", expanded=False):
        st.write(f"**Device:** {device}")
        if st.session_state.meta:
            st.json({"paths": st.session_state.meta.get("paths", {}),
                     "arch": st.session_state.meta.get("arch", {}),
                     "schema": st.session_state.meta.get("schema", {})})

    inp = dict(Iameas=Iameas, t=t, D_mm=D_mm, gap_mm=gap_mm, Voc=Voc, Ibf=Ibf)


# -----------------------------
# Main: Prediction panel (Demo Mode)
# -----------------------------
colL, colR = st.columns([1.2, 1])

with colL:
    st.subheader("Prediction")

    if not loaded:
        st.info("Train a model in Advanced → Training, then refresh/reload.")
        st.stop()

    bad = validate_inputs(inp)
    if bad:
        st.warning(f"Fix inputs (must be > 0): {bad}")
        st.stop()

    df_one = pd.DataFrame([inp])

    # auto-run once (nice demo behavior) or run when button pressed
    if "has_run_once" not in st.session_state:
        st.session_state.has_run_once = False

    do_run = run_btn or (not st.session_state.has_run_once)

    if do_run:
        st.session_state.has_run_once = True

        if show_unc:
            mu, sig = predict_mc_dropout(
                st.session_state.model,
                st.session_state.scaler_X,
                st.session_state.scaler_y,
                df_one,
                n=mc_n,
            )
            pred = float(mu[0])
            unc = float(sig[0])
            m1, m2 = st.columns([1, 1])
            m1.metric("IEmeas (cal/cm²)", f"{pred:.2f}")
            m2.metric("Uncertainty (1σ)", f"±{unc:.2f}")
        else:
            out = predict_with_artifacts(
                st.session_state.model,
                st.session_state.scaler_X,
                st.session_state.scaler_y,
                df_one,
            )
            pred = float(out["pred_IEmeas"].iloc[0])
            st.metric("IEmeas (cal/cm²)", f"{pred:.2f}")
            unc = None

        st.caption("Note: uncertainty is estimated with Monte Carlo dropout (not a calibrated safety guarantee).")

        # in-distribution check (1–99% bounds)
        # you can flip use_combined to False if you want bounds from ieeeall only
        bounds, (t_lo, t_med, t_hi) = training_bounds(use_combined=True)
        oor = [c for c in FEATURE_COLS if not (bounds[c][0] <= inp[c] <= bounds[c][1])]
        if oor:
            st.warning(
                f"Some inputs are outside the training distribution (1–99%): {oor}. "
                "Prediction may be less reliable."
            )
        else:
            st.info("Inputs are within the training distribution (1–99%).")

with colR:
    st.subheader("Prediction Context")

    if loaded and st.session_state.has_run_once:
        # recompute prediction from current sidebar inputs
        df_one = pd.DataFrame([inp])
        out = predict_with_artifacts(
            st.session_state.model,
            st.session_state.scaler_X,
            st.session_state.scaler_y,
            df_one
        )
        pred_now = float(out["pred_IEmeas"].iloc[0])

        # load + clean training set (combined)
        raw_df = load_mv_data(use_combined=True)
        clean_df = clean_data(raw_df)

        fig, (t_lo, t_med, t_hi) = plot_training_target_hist(clean_df, pred_now)
        st.pyplot(fig)
        plt.close(fig)

        # helpful numeric context
        in_range = (pred_now >= t_lo) and (pred_now <= t_hi)
        st.caption(
            f"p01={t_lo:.3g}, median={t_med:.3g}, p99={t_hi:.3g} | "
            f"prediction={pred_now:.3g} | "
            f"{'Within training range' if in_range else 'Outside training range'}"
        )

    else:
        st.write("Run a prediction to see context vs the training distribution.")


# -----------------------------
# Advanced: Train / Batch / Live (hidden by default)
# -----------------------------
with st.expander("Advanced (Training / Batch / Live)", expanded=False):
    tab_train, tab_predict, tab_live = st.tabs(["Training", "Batch Predict + Visualize", "Live Sensor Simulation"])

    # -----------------------------
    # Training tab
    # -----------------------------
    with tab_train:
        st.subheader("Train ResNet Regressor (tabular features → IEmeas)")

        c1, c2, c3, c4 = st.columns(4)
        use_combined = c1.checkbox("Use combined (epri+hcb+ieeeall)", value=True, key="train_use_combined")
        epochs = c2.number_input("Epochs", min_value=1, max_value=500, value=120, step=10, key="train_epochs")
        batch_size = c3.number_input("Batch size", min_value=8, max_value=512, value=32, step=8, key="train_bs")
        lr = c4.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f", key="train_lr")

        c5, c6, c7 = st.columns(3)
        dropout = c5.slider("Dropout", min_value=0.0, max_value=0.6, value=float(DEFAULT_DROPOUT), step=0.05, key="train_do")
        weight_decay = c6.number_input("Weight decay", min_value=0.0, max_value=1e-2, value=1e-5, format="%.6f", key="train_wd")
        out_dir_choice = c7.selectbox("Save artifacts to", ["Repo root", "Models/"], index=1, key="train_outdir")

        # Preview data stats
        try:
            raw_df, clean_df = cached_load_and_clean(use_combined=use_combined)
            st.write(f"Raw rows: **{len(raw_df)}** → Clean rows: **{len(clean_df)}**")
            st.write(
                "Target IEmeas stats (clean):",
                f"min={clean_df[TARGET_COL].min():.3f}, median={clean_df[TARGET_COL].median():.3f}, max={clean_df[TARGET_COL].max():.3f}",
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

        train_btn = st.button("▶ Train now", key="train_btn")

        if train_btn:
            (Xtr, ytr), (Xva, yva), (Xte, yte), scaler_X, scaler_y = split_and_scale(clean_df, seed=SEED)

            train_ds = ArcFlashDataset(Xtr, ytr)
            val_ds = ArcFlashDataset(Xva, yva)
            test_ds = ArcFlashDataset(Xte, yte)

            train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=int(batch_size), shuffle=False)

            model = ResNetRegressor(input_dim=len(FEATURE_COLS), hidden_dims=DEFAULT_HIDDEN_DIMS, dropout=float(dropout)).to(device)
            optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

            progress = st.progress(0)
            status = st.empty()
            chart_area = st.empty()

            train_losses = []
            val_losses = []
            best_val = float("inf")
            best_epoch = 0
            best_state = None

            for ep in range(int(epochs)):
                model.train()
                tr = 0.0
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    yp = model(xb)
                    loss = mse_loss(yp, yb)
                    loss.backward()
                    optimizer.step()
                    tr += float(loss.item())
                tr /= max(1, len(train_loader))
                train_losses.append(tr)

                model.eval()
                vl = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        yp = model(xb)
                        loss = mse_loss(yp, yb)
                        vl += float(loss.item())
                vl /= max(1, len(val_loader))
                val_losses.append(vl)

                scheduler.step(vl)

                if vl < best_val:
                    best_val = vl
                    best_epoch = ep
                    best_state = {
                        "epoch": ep,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": tr,
                        "val_loss": vl,
                        "arch": {"input_dim": len(FEATURE_COLS), "hidden_dims": DEFAULT_HIDDEN_DIMS, "dropout": float(dropout)},
                        "schema": {"feature_cols": FEATURE_COLS, "target_col": TARGET_COL},
                        "seed": SEED,
                    }

                progress.progress((ep + 1) / int(epochs))
                status.write(
                    f"Epoch {ep+1}/{int(epochs)} | Train MSE: {tr:.4f} | Val MSE: {vl:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

                fig, ax = plt.subplots(figsize=(7, 3))
                ax.plot(train_losses, label="Train MSE", alpha=0.85)
                ax.plot(val_losses, label="Val MSE", alpha=0.85)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("MSE")
                ax.set_title("Training History (live)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                chart_area.pyplot(fig)
                plt.close(fig)

            if best_state is None:
                st.error("Training failed to produce a best checkpoint.")
                st.stop()

            model.load_state_dict(best_state["model_state_dict"])
            model.eval()

            preds, actuals = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    yp = model(xb).cpu().numpy()
                    preds.append(yp)
                    actuals.append(yb.numpy())
            preds = np.vstack(preds)
            actuals = np.vstack(actuals)

            preds_orig = scaler_y.inverse_transform(preds).reshape(-1)
            actuals_orig = scaler_y.inverse_transform(actuals).reshape(-1)
            preds_orig = np.clip(preds_orig, 0, None)

            mae, rmse, rmsle, r2 = metrics_np(actuals_orig, preds_orig)

            st.success("Training complete ✅")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("MAE", f"{mae:.3f}")
            m2.metric("RMSE", f"{rmse:.3f}")
            m3.metric("RMSLE", f"{rmsle:.4f}" if np.isfinite(rmsle) else "n/a")
            m4.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "n/a")

            mask = (actuals_orig > 0) & (preds_orig > 0)
            if np.any(mask):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(actuals_orig[mask], preds_orig[mask], alpha=0.6, s=18)
                mn = float(min(actuals_orig[mask].min(), preds_orig[mask].min()))
                mx = float(max(actuals_orig[mask].max(), preds_orig[mask].max()))
                ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("Actual IEmeas (cal/cm²)")
                ax.set_ylabel("Predicted IEmeas (cal/cm²)")
                ax.set_title("Predicted vs Actual (Test)")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            out_dir = REPO_DIR if out_dir_choice == "Repo root" else (REPO_DIR / "Models")
            paths = artifact_paths(out_dir)

            torch.save(best_state, paths["model"])
            joblib.dump(scaler_X, paths["sx"])
            joblib.dump(scaler_y, paths["sy"])

            st.info(f"Saved artifacts to: {out_dir}")
            st.code(f"{paths['model']}\n{paths['sx']}\n{paths['sy']}")

            # load into session so main Demo updates immediately
            st.session_state.model = model
            st.session_state.scaler_X = scaler_X
            st.session_state.scaler_y = scaler_y
            st.session_state.meta = {
                "epoch": best_epoch + 1,
                "val_loss": float(best_val),
                "paths": {k: str(v) for k, v in paths.items()},
                "device": str(device),
                "arch": best_state.get("arch", {}),
                "schema": best_state.get("schema", {}),
            }
            st.session_state.train_history = {"train": train_losses, "val": val_losses}

            # refresh cached bounds next time (in case training distribution changed)
            training_bounds.clear()

    # -----------------------------
    # Batch predict tab
    # -----------------------------
    with tab_predict:
        st.subheader("Batch Predict + Visualize")

        if st.session_state.model is None:
            st.warning("Model not loaded. Train a model or add artifacts to repo root / Models/.")
            st.stop()

        uploaded = st.file_uploader("Upload a CSV (must include the 6 feature columns)", type=["csv"], key="batch_upload")
        st.caption(f"Required feature columns: {FEATURE_COLS}")

        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Uploaded shape:", df.shape)
            st.dataframe(df.head(25), use_container_width=True)

            try:
                out = predict_with_artifacts(st.session_state.model, st.session_state.scaler_X, st.session_state.scaler_y, df)
            except Exception as e:
                st.error(str(e))
                st.stop()

            st.success("Predictions generated ✅")
            st.dataframe(out.head(50), use_container_width=True)

            if TARGET_COL in out.columns:
                actual = out[TARGET_COL].astype(float).values
                pred = out["pred_IEmeas"].astype(float).values
                mae, rmse, rmsle, r2 = metrics_np(actual, pred)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MAE", f"{mae:.3f}")
                m2.metric("RMSE", f"{rmse:.3f}")
                m3.metric("RMSLE", f"{rmsle:.4f}" if np.isfinite(rmsle) else "n/a")
                m4.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "n/a")

                mask = (actual > 0) & (pred > 0)
                if np.any(mask):
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(actual[mask], pred[mask], alpha=0.6, s=18)
                    mn = float(min(actual[mask].min(), pred[mask].min()))
                    mx = float(max(actual[mask].max(), pred[mask].max()))
                    ax.plot([mn, mx], [mn, mx], linestyle="--", linewidth=2)
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    ax.set_xlabel("Actual IEmeas (cal/cm²)")
                    ax.set_ylabel("Predicted IEmeas (cal/cm²)")
                    ax.set_title("Predicted vs Actual")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    # -----------------------------
    # Live simulation tab
    # -----------------------------
    with tab_live:
        st.subheader("Live Sensor Simulation (replay rows → predict continuously)")

        if st.session_state.model is None:
            st.warning("Model not loaded. Train a model or add artifacts to repo root / Models/.")
            st.stop()

        st.write(
            "This simulates live sensor streaming by stepping through a dataset row-by-row "
            "and running the model on each row."
        )

        source_mode = st.radio("Data source", ["Use mv-data/ieeeall.csv (recommended)", "Upload a CSV"], index=0)

        if source_mode == "Upload a CSV":
            up = st.file_uploader("Upload CSV for live simulation", type=["csv"], key="live_upload")
            if up is None:
                st.stop()
            live_df = pd.read_csv(up)
        else:
            if not (DATA_DIR / "ieeeall.csv").exists():
                st.error("mv-data/ieeeall.csv not found.")
                st.stop()
            live_df = pd.read_csv(DATA_DIR / "ieeeall.csv")

        missing = [c for c in FEATURE_COLS if c not in live_df.columns]
        if missing:
            st.error(f"Live source is missing required columns: {missing}")
            st.stop()

        live_df2 = live_df.dropna(subset=FEATURE_COLS).copy()
        for c in FEATURE_COLS:
            live_df2 = live_df2[live_df2[c] > 0]

        if len(live_df2) < 5:
            st.error("Not enough usable rows in live dataset after filtering.")
            st.stop()

        speed_ms = st.slider("Update interval (ms)", min_value=200, max_value=5000, value=800, step=100)
        max_points = st.slider("Points on rolling plot", min_value=20, max_value=500, value=120, step=20)

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("▶ Start", key="live_start"):
                st.session_state.live_running = True
        with c2:
            if st.button("⏸ Stop", key="live_stop"):
                st.session_state.live_running = False
        with c3:
            if st.button("↺ Reset", key="live_reset"):
                st.session_state.live_idx = 0
                st.session_state.live_log = []

        pred_card = st.empty()
        table_area = st.empty()
        plot_area = st.empty()

        if st.session_state.live_running:
            steps = 10
            for _ in range(steps):
                idx = st.session_state.live_idx % len(live_df2)
                row = live_df2.iloc[[idx]].copy()

                out = predict_with_artifacts(st.session_state.model, st.session_state.scaler_X, st.session_state.scaler_y, row)
                pred = float(out["pred_IEmeas"].iloc[0])

                actual = None
                if TARGET_COL in row.columns and pd.notna(row[TARGET_COL].iloc[0]):
                    try:
                        actual = float(row[TARGET_COL].iloc[0])
                    except Exception:
                        actual = None

                st.session_state.live_log.append((time.time(), pred, actual))
                st.session_state.live_idx += 1

                pred_card.markdown(
                    f"""
                    ### Latest Prediction
                    **Predicted IEmeas:** `{pred:.2f}` cal/cm²  
                    **Row:** `{idx}`  
                    {"**Actual IEmeas:** `%.2f` cal/cm²" % actual if actual is not None else ""}
                    """
                )

                show = row[FEATURE_COLS].copy()
                show["pred_IEmeas"] = pred
                if actual is not None:
                    show["IEmeas_actual"] = actual
                    show["error"] = pred - actual
                table_area.dataframe(show, use_container_width=True)

                log = st.session_state.live_log[-int(max_points):]
                ps = np.array([p for _, p, _ in log])
                as_ = np.array([a if a is not None else np.nan for _, _, a in log])

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(ps, label="Predicted IEmeas", linewidth=2)
                if np.any(~np.isnan(as_)):
                    ax.plot(as_, label="Actual IEmeas", linewidth=2, alpha=0.8)
                ax.set_title("Live stream (rolling)")
                ax.set_xlabel("Step")
                ax.set_ylabel("cal/cm²")
                ax.grid(True, alpha=0.3)
                ax.legend()
                plot_area.pyplot(fig)
                plt.close(fig)

                time.sleep(speed_ms / 1000.0)

            st.experimental_rerun()
        else:
            if st.session_state.live_log:
                log = st.session_state.live_log[-int(max_points):]
                ps = np.array([p for _, p, _ in log])
                as_ = np.array([a if a is not None else np.nan for _, _, a in log])

                pred_card.markdown("### Live simulation stopped")
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(ps, label="Predicted IEmeas", linewidth=2)
                if np.any(~np.isnan(as_)):
                    ax.plot(as_, label="Actual IEmeas", linewidth=2, alpha=0.8)
                ax.set_title("Last rolling window")
                ax.grid(True, alpha=0.3)
                ax.legend()
                plot_area.pyplot(fig)
                plt.close(fig)