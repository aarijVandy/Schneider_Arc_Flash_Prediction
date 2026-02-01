import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def fig1():
    rng = np.array([24, 0])
    case_df = d[d["config"] == "EPRI HCB"]

    fig, ax = plt.subplots()
    sc = ax.scatter(case_df["IE"], case_df["IEmeas"],
                    c=case_df["Iameas"], cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Iₐ, kA")

    # 2x, 2.5x, 3x lines
    x = rng / 2.0
    ax.plot(x, rng, color="purple")
    ax.text(x[0], rng[0], "2x", va="bottom")

    x = rng / 2.5
    ax.plot(x, rng, color="darkblue")
    ax.text(x[0], rng[0], "2.5x", va="bottom")

    x = rng / 3.0
    ax.plot(x, rng, color="blue")
    ax.text(x[0], rng[0], "3x", va="bottom")

    ax.set_xlabel("Predicted, assuming HCB")
    ax.set_ylabel("Measured")
    ax.set_title("Incident energy, cal/cm²")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 25)
    fig.tight_layout()
    return fig, ax


def fig2():
    rng = np.array([24, 0])
    if "D_in" not in d.columns and "D_mm" in d.columns:
        d["D_in"] = (d["D_mm"] / 25.4).round().astype(int)

    case_df = d[d["config"] == "EPRI HCB"]

    fig, ax = plt.subplots()
    sc = ax.scatter(case_df["IE"], case_df["IEmeas"],
                    c=case_df["D_in"], cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("D, in")

    x = rng / 2.0
    ax.plot(x, rng, color="purple")
    ax.text(x[0], rng[0], "2x", va="bottom")

    x = rng / 2.5
    ax.plot(x, rng, color="darkblue")
    ax.text(x[0], rng[0], "2.5x", va="bottom")

    x = rng / 3.0
    ax.plot(x, rng, color="blue")
    ax.text(x[0], rng[0], "3x", va="bottom")

    ax.set_xlabel("Predicted, assuming HCB")
    ax.set_ylabel("Measured")
    ax.set_title("Incident energy, cal/cm²")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 25)
    fig.tight_layout()
    return fig, ax


def fig3():
    rng = np.array([10, 0])
    case_df = d[d["config"] == "EPRI VCB"]

    fig, ax = plt.subplots()
    sc = ax.scatter(case_df["IE"], case_df["IEmeas"],
                    c=case_df["Iameas"], cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Iₐ, kA")

    ax.plot(rng, rng, color="purple")  # y = x line

    ax.set_xlabel("Predicted, assuming VCB")
    ax.set_ylabel("Measured")
    ax.set_title("Incident energy, cal/cm²")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    fig.tight_layout()
    return fig, ax


def fig4():
    rng = np.array([14, 0])
    case_df = d[d["config"] == "EPRI Transformer"]

    fig, ax = plt.subplots()
    sc = ax.scatter(case_df["IE"], case_df["IEmeas"],
                    c=case_df["Iameas"], cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Iₐ, kA")

    x = rng
    ax.plot(x, rng, color="purple")
    ax.text(x[0], rng[0], "1x", va="bottom")

    x = rng / 1.5
    ax.plot(x, rng, color="darkblue")
    ax.text(x[0], rng[0], "1.5x", va="bottom")

    ax.set_xlabel("Predicted, assuming HCB")
    ax.set_ylabel("Measured")
    ax.set_title("Incident energy, cal/cm²")
    fig.tight_layout()
    return fig, ax


def fig5():
    rng = np.array([19, 0])
    rng2 = np.array([12, 0])
    case_df = d[d["config"] == "EPRI PMH-9"].dropna(
        subset=["IE", "IEmeas", "Iameas"]
    )

    fig, ax = plt.subplots()
    sc = ax.scatter(case_df["IE"], case_df["IEmeas"],
                    c=case_df["Iameas"], cmap="viridis")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Iₐ, kA")

    ax.plot(rng2, rng2, color="purple")
    ax.text(rng2[0], rng2[0], "1x", va="bottom")

    x = rng / 2.0
    ax.plot(x, rng, color="darkblue")
    ax.text(x[0], rng[0], "2x", va="bottom")

    ax.set_xlabel("Predicted, assuming HCB")
    ax.set_ylabel("Measured")
    ax.set_title("Incident energy, cal/cm²")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 20)
    fig.tight_layout()
    return fig, ax


def fig6():
    dt = hcb[hcb["energyratio"] < 5].copy()
    dt["arcV"] = dt["joules"] / dt["t"] / dt["Iameas"] / 3.0

    fig, ax = plt.subplots()

    markers = {"EPRI HCB": "o", "IEEE HCB": "s"}
    for cfg, m in markers.items():
        sub = dt[dt["config"] == cfg]
        if not sub.empty:
            ax.scatter(sub["gap_mm"], sub["arcV"], label=cfg, marker=m)

    ax.set_xlabel("Electrode gap, mm")
    ax.set_ylabel("Calculated arc voltage, V")
    ax.legend(title="config")
    ax.set_title("")  # Gadfly had no explicit title
    fig.tight_layout()
    return fig, ax


def fig7():
    dt = hcb[(hcb["energyratio"] < 5) &
             (hcb["D_in"] > 35) &
             (hcb["D_in"] < 40)].copy()

    fig, ax = plt.subplots()
    markers = {"EPRI HCB": "o", "IEEE HCB": "s"}
    for cfg, m in markers.items():
        sub = dt[dt["config"] == cfg]
        if not sub.empty:
            ax.scatter(sub["t"], sub["energyratio"], label=cfg, marker=m)

    ax.set_xlabel("Duration, secs")
    ax.set_ylabel("Energy ratio, cal/cm²/MJ")
    ax.set_title("Distance = 914 mm (36 in)")
    ax.legend(title="config")
    fig.tight_layout()
    return fig, ax


def fig8():
    dt = hcb[(hcb["energyratio"] < 5) &
             (hcb["D_in"] > 45) &
             (hcb["D_in"] < 50)].copy()

    fig, ax = plt.subplots()
    markers = {"EPRI HCB": "o", "IEEE HCB": "s"}
    for cfg, m in markers.items():
        sub = dt[dt["config"] == cfg]
        if not sub.empty:
            ax.scatter(sub["t"], sub["energyratio"], label=cfg, marker=m)

    ax.set_xlabel("Duration, secs")
    ax.set_ylabel("Energy ratio, cal/cm²/MJ")
    ax.set_title("Distance = 1219 mm (48 in)")
    ax.legend(title="config")
    fig.tight_layout()
    return fig, ax


def fig10():
    dt = hcb[hcb["energyratio"] < 5].copy()
    dt["It"] = dt["Iameas"] / dt["t"]
    dt["D_m"] = dt["D_in"] * 0.0254

    fig, ax = plt.subplots()

    configs = dt["config"].unique()
    heights = sorted(dt["height_in"].dropna().unique())
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    marker_map = {h: markers[i % len(markers)] for i, h in enumerate(heights)}

    for cfg in configs:
        sub_cfg = dt[dt["config"] == cfg]
        for h in heights:
            sub = sub_cfg[sub_cfg["height_in"] == h]
            if sub.empty:
                continue
            ax.scatter(sub["D_m"], sub["energyratio"],
                       label=f"{cfg}, {h} in",
                       marker=marker_map[h])

    ax.set_xlabel("Distance, m")
    ax.set_ylabel("Energy ratio, cal/cm²/MJ")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    return fig, ax


def fig12():
    dt = ieeeall.copy()
    dt = dt.dropna(subset=["height_in", "Voc", "config"])
    dt = dt[(dt["config"] == "VCB") &
            (dt["Voc"] > 1) &
            (dt["height_in"] == 36) &
            (dt["Ibf"] > 18) &
            (dt["Ibf"] < 22)]

    dt["D_m"] = dt["D_mm"] / 1000.0
    dt["Voc_cat"] = np.where(dt["Voc"] > 10, "14 kV", "2.7 kV")

    # size mapping based on gap_mm
    g_min, g_max = dt["gap_mm"].min(), dt["gap_mm"].max()
    min_s, max_s = 25.0, 150.0
    dt["size"] = min_s + (dt["gap_mm"] - g_min) / (g_max - g_min) * (max_s - min_s)

    fig, ax = plt.subplots()
    for voc in dt["Voc_cat"].unique():
        sub = dt[dt["Voc_cat"] == voc]
        ax.scatter(sub["D_m"], sub["IErate"],
                   s=sub["size"],
                   label=voc,
                   alpha=0.7)

    ax.set_xlabel("Distance, m")
    ax.set_ylabel("Heat rate, cal/cm²/sec")
    ax.legend(title="Voc")
    fig.tight_layout()
    return fig, ax


def fig13():
    dt = ieeeall.copy()
    dt = dt.dropna(subset=["height_in", "Voc", "config"])
    dt = dt[(dt["config"] == "HCB") &
            (dt["Voc"] > 1) &
            (dt["height_in"] == 36) &
            (dt["Ibf"] > 18) &
            (dt["Ibf"] < 22)]

    dt["D_m"] = dt["D_mm"] / 1000.0
    dt["Voc_cat"] = np.where(dt["Voc"] > 10, "14 kV", "2.7 kV")

    g_min, g_max = dt["gap_mm"].min(), dt["gap_mm"].max()
    min_s, max_s = 25.0, 150.0
    dt["size"] = min_s + (dt["gap_mm"] - g_min) / (g_max - g_min) * (max_s - min_s)

    fig, ax = plt.subplots()
    for voc in dt["Voc_cat"].unique():
        sub = dt[dt["Voc_cat"] == voc]
        ax.scatter(sub["D_m"], sub["IErate"],
                   s=sub["size"],
                   label=voc,
                   alpha=0.7)

    ax.set_xlabel("Distance, m")
    ax.set_ylabel("Heat rate, cal/cm²/sec")
    ax.legend(title="Voc")
    fig.tight_layout()
    return fig, ax
