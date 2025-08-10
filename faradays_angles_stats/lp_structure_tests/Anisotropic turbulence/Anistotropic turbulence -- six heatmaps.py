#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LP16 summary dashboard (one page, six heatmaps)

Reads the JSONL log file produced by the validation script and renders:
Top row (GLOBAL):   r[Re<PP*> vs m=2], r[Re<PP> vs m=2+4], r[Im<PP> vs m=2+4]
Bottom row (LOCAL): r[Re<PP> vs m=2+4], r[Im<PP> vs m=2+4], gain = local Re(m=2+4) - global Re(m=2+4)

- Rows = theta_deg  (0,15,30,45,60,75,90)
- Cols = R_frac     (0.15,0.22,0.30,0.40)
- Annotations: value in each cell, NaNs shown as '—'
- Saves to img/LP16_dashboard.png and img/LP16_dashboard.pdf

If multiple records exist per (theta,R), the most recent by timestamp is used.
If log path is not specified, the script will pick the latest matching file in img/.
"""

import json
import glob
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ----------------- CONFIG -----------------
OUT_DIR = pathlib.Path("img")
# If you know the exact filename, set it here; otherwise None to auto-pick latest:
LOG_PATH = None  # e.g., pathlib.Path("img/lp16_bestcorr_N256_ns192_nofaraday.jsonl")

# Order to display
THETA_ORDER = list(np.linspace(0.0, 90.0, 19))#[0, 15, 30, 45, 60, 75, 90]
R_ORDER     = list(np.linspace(0.15, 0.40, 11))#[0.15, 0.22, 0.30, 0.40]

# Figure settings
DPI = 200
FIGSIZE = (12.5, 7.8)  # width, height in inches

# ------------------------------------------

def find_latest_log():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cands = sorted(glob.glob(str(OUT_DIR / "lp16_bestcorr_*.jsonl")))
    if not cands:
        raise FileNotFoundError("No JSONL logs found in 'img/'. Run the validator first.")
    return pathlib.Path(cands[-1])

def read_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs

def latest_by_key(recs, key_fields=("theta_deg","R_frac")):
    """Return dict[(theta,R)] -> most recent record by timestamp."""
    buckets = defaultdict(list)
    for r in recs:
        try:
            k = (int(r["theta_deg"]), float(r["R_frac"]))
            buckets[k].append(r)
        except KeyError:
            continue
    latest = {}
    for k, lst in buckets.items():
        lst = [x for x in lst if "timestamp" in x]
        if not lst: 
            continue
        latest[k] = max(lst, key=lambda x: x["timestamp"])
    return latest

def build_matrix(latest_map, key):
    """Return 2D matrix (len(THETA_ORDER) x len(R_ORDER)) filled with values for 'key' (or NaN)."""
    M = np.full((len(THETA_ORDER), len(R_ORDER)), np.nan, dtype=float)
    for i, th in enumerate(THETA_ORDER):
        for j, R in enumerate(R_ORDER):
            rec = latest_map.get((th, R))
            if rec is not None and key in rec and rec[key] is not None:
                val = rec[key]
                try:
                    M[i, j] = float(val)
                except Exception:
                    M[i, j] = np.nan
    return M

def draw_heatmap(ax, data, title, vmin=-1, vmax=1, cmap="coolwarm", annotate=True, nan_color="#efefef"):
    """Draw a heatmap with annotations; NaNs colored light gray."""
    # Mask NaNs
    A = np.ma.masked_invalid(data)
    # Set colormap with NaN color
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(nan_color)

    im = ax.imshow(A, origin="upper", vmin=vmin, vmax=vmax, cmap=cm, aspect="auto")
    ax.set_title(title, fontsize=11)

    # Ticks and labels
    ax.set_yticks(range(len(THETA_ORDER)))
    ax.set_yticklabels([f"{t}°" for t in THETA_ORDER], fontsize=9)
    ax.set_xticks(range(len(R_ORDER)))
    ax.set_xticklabels([f"{r:.2f}" for r in R_ORDER], fontsize=9)

    # Gridlines for readability
    ax.set_xticks(np.arange(-.5, len(R_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(THETA_ORDER), 1), minor=True)
    ax.grid(which='minor', color='w', linewidth=0.8, alpha=0.6)

    # Annotations
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                txt = "—" if not np.isfinite(val) else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")

    return im

def main():
    log_path = pathlib.Path(LOG_PATH) if LOG_PATH else find_latest_log()
    print(f"Reading logs: {log_path}")
    recs = read_jsonl(log_path)
    latest_map = latest_by_key(recs)

    # Matrices we’ll plot
    # GLOBAL
    G_star_m2   = build_matrix(latest_map, "global_r_RePPstar_m2")   # Re<PP*> vs m=2
    G_Re_m24    = build_matrix(latest_map, "global_r_RePP_m24")      # Re<PP>  vs m=2+4
    G_Im_m24    = build_matrix(latest_map, "global_r_ImPP_m24")      # Im<PP>  vs m=2+4

    # LOCAL
    L_Re_m24    = build_matrix(latest_map, "local_r_RePP_m24")       # Re<PP> vs m=2+4
    L_Im_m24    = build_matrix(latest_map, "local_r_ImPP_m24")       # Im<PP> vs m=2+4

    # GAIN (local - global) for Re<PP> m=2+4
    Gain_Re_m24 = L_Re_m24 - G_Re_m24

    # Set up figure
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, constrained_layout=True)

    # Top row: GLOBAL
    im0 = draw_heatmap(
        axes[0,0], G_star_m2, r"GLOBAL  $\,\mathrm{Re}\langle PP^*\rangle$  vs  $m=2$",
        vmin=-1, vmax=1, cmap="coolwarm"
    )
    im1 = draw_heatmap(
        axes[0,1], G_Re_m24, r"GLOBAL  $\,\mathrm{Re}\langle PP\rangle$  vs  $m=2+4$",
        vmin=-1, vmax=1, cmap="coolwarm"
    )
    im2 = draw_heatmap(
        axes[0,2], G_Im_m24, r"GLOBAL  $\,\mathrm{Im}\langle PP\rangle$  vs  $m=2+4$",
        vmin=-1, vmax=1, cmap="coolwarm"
    )

    # Bottom row: LOCAL
    im3 = draw_heatmap(
        axes[1,0], L_Re_m24, r"LOCAL  $\,\mathrm{Re}\langle PP\rangle$  vs  $m=2+4$",
        vmin=-1, vmax=1, cmap="coolwarm"
    )
    im4 = draw_heatmap(
        axes[1,1], L_Im_m24, r"LOCAL  $\,\mathrm{Im}\langle PP\rangle$  vs  $m=2+4$",
        vmin=-1, vmax=1, cmap="coolwarm"
    )
    # Gain uses a symmetric scale around 0 based on the data
    gmax = np.nanmax(np.abs(Gain_Re_m24))
    gmax = 0.1 if not np.isfinite(gmax) else max(0.2, float(gmax))
    im5 = draw_heatmap(
        axes[1,2], Gain_Re_m24, r"GAIN  (LOCAL−GLOBAL)  $\,\mathrm{Re}\langle PP\rangle$  $m=2+4$",
        vmin=-gmax, vmax=gmax, cmap="PiYG"
    )

    # Colorbars (one for r, one for gain)
    cbar0 = fig.colorbar(im0, ax=[axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]], shrink=0.92, pad=0.02)
    cbar0.set_label("Correlation r", rotation=90)
    cbar1 = fig.colorbar(im5, ax=[axes[1,2]], shrink=0.92, pad=0.02)
    cbar1.set_label("Δr (local − global)", rotation=90)

    # Big title
    fig.suptitle("LP16 validation — global vs local frames across θ and R", fontsize=14, y=1.02)

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "LP16_dashboard.png"
    pdf = OUT_DIR / "LP16_dashboard.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    print(f"Saved:\n  {png}\n  {pdf}")

if __name__ == "__main__":
    main()
