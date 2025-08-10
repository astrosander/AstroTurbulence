#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LP16 noise & beam observing guide — one big figure (12 heatmaps)

Figure layout (4 rows × 3 cols):
Rows 1–2: order = 'beam_then_noise'
Rows 3–4: order = 'noise_then_beam'

Within each order:
  Row A (top of the pair):  Re⟨PP⟩   → [Global r, Local r, Gain (Local−Global)]
  Row B (bottom of the pair): Im⟨PP⟩ → [Global r, Local r, Gain (Local−Global)]

Axes per heatmap:
  x-axis: SNR per pixel (∞, 20, 10, 7, 5, 3, 2)
  y-axis: FWHM (px) (0, 1, 2, 4, 8)

Cell value: median across θ ∈ {0..90°}, R ∈ {0.15,0.22,0.30,0.40}, and trials (only valid bins).
Annotations: each cell shows the median (or '—' if insufficient data).

Input:
  - JSONL log produced by the white-noise simulation script.

Output:
  - img/LP16_noise_observing_guide.png
  - img/LP16_noise_observing_guide.pdf
"""

import json
import math
import glob
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# -------------------- CONFIG -------------------- #
OUT_DIR = pathlib.Path("img")
# If you want to hardcode a specific file, set LOG_PATH; otherwise it picks the latest lp16_noisegrid_*.jsonl in img/.
LOG_PATH = None  # e.g., pathlib.Path("img/lp16_noisegrid_N256_ns192_nofaraday.jsonl")

# Safety / validity thresholds
MIN_GLOBAL_BINS = 20   # require at least this many effective φ-bins in global frame
MIN_LOCAL_BINS  = 20   # same for local
MIN_SAMPLES_PER_CELL = 10  # require at least this many (θ,R,trial) samples per (SNR,FWHM,order) cell

# Heatmap styling
DPI = 220
FIGSIZE = (13.0, 12.0)
CMAP_R = "coolwarm"   # for r
CMAP_G = "PiYG"       # for gain
NAN_COLOR = "#efefef"

# ------------------------------------------------ #

def find_latest_log():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # img\\lp16_noisegrid_N256_ns192_nofaraday.jsonl

    # cands = sorted(glob.glob(str(OUT_DIR / "lp16_noisegrid_*.jsonl")))
    # print(cands)
    # if not cands:
    #     raise FileNotFoundError("No JSONL logs found in 'img/'.")
    return pathlib.Path("img\\lp16_noisegrid_N256_ns192_nofaraday.jsonl")

def read_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                pass
    if not recs:
        raise RuntimeError(f"No records found in {path}")
    return recs

def unique_sorted(values):
    # Handles None/NaN by converting None→np.inf for SNR use-case later if needed.
    uniq = []
    for v in values:
        if v not in uniq:
            uniq.append(v)
    try:
        return sorted(uniq)
    except Exception:
        return uniq

def prepare_axes_categories(records):
    # Collect unique SNRs (convert None to np.inf for display) and FWHM px
    snrs = []
    fwhms = []
    orders = []
    for r in records:
        s = r.get("snr_px", None)
        s = np.inf if (s is None) else float(s)
        snrs.append(s)
        fwhms.append(int(r.get("fwhm_px", 0)))
        orders.append(str(r.get("order", "")))
    snr_sorted = sorted(set(snrs), key=lambda x: (np.isfinite(x), x), reverse=True)  # put ∞ first
    fwhm_sorted = sorted(set(fwhms))
    order_sorted = [o for o in ["beam_then_noise", "noise_then_beam"] if o in set(orders)]
    return snr_sorted, fwhm_sorted, order_sorted

def median_or_nan(vals):
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return np.nan
    return float(np.nanmedian(vals))

def build_heat_matrices(records, snr_sorted, fwhm_sorted, order_name, field_global, field_local):
    """
    Build three matrices (FWHM x SNR): global r, local r, gain (local-global).
    Only counts records that meet bin coverage thresholds.
    """
    nF, nS = len(fwhm_sorted), len(snr_sorted)
    M_g = np.full((nF, nS), np.nan, dtype=float)
    M_l = np.full((nF, nS), np.nan, dtype=float)
    M_gain = np.full((nF, nS), np.nan, dtype=float)
    Cnt = np.zeros((nF, nS), dtype=int)  # counts used (for info)

    # Group everything by (fwhm, snr)
    # Each record includes *both* global and local stats for the same (θ,R,trial).
    by_cell = defaultdict(list)
    for r in records:
        if str(r.get("order","")) != order_name:
            continue
        snr = r.get("snr_px", None)
        snr = np.inf if (snr is None) else float(snr)
        fwhm = int(r.get("fwhm_px", 0))
        by_cell[(fwhm, snr)].append(r)

    # Fill matrices
    for iF, f in enumerate(fwhm_sorted):
        for jS, s in enumerate(snr_sorted):
            lst = by_cell.get((f, s), [])
            gvals, lvals, gains = [], [], []
            for r in lst:
                # Global validity
                if r.get("global_eff_bins", 0) < MIN_GLOBAL_BINS:
                    pass_valid_g = False
                else:
                    pass_valid_g = True
                # Local validity
                has_local = (r.get("local_sigma", None) is not None) and (r.get("local_eff_bins", 0) >= MIN_LOCAL_BINS)
                pass_valid_l = bool(has_local)

                # Pull numbers
                rv_g = r.get(field_global, None)
                rv_l = r.get(field_local, None)

                if pass_valid_g and (rv_g is not None) and np.isfinite(rv_g):
                    gvals.append(float(rv_g))
                if pass_valid_l and (rv_l is not None) and np.isfinite(rv_l):
                    lvals.append(float(rv_l))
                if pass_valid_g and pass_valid_l and (rv_g is not None) and (rv_l is not None) and np.isfinite(rv_g) and np.isfinite(rv_l):
                    gains.append(float(rv_l - rv_g))

            # Require enough samples to be confident
            if len(gvals) >= MIN_SAMPLES_PER_CELL:
                M_g[iF, jS] = median_or_nan(gvals)
            if len(lvals) >= MIN_SAMPLES_PER_CELL:
                M_l[iF, jS] = median_or_nan(lvals)
            if len(gains) >= MIN_SAMPLES_PER_CELL:
                M_gain[iF, jS] = median_or_nan(gains)
            Cnt[iF, jS] = len(gains)  # store the stricter count (both valid)

    return M_g, M_l, M_gain, Cnt

def draw_heat(ax, M, x_labels, y_labels, title, vmin, vmax, cmap, annotate=True, nan_color="#efefef"):
    """
    Draw a categorical heatmap (rows=y, cols=x) using imshow; annotate with numbers.
    """
    # Mask NaNs with special color
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(nan_color)
    A = np.ma.masked_invalid(M)

    im = ax.imshow(A, origin="lower", vmin=vmin, vmax=vmax, cmap=cm, aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("SNR per pixel", fontsize=9)
    ax.set_ylabel("FWHM (px)", fontsize=9)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.8, alpha=0.6)

    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                txt = "—" if not np.isfinite(val) else f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    return im

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(LOG_PATH) if LOG_PATH else find_latest_log()
    print(f"Reading: {log_path}")
    recs = read_jsonl(log_path)

    # Category axes
    snr_sorted, fwhm_sorted, order_sorted = prepare_axes_categories(recs)
    snr_labels = [("∞" if not np.isfinite(s) else (str(int(s)) if float(int(s)) == s else f"{s:g}")) for s in snr_sorted]
    fwhm_labels = [str(f) for f in fwhm_sorted]

    # Build matrices for each order and each observable set
    # Field names in logs:
    #   Re global:  'global_r_RePP_m24'
    #   Re local:   'local_r_RePP_m24'
    #   Im global:  'global_r_ImPP_m24'
    #   Im local:   'local_r_ImPP_m24'
    grids = {}
    for order in order_sorted:
        Re_g, Re_l, Re_gain, C_re = build_heat_matrices(
            recs, snr_sorted, fwhm_sorted, order,
            field_global="global_r_RePP_m24",
            field_local ="local_r_RePP_m24",
        )
        Im_g, Im_l, Im_gain, C_im = build_heat_matrices(
            recs, snr_sorted, fwhm_sorted, order,
            field_global="global_r_ImPP_m24",
            field_local ="local_r_ImPP_m24",
        )
        grids[order] = dict(Re_g=Re_g, Re_l=Re_l, Re_gain=Re_gain,
                            Im_g=Im_g, Im_l=Im_l, Im_gain=Im_gain)

    # Color limits
    r_vmin, r_vmax = -1.0, 1.0
    # Gain symmetric limit across all gains
    all_gains = []
    for o in order_sorted:
        g1 = grids[o]["Re_gain"]; g2 = grids[o]["Im_gain"]
        all_gains.extend(list(np.nan_to_num(g1, nan=np.nan).ravel()))
        all_gains.extend(list(np.nan_to_num(g2, nan=np.nan).ravel()))
    all_gains = np.array([x for x in all_gains if np.isfinite(x)], dtype=float)
    g_abs = 0.0 if all_gains.size==0 else float(np.nanmax(np.abs(all_gains)))
    g_abs = max(g_abs, 0.2)  # ensure some dynamic range
    g_vmin, g_vmax = -g_abs, g_abs

    # Figure layout: 4 rows × 3 cols (two orders × [Re-row, Im-row])
    n_rows_total = 2 * len(order_sorted)
    fig, axes = plt.subplots(n_rows_total, 3, figsize=FIGSIZE, constrained_layout=True)

    # If only one order present, make sure axes is 2D
    if n_rows_total == 1:
        axes = np.array([axes])

    heat_r_axes = []  # for shared colorbar (r)
    heat_g_axes = []  # for shared colorbar (gain)

    row = 0
    for order in order_sorted:
        # Titles per order
        row_Re = row
        row_Im = row + 1
        # RE row
        im0 = draw_heat(axes[row_Re, 0], grids[order]["Re_g"], snr_labels, fwhm_labels,
                        title=f"{order} — Re⟨PP⟩  (GLOBAL r, m=2+4)",
                        vmin=r_vmin, vmax=r_vmax, cmap=CMAP_R)
        im1 = draw_heat(axes[row_Re, 1], grids[order]["Re_l"], snr_labels, fwhm_labels,
                        title=f"{order} — Re⟨PP⟩  (LOCAL r, m=2+4)",
                        vmin=r_vmin, vmax=r_vmax, cmap=CMAP_R)
        im2 = draw_heat(axes[row_Re, 2], grids[order]["Re_gain"], snr_labels, fwhm_labels,
                        title=f"{order} — Re⟨PP⟩  (GAIN = LOCAL − GLOBAL)",
                        vmin=g_vmin, vmax=g_vmax, cmap=CMAP_G)
        heat_r_axes.extend([axes[row_Re,0], axes[row_Re,1]])
        heat_g_axes.append(axes[row_Re,2])

        # IM row
        im3 = draw_heat(axes[row_Im, 0], grids[order]["Im_g"], snr_labels, fwhm_labels,
                        title=f"{order} — Im⟨PP⟩  (GLOBAL r, m=2+4)",
                        vmin=r_vmin, vmax=r_vmax, cmap=CMAP_R)
        im4 = draw_heat(axes[row_Im, 1], grids[order]["Im_l"], snr_labels, fwhm_labels,
                        title=f"{order} — Im⟨PP⟩  (LOCAL r, m=2+4)",
                        vmin=r_vmin, vmax=r_vmax, cmap=CMAP_R)
        im5 = draw_heat(axes[row_Im, 2], grids[order]["Im_gain"], snr_labels, fwhm_labels,
                        title=f"{order} — Im⟨PP⟩  (GAIN = LOCAL − GLOBAL)",
                        vmin=g_vmin, vmax=g_vmax, cmap=CMAP_G)
        heat_r_axes.extend([axes[row_Im,0], axes[row_Im,1]])
        heat_g_axes.append(axes[row_Im,2])

        row += 2

    # Shared colorbars
    cbar_r = fig.colorbar(im0, ax=heat_r_axes, shrink=0.95, pad=0.02)
    cbar_r.set_label("Correlation r", rotation=90)
    cbar_g = fig.colorbar(im2, ax=heat_g_axes, shrink=0.95, pad=0.02)
    cbar_g.set_label("Gain Δr (LOCAL − GLOBAL)", rotation=90)

    fig.suptitle("LP16 under white noise & beam — median correlations over θ and R\n"
                 "(Values annotated in cells; blank = insufficient samples)",
                 fontsize=14, y=1.02)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "LP16_noise_observing_guide.png"
    pdf = OUT_DIR / "LP16_noise_observing_guide.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    print(f"Saved:\n  {png}\n  {pdf}")

if __name__ == "__main__":
    main()
