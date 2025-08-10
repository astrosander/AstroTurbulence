#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LP16 theory ⇄ experiment: single, self-explanatory validation figure.

Panel A: r vs SNR for Re⟨PP⟩ and Im⟨PP⟩ (GLOBAL solid, LOCAL dashed),
         16–84% bands across θ and R. Gray line at r=1 = LP16 ideal.
Panel B: Gain = r_local − r_global for Re and Im, with 16–84% bands.
         Green shading marks the "positive improvement" region.

Outputs:
  img/LP16_single_validating.png
  img/LP16_single_validating.pdf
"""

import json, glob, pathlib
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --------- CONFIG ---------
OUT_DIR = pathlib.Path("img")
LOG_PATH = None  # set explicit path if desired; else latest img/lp16_noisegrid_*.jsonl is used

# Choose the single configuration to display
FWHM_PX = 4
ORDER   = "beam_then_noise"     # or "noise_then_beam"

# Validity thresholds & plotting
MIN_GLOBAL_BINS = 20
MIN_LOCAL_BINS  = 20
MIN_SAMPLES_PER_POINT = 8
DPI = 240
FIGSIZE = (9.5, 7.2)
LW = 2.2
ALPHA_BAND = 0.22
Y_LIMS_TOP = (0.52, 1.02)
Y_LIMS_GAIN = (-0.02, 0.26)
# --------------------------

def find_latest_log():
    return pathlib.Path("img/lp16_noisegrid_N256_ns192_nofaraday.jsonl")

def read_jsonl(path):
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln: continue
            try:
                recs.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    if not recs:
        raise RuntimeError(f"No records in {path}")
    return recs

def snr_axis(records, order_name, fwhm_px):
    snrs = set()
    for r in records:
        if str(r.get("order","")) != order_name: continue
        if int(r.get("fwhm_px",-1)) != int(fwhm_px): continue
        s = r.get("snr_px", None)
        s = np.inf if (s is None) else float(s)
        snrs.add(s)
    snr_sorted = sorted(snrs, key=lambda x: (np.isfinite(x), x), reverse=True)  # ∞ first
    x = np.arange(len(snr_sorted))
    labels = [("∞" if not np.isfinite(s) else str(int(s))) for s in snr_sorted]
    return snr_sorted, x, labels

def group_by_snr(records, order_name, fwhm_px):
    by = defaultdict(list)
    for r in records:
        if str(r.get("order","")) != order_name: continue
        if int(r.get("fwhm_px",-1)) != int(fwhm_px): continue
        s = r.get("snr_px", None)
        s = np.inf if (s is None) else float(s)
        by[s].append(r)
    return by

def robust_summary(vals):
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmedian(a)), float(np.nanpercentile(a,16)), float(np.nanpercentile(a,84))

def collect_series(cell_recs, f_g, f_l):
    rg, rl, gn = [], [], []
    for r in cell_recs:
        okg = r.get("global_eff_bins",0) >= MIN_GLOBAL_BINS
        okl = (r.get("local_eff_bins",0) >= MIN_LOCAL_BINS) and (r.get("local_sigma",None) is not None)
        vg = r.get(f_g, None); vl = r.get(f_l, None)
        if okg and isinstance(vg,(int,float)) and np.isfinite(vg): rg.append(float(vg))
        if okl and isinstance(vl,(int,float)) and np.isfinite(vl): rl.append(float(vl))
        if okg and okl and isinstance(vg,(int,float)) and isinstance(vl,(int,float)) and np.isfinite(vg) and np.isfinite(vl):
            gn.append(float(vl - vg))
    mg, lg, hg = robust_summary(rg) if len(rg) >= MIN_SAMPLES_PER_POINT else (np.nan, np.nan, np.nan)
    ml, ll, hl = robust_summary(rl) if len(rl) >= MIN_SAMPLES_PER_POINT else (np.nan, np.nan, np.nan)
    mgain, lgain, hgain = robust_summary(gn) if len(gn) >= MIN_SAMPLES_PER_POINT else (np.nan, np.nan, np.nan)
    return (mg, lg, hg), (ml, ll, hl), (mgain, lgain, hgain), len(gn)

def annotate_values(ax, x, y, label):
    if np.isfinite(y):
        ax.text(x, y+0.007, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color="black")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = pathlib.Path(LOG_PATH) if LOG_PATH else find_latest_log()
    print(f"Reading: {log_path}")
    recs = read_jsonl(log_path)

    # Field names
    F_RE_G = "global_r_RePP_m24"
    F_RE_L = "local_r_RePP_m24"
    F_IM_G = "global_r_ImPP_m24"
    F_IM_L = "local_r_ImPP_m24"

    snr_sorted, xpos, snr_labels = snr_axis(recs, ORDER, FWHM_PX)
    by_snr = group_by_snr(recs, ORDER, FWHM_PX)

    # Build curves
    curves = {
        "ReG": {"m":[], "l":[], "h":[]},
        "ReL": {"m":[], "l":[], "h":[]},
        "ImG": {"m":[], "l":[], "h":[]},
        "ImL": {"m":[], "l":[], "h":[]},
        "GainRe": {"m":[], "l":[], "h":[]},
        "GainIm": {"m":[], "l":[], "h":[]},
        "Nused": []
    }

    for s in snr_sorted:
        cell = by_snr.get(s, [])
        (mg, lg, hg), (ml, ll, hl), (gRe_m,gRe_l,gRe_h), nused1 = collect_series(cell, F_RE_G, F_RE_L)
        (mg2,lg2,hg2),(ml2,ll2,hl2),(gIm_m,gIm_l,gIm_h), nused2 = collect_series(cell, F_IM_G, F_IM_L)

        curves["ReG"]["m"].append(mg);   curves["ReG"]["l"].append(lg);   curves["ReG"]["h"].append(hg)
        curves["ReL"]["m"].append(ml);   curves["ReL"]["l"].append(ll);   curves["ReL"]["h"].append(hl)
        curves["ImG"]["m"].append(mg2);  curves["ImG"]["l"].append(lg2);  curves["ImG"]["h"].append(hg2)
        curves["ImL"]["m"].append(ml2);  curves["ImL"]["l"].append(ll2);  curves["ImL"]["h"].append(hl2)
        curves["GainRe"]["m"].append(gRe_m); curves["GainRe"]["l"].append(gRe_l); curves["GainRe"]["h"].append(gRe_h)
        curves["GainIm"]["m"].append(gIm_m); curves["GainIm"]["l"].append(gIm_l); curves["GainIm"]["h"].append(gIm_h)
        curves["Nused"].append(int(min(nused1, nused2)))

    # Compute baseline r(∞) and average gain over noisy SNRs
    idx_inf = 0  # because ∞ is placed first
    rinf_ReG = curves["ReG"]["m"][idx_inf]
    rinf_ReL = curves["ReL"]["m"][idx_inf]
    rinf_ImG = curves["ImG"]["m"][idx_inf]
    rinf_ImL = curves["ImL"]["m"][idx_inf]
    # average gains excluding ∞
    mask_noisy = np.arange(len(snr_sorted)) != idx_inf
    mean_gain_re = np.nanmean(np.array(curves["GainRe"]["m"], float)[mask_noisy])
    mean_gain_im = np.nanmean(np.array(curves["GainIm"]["m"], float)[mask_noisy])

    # Plot
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 2])
    ax_top = fig.add_subplot(gs[0,0])
    ax_bot = fig.add_subplot(gs[1,0])

    # Background SNR regimes (very light so it's obvious but not distracting)
    def span(ax, x0, x1, color, label=None):
        ax.axvspan(x0, x1, color=color, alpha=0.06, ec=None)
        if label:
            ax.text((x0+x1)/2, ax.get_ylim()[1]-0.03*(ax.get_ylim()[1]-ax.get_ylim()[0]),
                    label, ha="center", va="top", fontsize=9, color="black")

    # Label high/moderate/low regimes by x-index ranges (∞,20,10 | 7,5 | 3,2)
    # (works for your SNR set; adjust if needed)
    n = len(xpos)
    labels_map = {lab:(i) for i,lab in enumerate(snr_labels)}
    if set(['∞','20','10','7','5','3','2']).issubset(set(snr_labels)):
        span(ax_top, labels_map['∞']-0.5, labels_map['10']+0.5, "#1f77b4", "High SNR")
        span(ax_top, labels_map['7']-0.5,  labels_map['5']+0.5,  "#ff7f0e", "Moderate")
        span(ax_top, labels_map['3']-0.5,  labels_map['2']+0.5,  "#2ca02c", "Low SNR")
        span(ax_bot, labels_map['∞']-0.5, labels_map['10']+0.5, "#1f77b4")
        span(ax_bot, labels_map['7']-0.5,  labels_map['5']+0.5,  "#ff7f0e")
        span(ax_bot, labels_map['3']-0.5,  labels_map['2']+0.5,  "#2ca02c")

    # Panel A — r vs SNR
    def draw_with_band(ax, x, m, l, h, color, ls, label):
        m = np.array(m, float); l = np.array(l, float); h = np.array(h, float)
        ax.plot(x, m, ls, color=color, lw=LW, marker='o', label=label)
        ax.fill_between(x, l, h, color=color, alpha=ALPHA_BAND, linewidth=0)

    draw_with_band(ax_top, xpos, curves["ReG"]["m"], curves["ReG"]["l"], curves["ReG"]["h"], "#1f77b4", "-",  "Re⟨PP⟩ GLOBAL")
    draw_with_band(ax_top, xpos, curves["ReL"]["m"], curves["ReL"]["l"], curves["ReL"]["h"], "#1f77b4", "--", "Re⟨PP⟩ LOCAL")
    draw_with_band(ax_top, xpos, curves["ImG"]["m"], curves["ImG"]["l"], curves["ImG"]["h"], "#2ca02c", "-",  "Im⟨PP⟩ GLOBAL")
    draw_with_band(ax_top, xpos, curves["ImL"]["m"], curves["ImL"]["l"], curves["ImL"]["h"], "#2ca02c", "--", "Im⟨PP⟩ LOCAL")

    ax_top.axhline(1.0, color="gray", lw=1.0, ls=":", label="LP16 ideal r=1")
    ax_top.set_xlim(xpos[0]-0.4, xpos[-1]+0.4)
    ax_top.set_ylim(*Y_LIMS_TOP)
    ax_top.set_xticks(xpos); ax_top.set_xticklabels(snr_labels)
    ax_top.set_ylabel("Correlation r (median ± 16–84%)")
    ax_top.grid(True, alpha=0.25, linestyle='--')
    ax_top.set_title(f"LP16 validation — order: {ORDER}, FWHM={FWHM_PX}px — aggregated over θ and R")

    # Callouts at a couple of points (∞ and SNR=3 if present)
    if '∞' in snr_labels:
        i = snr_labels.index('∞')
        annotate_values(ax_top, xpos[i], curves["ReG"]["m"][i], "ReG")
        annotate_values(ax_top, xpos[i], curves["ImG"]["m"][i], "ImG")
    if '3' in snr_labels:
        i = snr_labels.index('3')
        annotate_values(ax_top, xpos[i], curves["ReL"]["m"][i], "ReL")
        annotate_values(ax_top, xpos[i], curves["ImL"]["m"][i], "ImL")

    # Panel B — gains
    ax_bot.axhline(0.0, color="k", lw=1.0)
    # green band for positive improvement
    ax_bot.axhspan(0.0, ax_bot.get_ylim()[1], color="#55a868", alpha=0.07, zorder=0)

    draw_with_band(ax_bot, xpos, curves["GainRe"]["m"], curves["GainRe"]["l"], curves["GainRe"]["h"], "#1f77b4", "-", "Gain (Re) LOCAL−GLOBAL")
    draw_with_band(ax_bot, xpos, curves["GainIm"]["m"], curves["GainIm"]["l"], curves["GainIm"]["h"], "#2ca02c", "-", "Gain (Im) LOCAL−GLOBAL")

    ax_bot.set_xlim(xpos[0]-0.4, xpos[-1]+0.4)
    ax_bot.set_ylim(*Y_LIMS_GAIN)
    ax_bot.set_xticks(xpos); ax_bot.set_xticklabels(snr_labels)
    ax_bot.set_xlabel("SNR per pixel")
    ax_bot.set_ylabel("Δr = LOCAL − GLOBAL")
    ax_bot.grid(True, alpha=0.25, linestyle='--')

    # Legend & text box
    leg = ax_top.legend(loc="lower right", ncol=2, frameon=False)
    txt = (
        f"Noise order: {ORDER}\n"
        f"Beam FWHM: {FWHM_PX} px\n"
        f"r(∞): Re[G/L]={rinf_ReG:.2f}/{rinf_ReL:.2f}, Im[G/L]={rinf_ImG:.2f}/{rinf_ImL:.2f}\n"
        f"Avg gain (SNR<∞): Re {mean_gain_re:+.2f}, Im {mean_gain_im:+.2f}\n"
        "Bands: 16–84% across θ∈{0…90°}, R∈{0.15,0.22,0.30,0.40}"
    )
    ax_top.text(0.01, 0.04, txt, transform=ax_top.transAxes, ha="left", va="bottom",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    # Footer: sample counts per SNR for transparency
    counts_str = "samples per SNR: " + ", ".join([f"{lab}:{n}" for lab,n in zip(snr_labels, curves["Nused"])])
    fig.text(0.5, 0.01, counts_str, ha="center", va="bottom", fontsize=9, color="#444444")

    # Save
    png = OUT_DIR / "LP16_single_validating.png"
    pdf = OUT_DIR / "LP16_single_validating.pdf"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf, dpi=DPI, bbox_inches="tight")
    print(f"Saved:\n  {png}\n  {pdf}")

if __name__ == "__main__":
    main()
