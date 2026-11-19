#!/usr/bin/env python3
"""
MSO58B Harmonic Analyzer (ABD 1.8)
==================================

GUI app that
  1. connects to a Tektronix MSO58B (MSO5 / MSO5B series) over Ethernet or USB,
  2. acquires the MCU AC phase currents (default CH1-CH3) and AC phase
     voltages (default CH6-CH8) and transfers them to the PC,
  3. runs the same harmonic analysis as MCU_Harmonic_Analyzer.m
     (Hann-windowed FFT, ABD 1.8 Table 9-1 current-distortion limits,
     THD, fundamental / 3rd-harmonic phasors, voltage phasors, PASS/FAIL),
     shows the six plots, and writes the same Excel report.

Install (once):
    pip install PySide6 numpy matplotlib openpyxl pillow pyvisa pyvisa-py

Run:
    python mso58b_harmonic_analyzer.py

Interfaces
----------
* Ethernet - raw socket: needs no VISA driver. On the scope enable
  Utility > I/O > Socket Server (port 4000, protocol "None"/raw).
* Ethernet - VISA:  resource TCPIP0::<ip>::inst0::INSTR (VXI-11) or
  TCPIP0::<ip>::hislip0::INSTR. Uses an installed VISA (TekVISA / NI-VISA /
  Keysight) or the pure-Python pyvisa-py backend.
* USB - VISA (USBTMC): resource USB0::0x0699::0x....::<serial>::INSTR.
  On Windows install TekVISA or NI-VISA.
* Simulated scope: generates realistic 400 Hz test waveforms so the app can be
  tried without hardware.

The analysis maths (FFT, window, normalisation, limits) is a line-by-line
port of MCU_Harmonic_Analyzer.m, including MATLAB's round-half-away-from-zero
rule for bin indices, so results match the MATLAB tool.
"""

from __future__ import annotations

import datetime as _dt
import math
import os
import re
import socket
import sys
import threading
import time
import traceback

import numpy as np

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavToolbar,
)

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

APP_NAME = "MSO58B Harmonic Analyzer (ABD 1.8)"
PH = "ABC"
PH_COLORS = [(0, 0.30, 0.85), (0.10, 0.60, 0.10), (0.80, 0.00, 0.80)]  # blue, green, magenta


# ============================================================================
#  Small numeric helpers
# ============================================================================
def mround(x):
    """MATLAB round(): half away from zero (numpy rounds half to even)."""
    a = np.asarray(x, dtype=float)
    r = np.sign(a) * np.floor(np.abs(a) + 0.5)
    if r.ndim == 0:
        return int(r)
    return r.astype(np.int64)


def hann_win(n: int) -> np.ndarray:
    """Symmetric Hann window, same as MATLAB hann(n)."""
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    return 0.5 * (1 - np.cos(2 * np.pi * k / (n - 1)))


def wrap180(a):
    return np.mod(np.asarray(a, dtype=float) + 180, 360) - 180


# ============================================================================
#  ABD 1.8 Table 9-1 current distortion limits
# ============================================================================
#            h   1-phase  3ph<=20kVA  3ph>20kVA      (limit in % of I1)
ABD_TABLE = np.array([
    [2, 0.50, 0.50, 0.50],
    [3, 5.00, 2.00, 2.00],
    [5, 6.00, 2.00, 2.00],
    [7, 4.29, 2.00, 2.00],
    [9, 1.67, 1.11, 1.11],
    [11, 2.73, 10.00, 3.00],
    [13, 2.31, 8.00, 3.00],
    [15, 1.00, 0.67, 0.67],
    [17, 1.76, 4.00, 4.00],
    [19, 1.58, 4.00, 4.00],
    [21, 0.71, 0.48, 0.48],
    [23, 1.30, 3.00, 3.00],
    [25, 1.20, 3.00, 3.00],
    [27, 0.56, 0.37, 0.37],
    [29, 1.03, 1.03, 1.03],
    [31, 0.97, 0.97, 0.97],
    [33, 0.45, 0.30, 0.30],
    [35, 0.86, 0.86, 0.86],
    [37, 0.81, 0.81, 0.81],
    [39, 0.38, 0.26, 0.26],
])
ABD_NAMES = {1: "single-phase equipment",
             2: "three-phase equipment rated 20 kVA or less",
             3: "three-phase equipment rated more than 20 kVA"}


def abd_limits(col: int):
    """col: 1 = single-phase, 2 = three-phase <= 20 kVA, 3 = three-phase > 20 kVA.
    Returns (limit_function(h, I1) -> % of I1, description)."""
    table = {int(r[0]): r[col] for r in ABD_TABLE}

    def lim_fcn(h, I1):
        h = np.asarray(h)
        floor_pct = max(0.25, 100 * 0.010 / I1)  # 0.25 % of I1 or 10 mA, whichever is greater
        v = np.full(h.shape, floor_pct, dtype=float)
        for i, hh in enumerate(h):
            if int(hh) in table:
                v[i] = table[int(hh)]
        v[h == 1] = np.nan  # the fundamental has no limit
        return v

    msg = ("ABD 1.8 Table 9-1, %s (even harmonics 2<h<40 and all h>=40: "
           "0.25%% of I1 or 10 mA, whichever is greater)" % ABD_NAMES[col])
    return lim_fcn, msg


# ============================================================================
#  Analysis (same maths as MCU_Harmonic_Analyzer.m / tomas_fft_a350f.m)
# ============================================================================
METHOD_NAMES = {1: "Single window",
                2: "Peak of sliding windows (50% overlap)",
                3: "Average of sliding windows (50% overlap)"}


def amp_angle(y, Fs, f0, N_hann, w, win_num):
    """Peak amplitude and phase angle [deg] of the f0 component in one Hann window."""
    s = int(mround((win_num - 1) * N_hann))
    seg = y[s:s + N_hann] * w
    seg[~np.isfinite(seg)] = 0
    X = np.fft.fft(seg)
    k = int(mround(f0 / Fs * N_hann))
    mag = 2 * abs(X[k]) / w.sum()
    return mag, math.degrees(np.angle(X[k]))


def analyze(D: dict, o: dict) -> dict:
    t = D["t"]
    I = D["I"]
    V = D.get("V")
    N = len(t)
    Ts = (t[-1] - t[0]) / N  # same definition as the original script
    Fs = 1 / Ts
    f0 = o["f_fund"]
    if Fs < 20 * f0:
        raise ValueError("The sample rate looks too low (%.0f S/s) for a %g Hz fundamental. "
                         "Check that the time column is in seconds." % (Fs, f0))

    N_hann = int(mround(o["hann_dur"] / Ts))
    if N_hann > N:
        raise ValueError("The record is only %.3f s long, which is shorter than the %.3f s Hann "
                         "window. Reduce \"Hann window length\" or capture a longer record."
                         % (N * Ts, o["hann_dur"]))
    w = hann_win(N_hann)
    sw = w.sum()
    nb = int(mround(N_hann / 2)) + 1
    f = Fs * np.arange(nb) / N_hann

    if o["method"] == 1:
        s0 = int(mround((o["win_num"] - 1) * N_hann))
        if s0 + N_hann > N:
            raise ValueError("Window number %d is past the end of the data (maximum is %d for this "
                             "record)." % (o["win_num"], N // N_hann))
        starts = np.array([s0])
        ph_win = o["win_num"]
    else:
        nW = int(np.floor((N - N_hann) / (N_hann / 2))) + 1
        starts = mround(np.arange(nW) * N_hann / 2)
        starts = starts[starts + N_hann <= N]
        ph_win = 1

    nPh = I.shape[1]
    max_harm = int(min(375, np.floor(Fs / f0 / 2)))
    H = np.arange(1, max_harm + 1)

    Arms = np.full((max_harm, 3), np.nan)
    Apu = np.full((max_harm, 3), np.nan)
    spec_pu = np.full((nb, nPh), np.nan)
    fund_rms = np.full(3, np.nan)
    fund_ang = np.full(3, np.nan)
    base = np.full(3, np.nan)
    h3_rms = np.full(3, np.nan)
    h3_ang = np.full(3, np.nan)
    thd = np.full(3, np.nan)

    kF = int(mround(f0 / Fs * N_hann))
    kH = np.minimum(mround(H * f0 / Fs * N_hann), nb - 1)

    for p in range(nPh):
        y = I[:, p]
        Xmag = np.zeros(nb)
        for s in starts:
            seg = y[s:s + N_hann] * w
            seg[~np.isfinite(seg)] = 0
            Xm = 2 * np.abs(np.fft.fft(seg)[:nb]) / sw
            if o["method"] == 3:
                Xmag = Xmag + Xm  # average
            else:
                Xmag = np.maximum(Xmag, Xm)  # peak (or single window)
        if o["method"] == 3:
            Xmag = Xmag / len(starts)
        Xrms = Xmag / np.sqrt(2)

        fund_rms[p] = Xrms[kF]
        base[p] = o["nominal"] if o["use_nominal"] else fund_rms[p]
        spec_pu[:, p] = Xrms / base[p]
        Arms[:, p] = Xrms[kH]
        Apu[:, p] = Arms[:, p] / base[p]
        thd[p] = np.sqrt(np.sum(Arms[1:min(40, max_harm), p] ** 2)) / fund_rms[p]

        _, fund_ang[p] = amp_angle(y, Fs, f0, N_hann, w, ph_win)
        m3, h3_ang[p] = amp_angle(y, Fs, 3 * f0, N_hann, w, ph_win)
        h3_rms[p] = m3 / np.sqrt(2)

    # ABD limit in per-unit. I1 = nominal current when normalising to the MCU
    # rating, otherwise the largest measured fundamental (used for the 10 mA rule).
    I1 = o["nominal"] if o["use_nominal"] else np.nanmax(fund_rms)
    lim_fcn, _ = abd_limits(o["lim_col"])
    lim = lim_fcn(H, I1) / 100

    v_rms = np.full(3, np.nan)
    v_ang = np.full(3, np.nan)
    have_v = V is not None and V.ndim == 2 and V.shape[1] == 3
    if have_v:
        for k in range(3):
            m, v_ang[k] = amp_angle(V[:, k], Fs, f0, N_hann, w, ph_win)
            v_rms[k] = m / np.sqrt(2)

    with np.errstate(invalid="ignore"):
        fail = Apu > lim[:, None]  # NaN never fails
    fail[0, :] = False
    result = np.array(["PASS"] * max_harm, dtype=object)
    result[np.isnan(lim)] = "No limit"
    result[fail.any(axis=1)] = "FAIL"
    result[0] = "Fundamental"
    n_fail = fail.sum(axis=0)

    # Report view: standard harmonics (2, 4, odd 3-59) plus anything failing, up to 75
    keep = np.ones(max_harm, dtype=bool)
    if o["filter"]:
        keep[:] = False
        keep[0] = True
        std = set([2, 4] + list(range(3, 60, 2)))
        for hh in range(2, min(75, max_harm) + 1):
            keep[hh - 1] = (hh in std) or bool(fail[hh - 1, :].any())

    if np.all(np.isnan(lim[1:])):
        overall = "No harmonic limits loaded - PASS/FAIL not evaluated."
    elif n_fail.sum() > 0:
        overall = "FAIL: %d harmonic(s) exceed the ABD 1.8 limit." % int(fail.any(axis=1).sum())
    else:
        overall = "PASS: all harmonics are within the ABD 1.8 limits."

    # Harmonic table (same columns as the MATLAB report)
    cols = ["Harmonic", "Frequency_Hz", "ABD_limit_pu"]
    data = [H, H * f0, lim]
    for p in range(nPh):
        cols.append("Phase%s_Arms" % PH[p]); data.append(Arms[:, p])
    for p in range(nPh):
        cols.append("Phase%s_pu" % PH[p]); data.append(Apu[:, p])
    cols.append("Result"); data.append(result)

    return dict(Fs=Fs, Ts=Ts, N=N, N_hann=N_hann, n_win=len(starts), f=f, spec_pu=spec_pu,
                H=H, lim=lim, Arms=Arms, Apu=Apu, fund_rms=fund_rms, fund_ang=fund_ang,
                base=base, h3_rms=h3_rms, h3_ang=h3_ang, thd=thd, n_fail=n_fail,
                v_rms=v_rms, v_ang=v_ang, have_v=have_v, nPh=nPh,
                table_cols=cols, table_data=data, result=result, fail=fail,
                keep=keep, overall=overall)


# ============================================================================
#  Plots (same six figures as the MATLAB tool)
# ============================================================================
PLOT_NAMES = ["Time_domain", "Spectrum_30kHz", "Spectrum_low_freq",
              "Phasor_fundamental", "Phasor_3rd_harmonic", "Harmonics_vs_limits"]
PLOT_TITLES = ["Time domain", "Spectrum 0-30 kHz", "Spectrum 0-2.2 kHz",
               "Phasor - fundamental", "Phasor - 3rd harmonic", "Harmonics vs ABD 1.8"]


def make_figures(D, R, o):
    nPh = R["nPh"]
    leg = ["Phase %s" % PH[p] for p in range(nPh)]
    if o["use_nominal"]:
        base_txt = "per-unit of %.4g A nominal" % o["nominal"]
    else:
        base_txt = "per-unit of measured fundamental"
    figs = []

    def new_fig(w=9.0, h=5.2):
        fg = Figure(figsize=(w, h), dpi=100, facecolor="w")
        figs.append(fg)
        return fg

    # 1 - Time domain (first 3 periods)
    fg = new_fig()
    ax = fg.add_subplot(111)
    i_max = min(R["N"], int(mround(3 * R["Fs"] / o["f_fund"])))
    for p in range(nPh):
        ax.plot(D["t"][:i_max], D["I"][:i_max, p], color=PH_COLORS[p], lw=1)
    ax.set_title("Input current (first 3 periods)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Current [A]")
    ax.set_xlim(D["t"][0], D["t"][i_max - 1]); ax.grid(True); ax.legend(leg)
    fg.tight_layout()

    # 2 - Spectrum 0-30 kHz
    fg = new_fig()
    ax = fg.add_subplot(111)
    for p in range(nPh):
        ax.plot(R["f"], R["spec_pu"][:, p], color=PH_COLORS[p], lw=1)
    ax.set_title("Single-sided amplitude spectrum with Hann window")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Magnitude [%s]" % base_txt)
    ax.set_xlim(0, 30e3); ax.grid(True); ax.legend(leg)
    fg.tight_layout()

    # 3 - Spectrum 0-2.2 kHz zoom
    fg = new_fig()
    ax = fg.add_subplot(111)
    for p in range(nPh):
        ax.plot(R["f"], R["spec_pu"][:, p], color=PH_COLORS[p], lw=1)
    ax.axhline(0.0025, ls="--", color="k", lw=1)
    ax.text(2.18e3, 0.0025, "0.25 %", ha="right", va="bottom")
    ax.set_title("Single-sided amplitude spectrum with Hann window (zoom)")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Magnitude [%s]" % base_txt)
    ax.set_xlim(0, 2.2e3); ax.set_ylim(0, 0.03); ax.grid(True); ax.legend(leg)
    fg.tight_layout()

    ANG, DEG = "∠", "°"
    th = np.linspace(0, 2 * np.pi, 200)

    # 4 - Phasor plot of fundamental
    fg = new_fig(9.0, 7.0)
    ax = fg.add_subplot(111)
    for r in range(2, 21):
        ax.plot(r * np.cos(th), r * np.sin(th), color=(0.6, 0.6, 0.6), lw=0.6)
    for k in range(3):
        if R["have_v"]:
            L, a = R["v_rms"][k] / 10, R["v_ang"][k]
        else:
            L, a = 20, R["fund_ang"][0] - k * 120
        ax.plot([0, L * np.cos(np.radians(a))], [0, L * np.sin(np.radians(a))], "-.",
                lw=1, color=PH_COLORS[k])
    for p in range(nPh):
        r_, a_ = R["fund_rms"][p], np.radians(R["fund_ang"][p])
        ax.plot([0, r_ * np.cos(a_)], [0, r_ * np.sin(a_)], lw=3, color=PH_COLORS[p])
    tx = lambda x, y, s: ax.text(x, y, s, fontsize=9)  # noqa: E731
    for p in range(nPh):
        tx(13, -14 - 2 * p, "I%d: %.2f A" % (p + 1, R["fund_rms"][p]))
    fa = R["fund_ang"]
    if nPh == 3:
        tx(13, 18, "%sI1-I2: %.2f %s" % (ANG, wrap180(fa[0] - fa[1]), DEG))
        tx(13, 16, "%sI2-I3: %.2f %s" % (ANG, wrap180(fa[1] - fa[2]), DEG))
        tx(13, 14, "%sI1-I3: %.2f %s" % (ANG, wrap180(fa[0] - fa[2]), DEG))
    if R["have_v"]:
        va, vr = R["v_ang"], R["v_rms"]
        for k in range(3):
            tx(-24, -14 - 2 * k, "V%d: %.2f V" % (k + 1, vr[k]))
        tx(13, 10, "%sV1-V2: %.2f %s" % (ANG, wrap180(va[0] - va[1]), DEG))
        tx(13, 8, "%sV2-V3: %.2f %s" % (ANG, wrap180(va[1] - va[2]), DEG))
        tx(13, 6, "%sV1-V3: %.2f %s" % (ANG, wrap180(va[0] - va[2]), DEG))
        for p in range(nPh):
            tx(-24, 18 - 2 * p, "%sV%d-I%d: %.2f %s" % (ANG, p + 1, p + 1,
                                                       wrap180(va[p] - fa[p]), DEG))
        note = "Solid: current [A rms]   Dash-dot: voltage [V rms / 10]"
    else:
        note = "Solid: current [A rms]   Dash-dot: ideal 120 deg reference"
    tx(-24, -20.5, note)
    ax.set_aspect("equal"); ax.set_xlim(-25, 25); ax.set_ylim(-22, 21)
    ax.set_title("Phasor plot of fundamental current")
    fg.tight_layout()

    # 5 - Phasor plot of 3rd harmonic
    fg = new_fig(9.0, 7.0)
    ax = fg.add_subplot(111)
    rmax = np.nanmax(R["h3_rms"][:nPh])
    if not rmax > 0:
        rmax = 1.0
    for r in rmax * np.arange(0.25, 1.26, 0.25):
        ax.plot(r * np.cos(th), r * np.sin(th), color=(0.7, 0.7, 0.7), lw=0.6)
    for p in range(nPh):
        r_, a_ = R["h3_rms"][p], R["h3_ang"][p]
        ax.plot([0, r_ * np.cos(np.radians(a_))], [0, r_ * np.sin(np.radians(a_))],
                lw=3, color=PH_COLORS[p])
        ax.text(-1.45 * rmax, (1.35 - 0.15 * p) * rmax,
                "I%d (3rd): %.4f A  %s %.1f%s" % (p + 1, r_, ANG, wrap180(a_), DEG), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-1.5 * rmax, 1.5 * rmax); ax.set_ylim(-1.45 * rmax, 1.45 * rmax)
    ax.set_title("Phasor plot of current (3rd harmonic, A rms)")
    fg.tight_layout()

    # 6 - Harmonics vs ABD limits (grouped bars, limit as green outline)
    fg = new_fig(15.0, 5.6)
    ax = fg.add_subplot(111)
    h_all = np.arange(2, min(75, len(R["H"])) + 1)
    is_fail = R["result"][h_all - 1] == "FAIL"
    show = np.isin(h_all, [2, 4] + list(range(3, 60, 2)) + [61, 65, 67, 71]) | is_fail
    hh = h_all[show]
    fail_shown = is_fail[show]
    Y = R["Apu"][hh - 1, :nPh]
    x = np.arange(1, len(hh) + 1)
    bar_cols = [(0.27, 0.45, 0.85), (0.92, 0.25, 0.20), (0.98, 0.75, 0.10)]
    group_w = 0.8
    bw = group_w / nPh
    bars = []
    for p in range(nPh):
        off = -group_w / 2 + bw * (p + 0.5)
        bars.append(ax.bar(x + off, Y[:, p], width=bw * 0.95, color=bar_cols[p],
                           edgecolor=(0.25, 0.25, 0.25), linewidth=0.5))
    lim_color = (0.10, 0.55, 0.15)
    bl = ax.bar(x, R["lim"][hh - 1], width=0.92, facecolor="none",
                edgecolor=lim_color, linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in hh])
    for lab, fl in zip(ax.get_xticklabels(), fail_shown):
        if fl:
            lab.set_color("red")
    ax.set_xlim(0.4, len(hh) + 0.6)
    y_max = np.nanmax(np.concatenate([Y.ravel(), R["lim"][hh - 1]]))
    if not y_max > 0:
        y_max = 0.01
    step = 0.01 if y_max <= 0.05 else 0.02
    y_top = math.ceil(y_max * 1.15 / step) * step
    ticks = np.arange(0, y_top + step / 2, step)
    ax.set_ylim(0, y_top); ax.set_yticks(ticks)
    ax2 = ax.twinx()
    ax2.set_ylim(0, y_top); ax2.set_yticks(ticks)
    ax.set_ylabel("Limit (I$_h$) in pu of I$_1$   (0.01 = 1 %)")
    ax.yaxis.grid(True, color=(0.5, 0.5, 0.5), alpha=0.5)
    ax.set_axisbelow(True)
    for a in (ax, ax2):
        for sp in ("top",):
            a.spines[sp].set_visible(False)
    ax.set_xlabel("Harmonic order")
    ax.set_title("ZOOM - Comparison phase A/B/C vs. ABD 1.8  (%s)" % o["mcu_name"],
                 fontsize=15, color=(0.35, 0.35, 0.35))
    ax.legend([b for b in bars] + [bl], leg + ["ABD 1.8 limit"],
              loc="center left", bbox_to_anchor=(1.04, 0.5), frameon=False)
    fg.tight_layout()

    return figs


def export_pngs(figs, folder):
    os.makedirs(folder, exist_ok=True)
    out = []
    for k, (fg, nm) in enumerate(zip(figs, PLOT_NAMES), start=1):
        p = os.path.join(folder, "%d_%s.png" % (k, nm))
        fg.savefig(p, dpi=120)
        out.append(p)
    return out


# ============================================================================
#  Excel report (same sheets as the MATLAB tool)
# ============================================================================
def write_report(xlsx, D, R, o, figs):
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    nPh = R["nPh"]
    bold = Font(bold=True)
    norm_txt = ("MCU nominal current (%.4g A rms)" % o["nominal"]) if o["use_nominal"] \
        else "Measured fundamental current"

    info = [
        ["MCU harmonic analysis report", ""],
        ["Created", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Data source", D.get("desc", "")],
        ["File format detected", D.get("format", "")],
        ["MCU type", o["mcu_name"]],
        ["Nominal current [A rms]", o["nominal"]],
        ["Harmonics expressed relative to", norm_txt],
        ["AC fundamental [Hz]", o["f_fund"]],
        ["Sample rate [S/s]", R["Fs"]],
        ["Record length [s]", R["N"] * R["Ts"]],
        ["Hann window [s]", o["hann_dur"]],
        ["FFT method", METHOD_NAMES[o["method"]]],
        ["Number of windows used", R["n_win"]],
        ["Current polarity inverted", "Yes" if o["invert"] else "No"],
        ["Harmonic limits", o["lim_msg"]],
        ["OVERALL RESULT", R["overall"]],
    ]
    summary = [list(r) for r in info]
    bold_rows = [1, len(info)]
    summary += [[], []]
    bold_rows.append(len(summary) + 1)
    summary.append(["Phase", "Fundamental_Arms", "Fundamental_angle_deg", "THD_2to40_pct",
                    "Third_harmonic_Arms", "Harmonics_over_limit"])
    for p in range(nPh):
        summary.append([PH[p], R["fund_rms"][p], R["fund_ang"][p], 100 * R["thd"][p],
                        R["h3_rms"][p], int(R["n_fail"][p])])
    if R["have_v"]:
        summary += [[], []]
        bold_rows.append(len(summary) + 1)
        summary.append(["Phase", "Fundamental_Vrms", "Fundamental_angle_deg",
                        "V_minus_I_angle_deg"])
        for p in range(3):
            summary.append([PH[p], R["v_rms"][p], R["v_ang"][p],
                            float(wrap180(R["v_ang"][p] - R["fund_ang"][p]))])

    def table_rows(mask=None):
        cols, data = R["table_cols"], R["table_data"]
        n = len(data[0])
        idx = np.arange(n) if mask is None else np.flatnonzero(mask)
        rows = [list(cols)]
        for i in idx:
            rows.append([c[i] for c in data])
        return rows

    sel = R["f"] <= 30e3
    spec = [["Frequency_Hz"] + ["Phase%s_pu" % PH[p] for p in range(nPh)]]
    spec += np.column_stack([R["f"][sel], R["spec_pu"][sel, :]]).tolist()

    base, _ = os.path.splitext(xlsx)
    png_dir = base + "_plots"
    pngs = export_pngs(figs, png_dir)

    wb = Workbook()
    wb.remove(wb.active)

    def put_sheet(name, rows, bold_rows_):
        ws = wb.create_sheet(name)
        widths = {}
        for r, row in enumerate(rows, start=1):
            for c, v in enumerate(row, start=1):
                if v is None or (isinstance(v, str) and v == ""):
                    continue
                if isinstance(v, (float, np.floating)):
                    if not np.isfinite(v):
                        continue
                    v = float(v)
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                cell = ws.cell(row=r, column=c, value=v)
                if r in bold_rows_:
                    cell.font = bold
                wlen = min(len(v), 60) + 2 if isinstance(v, str) else 14
                widths[c] = max(widths.get(c, 8), wlen)
        for c, wv in widths.items():
            ws.column_dimensions[get_column_letter(c)].width = wv
        return ws

    put_sheet("Summary", summary, set(bold_rows))
    put_sheet("Harmonics", table_rows(R["keep"]), {1})
    put_sheet("All harmonics", table_rows(), {1})
    put_sheet("Spectrum", spec, {1})
    ws = put_sheet("Plots", [["Plots (also saved as PNG files in this folder):"], [png_dir]], {1})
    row = 4
    for p in pngs:
        img = XLImage(p)
        w_px = 1000
        h_px = int(round(w_px * img.height / img.width))
        img.width, img.height = w_px, h_px
        ws.add_image(img, "A%d" % row)
        row += math.ceil(h_px / 20) + 2
    wb.save(xlsx)
    return "Spreadsheet written with %d plots on the \"Plots\" sheet (PNGs in %s)." % (
        len(pngs), png_dir)


# ============================================================================
#  Data files (load existing scope CSV exports, save captured data)
# ============================================================================
def _split_fields(ln):
    parts = [p.strip() for p in re.split(r"[,;\t]", ln.rstrip("\r\n"))]
    return [re.sub(r'^"|"$', "", p) for p in parts]


def _is_num(s):
    try:
        return math.isfinite(float(s))
    except ValueError:
        return False


def _pick_delim(line):
    if "," in line:
        return ","
    if ";" in line:
        return ";"
    return "\t"


def _cdaq_cols(names, kind):
    idx = []
    for ph in PH:
        k = next((i for i, n in enumerate(names)
                  if re.search("M1ACIN_PH%s_%s" % (ph, kind), n, re.I)), None)
        if k is None:
            k = next((i for i, n in enumerate(names)
                      if re.search("_PH%s_%s" % (ph, kind), n, re.I)), None)
        if k is not None:
            idx.append(k)
    return idx


def read_scope_csv(path):
    """Port of readScopeCsv from the MATLAB tool: Tektronix / CDAQ / power analyser /
    generic scope CSV export, any number of header lines, first column = time."""
    nm = os.path.basename(path)
    hdr = []
    first = None
    with open(path, "r", errors="replace") as fh:
        n_head = 0
        while n_head < 1000:
            ln = fh.readline()
            if not ln:
                break
            parts = _split_fields(ln)
            is_num = [_is_num(p) for p in parts]
            is_data = len(parts) >= 2 and is_num[0] and is_num[1]
            if not is_data and len(parts) >= 3 and hdr and len(hdr[-1]) == len(parts):
                is_data = sum(is_num) >= max(2, len(parts) - 2)
            if is_data:
                first = ln
                break
            n_head += 1
            hdr.append(parts)
    if first is None:
        raise ValueError("No numeric data found in the first 1000 lines of %s." % nm)
    delim = _pick_delim(first)
    first_parts = _split_fields(first)
    n_all = len(first_parts)
    num_idx = [c for c in range(n_all) if _is_num(first_parts[c])]

    try:
        M = np.loadtxt(path, delimiter=delim, skiprows=n_head, usecols=num_idx, ndmin=2,
                       comments=None)
    except ValueError:
        rows = []
        with open(path, "r", errors="replace") as fh:
            for _ in range(n_head):
                fh.readline()
            for ln in fh:
                parts = ln.rstrip("\r\n").split(delim)
                try:
                    rows.append([float(parts[c]) for c in num_idx])
                except (ValueError, IndexError):
                    break
        M = np.array(rows, dtype=float)
    if M.shape[0] < 100:
        raise ValueError("Only %d data rows could be read from %s." % (M.shape[0], nm))

    names = [""] * n_all
    label_row = []
    for r in hdr:
        if r and r[0].lower() == "labels":
            label_row = r
    last_row = hdr[-1] if hdr else []
    for c in range(n_all):
        if c < len(label_row) and label_row[c]:
            names[c] = label_row[c]
        elif c < len(last_row):
            names[c] = last_row[c]
    names = [names[c] for c in num_idx]
    lname = [n.lower() for n in names]
    n_col = len(names)
    is_cur = [i for i, n in enumerate(lname) if "current" in n]
    is_vol = [i for i, n in enumerate(lname) if "voltage" in n]
    find = lambda key: next((i for i, n in enumerate(names) if n.lower() == key.lower()), None)  # noqa
    i_trace = find("TraceName")
    iI = [i for i in (find("I1"), find("I2"), find("I3")) if i is not None]
    iU = [i for i in (find("U1"), find("U2"), find("U3")) if i is not None]
    iCdI = _cdaq_cols(names, "ISENSE")
    iCdV = _cdaq_cols(names, "VSENSE")
    i_time = find("Time_s")

    V = None
    if iCdI:
        t = M[:, i_time if i_time is not None else 0]
        I = M[:, iCdI]
        if len(iCdV) == 3:
            V = M[:, iCdV]
        fmt = "CDAQ export: currents %s; voltages %s" % (
            ", ".join(names[i] for i in iCdI), ", ".join(names[i] for i in iCdV))
    elif is_cur:
        t = M[:, 0]
        I = M[:, is_cur[:3]]
        if len(is_vol) >= 3:
            V = M[:, is_vol[:3]]
        fmt = "Tektronix export (labelled PhaseA/B/C current channels)"
    elif i_trace is not None and iI:
        t = M[:, i_trace] * 10e-9
        I = M[:, iI]
        if len(iU) == 3:
            V = M[:, iU]
        fmt = "Power analyser export (I1-I3, U1-U3); time = sample no. x 10 ns"
    else:
        t = M[:, 0]
        I = M[:, 1:min(4, n_col)]
        if n_col >= 7:
            V = M[:, 4:7]
            fmt = ("Tektronix export: column 1 = time, 2-4 = phase A/B/C current, "
                   "5-7 = phase A/B/C voltage")
        elif n_col == 2:
            fmt = "Scope export: column 1 = time, column 2 = current"
        else:
            fmt = "Scope export: column 1 = time, columns 2-%d = phase A/B/C current" % min(4, n_col)
    return t, I, V, fmt


def tidy_data(t, I, V, fmt, desc, base_name):
    I = np.asarray(I, dtype=float)
    if I.ndim == 1:
        I = I[:, None]
    I = I[:, :3]
    if V is not None:
        V = np.asarray(V, dtype=float)[:, :3]
    good = np.isfinite(t) & np.isfinite(I).any(axis=1)
    idx = np.flatnonzero(good)
    if idx.size == 0:
        raise ValueError("No numeric data found.")
    i1, i2 = idx[0], idx[-1] + 1
    D = dict(t=np.asarray(t[i1:i2], dtype=float), I=I[i1:i2],
             V=None if V is None else V[i1:i2], format=fmt, desc=desc, base_name=base_name)
    if D["t"][-1] <= D["t"][0]:
        raise ValueError("The time column is not increasing - check the file format.")
    return D


def load_data_file(path):
    nm, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() == ".npz":
        z = np.load(path, allow_pickle=False)
        V = z["V"] if "V" in z.files and z["V"].size else None
        fmt = str(z["format"]) if "format" in z.files else "Saved capture (.npz)"
        return tidy_data(z["t"], z["I"], V, fmt, path, nm)
    t, I, V, fmt = read_scope_csv(path)
    return tidy_data(t, I, V, fmt, path, nm)


def save_data_file(path, D):
    ext = os.path.splitext(path)[1].lower()
    V = D["V"]
    if ext == ".npz":
        np.savez(path, t=D["t"], I=D["I"], V=V if V is not None else np.zeros((0, 0)),
                 format=D.get("format", ""), desc=D.get("desc", ""))
        return
    cols = ["TIME"] + ["Phase%s_Current" % PH[p] for p in range(D["I"].shape[1])]
    M = [D["t"][:, None], D["I"]]
    if V is not None:
        cols += ["Phase%s_Voltage" % PH[p] for p in range(V.shape[1])]
        M.append(V)
    M = np.hstack(M)
    with open(path, "w", newline="") as fh:
        fh.write("Source,%s\n" % D.get("desc", "").replace(",", ";"))
        fh.write(",".join(cols) + "\n")
        fmt = ["%.10g"] + ["%.6g"] * (M.shape[1] - 1)
        np.savetxt(fh, M, fmt=fmt, delimiter=",")


# ============================================================================
#  Scope communication
# ============================================================================
class ScopeError(RuntimeError):
    pass


class SocketTransport:
    """Raw TCP socket (Tek socket server, default port 4000). No VISA needed."""

    def __init__(self, host, port, timeout):
        self.desc = "socket %s:%d" % (host, port)
        try:
            self.sock = socket.create_connection((host, int(port)), timeout=10)
        except OSError as e:
            raise ScopeError("Cannot connect to %s:%d (%s). Check the IP address and that the "
                             "scope's Socket Server is enabled (Utility > I/O)." % (host, port, e))
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.buf = bytearray()
        self.timeout = timeout

    def _fill(self):
        try:
            chunk = self.sock.recv(1 << 20)
        except socket.timeout:
            raise ScopeError("Timeout waiting for the scope (%g s)." % self.timeout)
        if not chunk:
            raise ScopeError("The scope closed the connection.")
        self.buf += chunk

    def clear(self):
        self.buf.clear()
        self.sock.setblocking(False)
        try:
            while True:
                if not self.sock.recv(1 << 20):
                    break
        except (BlockingIOError, OSError):
            pass
        finally:
            self.sock.settimeout(self.timeout)

    def write(self, cmd):
        self.sock.sendall((cmd + "\n").encode("ascii"))

    def _read_line(self):
        while b"\n" not in self.buf:
            self._fill()
        i = self.buf.index(b"\n")
        line = bytes(self.buf[:i])
        del self.buf[:i + 1]
        return line.decode("ascii", errors="replace").strip()

    def query(self, cmd):
        self.clear()
        self.write(cmd)
        return self._read_line()

    def _read_n(self, n):
        while len(self.buf) < n:
            self._fill()
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def read_block_int16(self, progress=None):
        skipped = 0
        while True:
            b = self._read_n(1)
            if b == b"#":
                break
            skipped += 1
            if skipped > 64:
                raise ScopeError("Unexpected reply from the scope (no binary block header).")
        nd = int(self._read_n(1))
        if nd == 0:
            raise ScopeError("Indefinite-length binary blocks are not supported.")
        L = int(self._read_n(nd))
        data = bytearray()
        while len(data) < L:
            if not self.buf:
                self._fill()
            take = min(L - len(data), len(self.buf))
            data += self.buf[:take]
            del self.buf[:take]
            if progress:
                progress(len(data) / L)
        # trailing line feed (if it has arrived) is dropped by the next clear()
        return np.frombuffer(bytes(data), dtype="<i2")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class VisaTransport:
    """PyVISA (USBTMC, VXI-11, HiSLIP, or VISA SOCKET resources)."""

    def __init__(self, resource, timeout, backend=""):
        try:
            import pyvisa
        except ImportError:
            raise ScopeError("PyVISA is not installed. Run:  pip install pyvisa pyvisa-py")
        try:
            self.rm = pyvisa.ResourceManager(backend)
        except Exception as e:  # noqa: BLE001
            raise ScopeError("Could not start the VISA library (%s). Install TekVISA / NI-VISA, "
                             "or choose the pyvisa-py backend." % e)
        try:
            self.inst = self.rm.open_resource(resource)
        except Exception as e:  # noqa: BLE001
            raise ScopeError("Cannot open VISA resource %s (%s)." % (resource, e))
        self.inst.timeout = int(timeout * 1000)
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        try:
            self.inst.chunk_size = 1 << 20
        except Exception:  # noqa: BLE001
            pass
        self.desc = "VISA %s" % resource

    def clear(self):
        try:
            self.inst.clear()
        except Exception:  # noqa: BLE001
            pass

    def write(self, cmd):
        self.inst.write(cmd)

    def query(self, cmd):
        try:
            return self.inst.query(cmd).strip()
        except Exception as e:  # noqa: BLE001
            raise ScopeError("No reply to \"%s\" (%s)." % (cmd, e))

    def read_block_int16(self, progress=None):
        try:
            y = self.inst.read_binary_values(datatype="h", is_big_endian=False,
                                             container=np.array, expect_termination=True)
        except Exception as e:  # noqa: BLE001
            raise ScopeError("Waveform transfer failed (%s)." % e)
        if progress:
            progress(1.0)
        return np.asarray(y, dtype=np.int16)

    def close(self):
        try:
            self.inst.close()
        except Exception:  # noqa: BLE001
            pass


class SimTransport:
    """Simulated MSO58B: answers the SCPI commands this app uses with 400 Hz
    three-phase test waveforms (currents with harmonics, 115 V rms voltages)."""

    def __init__(self):
        self.desc = "simulated scope"
        self.fs = 2.5e6
        self.rl = 1_250_000
        self.src = "CH1"
        self.pending = None
        self.rng = np.random.default_rng()

    def clear(self):
        pass

    def write(self, cmd):
        c = cmd.strip().upper()
        if c.startswith("HORIZONTAL:MODE:SAMPLERATE"):
            self.fs = float(c.split()[-1])
        elif c.startswith("HORIZONTAL:MODE:RECORDLENGTH"):
            self.rl = int(float(c.split()[-1]))
        elif c.startswith("DATA:SOURCE"):
            self.src = c.split()[-1]
        elif c == "CURVE?":
            self.pending = self._wave(self.src)

    def _wave(self, ch):
        n = int(min(self.rl, 5_000_000))
        t = np.arange(n) / self.fs
        k = int(ch[2:]) if ch[2:].isdigit() else 1
        f0 = 400.0
        if k in (1, 2, 3):
            ph = -(k - 1) * 120.0
            I1 = 10.0 * np.sqrt(2)
            y = I1 * np.sin(2 * np.pi * f0 * t + np.radians(ph - 8))
            for h, a, extra in ((3, 0.004, 0), (5, 0.012, 15), (7, 0.008, 40), (11, 0.025, 10),
                                (13, 0.015, 70), (17, 0.006, 0), (19, 0.005, 30), (23, 0.004, 0),
                                (2, 0.0008, 0)):
                y += a * I1 * np.sin(2 * np.pi * h * f0 * t + np.radians(h * (ph - 8) + extra))
            y += self.rng.normal(0, 0.01, n)
            ymult = 50 / 32768
        else:
            ph = -(k - 6) * 120.0
            y = 115 * np.sqrt(2) * np.sin(2 * np.pi * f0 * t + np.radians(ph))
            y += 0.8 * np.sin(2 * np.pi * 5 * f0 * t) + self.rng.normal(0, 0.05, n)
            ymult = 250 / 32768
        self._ymult = ymult
        return np.clip(np.round(y / ymult), -32768, 32767).astype("<i2")

    def query(self, cmd):
        c = cmd.strip().upper()
        if c == "*IDN?":
            return "TEKTRONIX,MSO58B,SIMULATED,CF:91.1CT FV:2.4.6"
        if c in ("*OPC?",):
            return "1"
        if c in ("*ESR?",):
            return "0"
        if c == "ACQUIRE:STATE?":
            time.sleep(0.2)
            return "0"
        if c.startswith("DISPLAY:GLOBAL"):
            return "1"
        if c == "HORIZONTAL:SAMPLERATE?":
            return "%g" % self.fs
        if c == "HORIZONTAL:RECORDLENGTH?":
            return "%d" % self.rl
        if c.startswith("WFMOUTPRE:"):
            self._wave_meta()
            key = c.split(":")[1].rstrip("?")
            return {"XINCR": "%.6E" % (1 / self.fs), "XZERO": "%.6E" % (-self.rl / self.fs / 2),
                    "PT_OFF": "0", "YMULT": "%.6E" % self._ymult, "YOFF": "0", "YZERO": "0",
                    "YUNIT": '"A"' if self.src in ("CH1", "CH2", "CH3") else '"V"',
                    "NR_PT": "%d" % int(min(self.rl, 5_000_000))}.get(key, "0")
        if c == "ALLEV?":
            return '0,"No events to report - queue empty"'
        return "0"

    def _wave_meta(self):
        k = int(self.src[2:]) if self.src[2:].isdigit() else 1
        self._ymult = 50 / 32768 if k in (1, 2, 3) else 250 / 32768

    def read_block_int16(self, progress=None):
        y = self.pending
        self.pending = None
        if y is None:
            raise ScopeError("No waveform requested.")
        time.sleep(0.3)
        if progress:
            progress(1.0)
        return y

    def close(self):
        pass


class Cancelled(Exception):
    pass


def _q_num(tr, cmd):
    r = tr.query(cmd)
    try:
        return float(r.split()[-1].strip('"'))
    except (ValueError, IndexError):
        raise ScopeError("Unexpected reply to \"%s\": %r" % (cmd, r))


def acquire(tr, cfg, log, progress, cancel: threading.Event):
    """Acquire and transfer the configured channels. Returns a data dict D."""
    tr.clear()
    tr.write("HEADer OFF")
    tr.write("VERBose ON")
    tr.write("*CLS")
    chans = [("I", p, ch, sc) for p, (ch, sc) in enumerate(cfg["cur"]) if ch]
    if cfg["use_v"]:
        chans += [("V", p, ch, sc) for p, (ch, sc) in enumerate(cfg["vol"])]
    if not chans:
        raise ScopeError("No channels selected.")

    # Make sure the channels are on
    for _, _, ch, _ in chans:
        if cfg["mode"] == 1:
            tr.write("DISplay:GLObal:%s:STATE ON" % ch)
        else:
            st = _q_num(tr, "DISplay:GLObal:%s:STATE?" % ch)
            if st == 0:
                raise ScopeError("%s is not displayed on the scope, so it has no waveform. "
                                 "Turn it on, or use 'Single acquisition'." % ch)

    if cfg["apply_h"]:
        log("Setting horizontal: %.4g S/s, %d points" % (cfg["fs"], cfg["rl"]))
        tr.write("HORizontal:MODe MANual")
        tr.write("HORizontal:MODe:SAMPLERate %.6g" % cfg["fs"])
        tr.write("HORizontal:MODe:RECOrdlength %d" % cfg["rl"])
        tr.query("*OPC?")

    if cfg["mode"] == 1:
        progress(0, "Waiting for trigger (single acquisition)...")
        tr.write("ACQuire:STOPAfter SEQuence")
        tr.write("ACQuire:STATE RUN")
        t0 = time.time()
        time.sleep(0.1)
        while True:
            if _q_num(tr, "ACQuire:STATE?") == 0:
                break
            if cancel.is_set():
                tr.write("ACQuire:STATE STOP")
                raise Cancelled()
            if time.time() - t0 > cfg["trig_timeout"]:
                tr.write("ACQuire:STATE STOP")
                raise ScopeError("No trigger within %g s. Set the trigger mode to Auto or check "
                                 "the trigger source/level." % cfg["trig_timeout"])
            time.sleep(0.2)
        log("Acquisition complete (%.1f s)." % (time.time() - t0))
    else:
        tr.write("ACQuire:STATE STOP")
        tr.query("*OPC?")
        log("Scope stopped; transferring the waveforms on screen.")

    fs = _q_num(tr, "HORizontal:SAMPLERate?")
    rl = int(_q_num(tr, "HORizontal:RECOrdlength?"))
    log("Scope: %.4g S/s, record length %d points (%.3f s)" % (fs, rl, rl / fs))

    tr.write("DATa:ENCdg SRIbinary")
    tr.write("WFMOutpre:BYT_Nr 2")
    tr.write("DATa:STARt 1")
    tr.write("DATa:STOP %d" % rl)

    t = None
    I_cols, V_cols = {}, {}
    n_ch = len(chans)
    for j, (kind, p, ch, scale) in enumerate(chans):
        if cancel.is_set():
            raise Cancelled()
        what = "Phase %s %s" % (PH[p], "current" if kind == "I" else "voltage")
        msg = "Transferring %s (%s), %d of %d..." % (ch, what, j + 1, n_ch)
        progress(int(100 * j / n_ch), msg)
        tr.write("DATa:SOUrce %s" % ch)
        xinc = _q_num(tr, "WFMOutpre:XINcr?")
        xzero = _q_num(tr, "WFMOutpre:XZEro?")
        ptoff = _q_num(tr, "WFMOutpre:PT_Off?")
        ymult = _q_num(tr, "WFMOutpre:YMUlt?")
        yoff = _q_num(tr, "WFMOutpre:YOFf?")
        yzero = _q_num(tr, "WFMOutpre:YZEro?")
        yunit = tr.query("WFMOutpre:YUNit?").strip().strip('"')
        tr.write("CURVe?")
        raw = tr.read_block_int16(
            lambda fr, j=j, msg=msg: progress(int(100 * (j + fr) / n_ch), msg))
        y = ((raw.astype(float) - yoff) * ymult + yzero) * scale
        if kind == "I" and yunit.upper() != "A" and scale == 1:
            log("WARNING: %s vertical unit is \"%s\", not A. Set the probe so the scope reads "
                "amps, or enter a scale factor (A per %s)." % (ch, yunit, yunit))
        if kind == "V" and yunit.upper() != "V" and scale == 1:
            log("WARNING: %s vertical unit is \"%s\", not V." % (ch, yunit))
        if t is None:
            t = xzero + (np.arange(len(y)) - ptoff) * xinc
        (I_cols if kind == "I" else V_cols)[p] = y
        log("  %s -> %s: %d points, unit %s, scale x%g" % (ch, what, len(y), yunit, scale))
    progress(100, "Finishing...")

    n = min([len(t)] + [len(v) for v in I_cols.values()] + [len(v) for v in V_cols.values()])
    I = np.column_stack([I_cols[p][:n] for p in sorted(I_cols)])
    V = np.column_stack([V_cols[p][:n] for p in sorted(V_cols)]) if len(V_cols) == 3 else None

    try:
        esr = int(_q_num(tr, "*ESR?"))
        if esr & (4 | 8 | 16 | 32):
            log("Scope reported: %s" % tr.query("ALLEv?"))
    except ScopeError:
        pass

    if cfg["restart"]:
        tr.write("ACQuire:STOPAfter RUNSTop")
        tr.write("ACQuire:STATE RUN")

    stamp = _dt.datetime.now()
    cur_txt = ", ".join(ch for ch, _ in cfg["cur"] if ch)
    vol_txt = ", ".join(ch for ch, _ in cfg["vol"]) if V is not None else "none"
    fmt = "Tektronix %s live capture: currents %s; voltages %s" % (
        cfg.get("model", "MSO58B"), cur_txt, vol_txt)
    desc = "%s via %s, captured %s" % (cfg.get("idn", "Tektronix scope"), tr.desc,
                                       stamp.strftime("%Y-%m-%d %H:%M:%S"))
    return tidy_data(t[:n], I, V, fmt, desc, "MSO58B_%s" % stamp.strftime("%Y%m%d_%H%M%S"))


# ============================================================================
#  GUI
# ============================================================================
class Task(QtCore.QThread):
    """Runs fn(log, progress, cancel) in a background thread."""
    log = QtCore.Signal(str)
    prog = QtCore.Signal(int, str)
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            res = self.fn(self.log.emit, lambda v, m="": self.prog.emit(int(v), m), self.cancel)
            self.done.emit(res)
        except Cancelled:
            self.failed.emit("Cancelled.")
        except ScopeError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.failed.emit("%s: %s" % (type(e).__name__, e))


def _form_row(form, label, widget, tip=None):
    if tip:
        widget.setToolTip(tip)
    form.addRow(label, widget)
    return widget


HELP_HTML = """
<h3>Connecting to the MSO58B</h3>
<p><b>Ethernet - raw socket</b> (no driver needed)<br>
On the scope: <i>Utility &gt; I/O &gt; Socket Server</i>, set it <b>On</b>, port <b>4000</b>,
protocol <b>None</b> (not Terminal). Read the scope's IP address under
<i>Utility &gt; I/O &gt; LAN</i> and enter it here.</p>
<p><b>Ethernet - VISA</b><br>
Uses <code>TCPIP0::&lt;ip&gt;::inst0::INSTR</code> (VXI-11) by default; HiSLIP
(<code>TCPIP0::&lt;ip&gt;::hislip0::INSTR</code>) is faster if your VISA supports it.</p>
<p><b>USB - VISA</b><br>
Connect the scope's rear <b>USB Device</b> port (type B) to the PC. On Windows install
<b>TekVISA</b> or <b>NI-VISA</b> and keep "System VISA" selected. Click <i>Scan</i> to
list instruments; the MSO58B shows as <code>USB0::0x0699::...::INSTR</code>.<br>
The pure-Python <i>pyvisa-py</i> backend also works for LAN, and for USB if
<code>pyusb</code> + libusb are installed.</p>
<p><b>Simulated scope</b> generates 400 Hz test waveforms so you can try the app without hardware.</p>
<h3>Channels</h3>
<p>Default mapping: CH1-CH3 = phase A/B/C current, CH6-CH8 = phase A/B/C voltage. Current
probes should be set on the scope so the channel reads in <b>A</b>; otherwise enter a scale
factor (e.g. A per V of a current transducer) on the Acquire tab.</p>
<h3>Record length</h3>
<p>The analysis needs at least one 0.2 s Hann window (5 Hz resolution). For the sliding-window
methods capture longer records. 2.5 MS/s x 1.25 Mpts = 0.5 s is a good default
(harmonics to the 375th of 400 Hz need &ge; 300 kS/s).</p>
"""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1320, 900)
        self.settings = QtCore.QSettings("MCUHarmonic", "MSO58BApp")
        self.tr_ = None  # transport
        self.idn = ""
        self.D = None
        self.R = None
        self.o = None
        self.D_used = None
        self.figs = []
        self.task = None

        split = QtWidgets.QSplitter(Qt.Vertical)
        self.tabs = QtWidgets.QTabWidget()
        split.addWidget(self.tabs)
        self.logw = QtWidgets.QPlainTextEdit(readOnly=True)
        self.logw.setMaximumBlockCount(5000)
        f = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.logw.setFont(f)
        split.addWidget(self.logw)
        split.setSizes([720, 160])
        self.setCentralWidget(split)

        self._build_conn_tab()
        self._build_acq_tab()
        self._build_ana_tab()
        self._build_plot_tab()
        self._restore()
        self._iface_changed()
        self._set_connected(False)
        self.log("Ready. 1) Connect to the scope  2) Acquire data  3) Run the analysis.")

    # ---------------------------------------------------------------- logging
    def log(self, msg):
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        for line in str(msg).splitlines():
            self.logw.appendPlainText("%s  %s" % (ts, line))
        self.logw.verticalScrollBar().setValue(self.logw.verticalScrollBar().maximum())

    def error(self, msg, title="Error"):
        self.log("ERROR: " + msg)
        QtWidgets.QMessageBox.critical(self, title, msg)

    # ---------------------------------------------------------- connection
    def _build_conn_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        box = QtWidgets.QGroupBox("Scope connection")
        box.setMaximumWidth(560)
        form = QtWidgets.QFormLayout(box)
        self.iface = QtWidgets.QComboBox()
        self.iface.addItems(["Ethernet - raw socket (no VISA driver needed)",
                             "Ethernet - VISA (VXI-11 / HiSLIP)",
                             "USB - VISA (USBTMC)",
                             "Simulated scope (demo, no hardware)"])
        self.iface.currentIndexChanged.connect(self._iface_changed)
        form.addRow("Interface:", self.iface)
        self.ip = _form_row(form, "Scope IP address / host:", QtWidgets.QLineEdit("192.168.1.100"))
        self.ip.textChanged.connect(self._iface_changed)
        self.port = QtWidgets.QSpinBox(); self.port.setRange(1, 65535); self.port.setValue(4000)
        _form_row(form, "Socket port:", self.port, "MSO5 socket server default port is 4000")
        self.visa_lib = QtWidgets.QComboBox()
        self.visa_lib.addItems(["System VISA (TekVISA / NI-VISA / Keysight)",
                                "pyvisa-py (pure Python)"])
        form.addRow("VISA library:", self.visa_lib)
        rrow = QtWidgets.QHBoxLayout()
        self.res = QtWidgets.QComboBox(); self.res.setEditable(True)
        self.res.setMinimumWidth(300)
        self.scan_btn = QtWidgets.QPushButton("Scan...")
        self.scan_btn.clicked.connect(self._scan)
        rrow.addWidget(self.res, 1); rrow.addWidget(self.scan_btn)
        form.addRow("VISA resource:", rrow)
        self.timeout = QtWidgets.QDoubleSpinBox(); self.timeout.setRange(1, 600)
        self.timeout.setValue(60); self.timeout.setSuffix(" s")
        _form_row(form, "I/O timeout:", self.timeout,
                  "Long records take a while to transfer; 60 s is a safe default.")
        brow = QtWidgets.QHBoxLayout()
        self.conn_btn = QtWidgets.QPushButton("Connect")
        self.conn_btn.setStyleSheet("font-weight: bold; padding: 6px 18px;")
        self.conn_btn.clicked.connect(self._connect)
        self.disc_btn = QtWidgets.QPushButton("Disconnect")
        self.disc_btn.clicked.connect(self._disconnect)
        brow.addWidget(self.conn_btn); brow.addWidget(self.disc_btn); brow.addStretch(1)
        form.addRow("", brow)
        self.conn_lbl = QtWidgets.QLabel()
        form.addRow("Status:", self.conn_lbl)
        self.idn_lbl = QtWidgets.QLabel("-"); self.idn_lbl.setWordWrap(True)
        form.addRow("Instrument:", self.idn_lbl)
        lay.addWidget(box)
        hb = QtWidgets.QTextBrowser(); hb.setHtml(HELP_HTML)
        lay.addWidget(hb, 1)
        self.tabs.addTab(w, "1. Scope connection")

    def _iface_changed(self, *_):
        i = self.iface.currentIndex()
        self.ip.setEnabled(i in (0, 1))
        self.port.setEnabled(i == 0)
        self.visa_lib.setEnabled(i in (1, 2))
        self.res.setEnabled(i in (1, 2))
        self.scan_btn.setEnabled(i in (1, 2))
        if i == 1:
            cur = self.res.currentText().strip()
            if not cur or cur.upper().startswith("TCPIP"):
                kind = "hislip0" if "hislip" in cur.lower() else "inst0"
                self.res.setEditText("TCPIP0::%s::%s::INSTR" % (self.ip.text().strip(), kind))

    def _visa_backend(self):
        return "@py" if self.visa_lib.currentIndex() == 1 else ""

    def _scan(self):
        try:
            import pyvisa
            rm = pyvisa.ResourceManager(self._visa_backend())
            found = list(rm.list_resources("?*::INSTR"))
        except Exception as e:  # noqa: BLE001
            self.error("VISA scan failed: %s\nInstall TekVISA/NI-VISA, or select pyvisa-py." % e)
            return
        if not found:
            self.log("VISA scan: no instruments found.")
            return
        self.res.clear()
        self.res.addItems(found)
        want = "USB" if self.iface.currentIndex() == 2 else "TCPIP"
        tek = [r for r in found if r.startswith(want) and "0x0699" in r] or \
              [r for r in found if r.startswith(want)]
        if tek:
            self.res.setCurrentText(tek[0])
        self.log("VISA scan found: " + ", ".join(found))

    def _connect(self):
        self._disconnect(quiet=True)
        i = self.iface.currentIndex()
        host, port, tmo = self.ip.text().strip(), self.port.value(), self.timeout.value()
        res, backend = self.res.currentText().strip(), self._visa_backend()
        self._save_settings()

        def job(log, prog, cancel):
            if i == 0:
                tr = SocketTransport(host, port, tmo)
            elif i in (1, 2):
                if not res:
                    raise ScopeError("Enter a VISA resource name or click Scan.")
                tr = VisaTransport(res, tmo, backend)
            else:
                tr = SimTransport()
            try:
                tr.clear()
                tr.write("HEADer OFF")
                tr.write("VERBose ON")
                idn = tr.query("*IDN?")
            except Exception:
                tr.close()
                raise
            return tr, idn

        self.conn_btn.setEnabled(False)
        self.conn_lbl.setText("Connecting...")
        self._run(job, self._connected, busy_text=None,
                  on_fail=lambda m: (self._set_connected(False),
                                     self.error(m, "Connection failed")))

    def _connected(self, res):
        self.tr_, self.idn = res
        self._set_connected(True)
        self.log("Connected via %s: %s" % (self.tr_.desc, self.idn))
        if "MSO5" not in self.idn.upper().replace(" ", ""):
            self.log("Note: instrument is not an MSO5-series scope; commands may differ.")

    def _disconnect(self, quiet=False):
        if self.tr_ is not None:
            self.tr_.close()
            self.tr_ = None
            if not quiet:
                self.log("Disconnected.")
        self._set_connected(False)

    def _set_connected(self, ok):
        self.conn_btn.setEnabled(True)
        self.disc_btn.setEnabled(ok)
        self.acq_btn.setEnabled(ok)
        self.readh_btn.setEnabled(ok)
        if ok:
            self.conn_lbl.setText("● Connected")
            self.conn_lbl.setStyleSheet("color: #0a7a2f; font-weight: bold;")
            parts = self.idn.split(",")
            self.idn_lbl.setText(self.idn)
            self.model = parts[1].strip() if len(parts) > 1 else "scope"
        else:
            self.conn_lbl.setText("● Not connected")
            self.conn_lbl.setStyleSheet("color: #888;")
            self.idn_lbl.setText("-")

    # ---------------------------------------------------------- acquisition
    def _build_acq_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        chs = ["CH%d" % k for k in range(1, 9)]

        gi = QtWidgets.QGroupBox("AC current channels  (channel / scale factor)")
        gl = QtWidgets.QGridLayout(gi)
        self.chI, self.sclI = [], []
        for k in range(3):
            cb = QtWidgets.QComboBox(); cb.addItems(chs + ([] if k == 0 else ["None"]))
            cb.setCurrentText("CH%d" % (k + 1))
            sp = QtWidgets.QDoubleSpinBox(); sp.setDecimals(6); sp.setRange(-1e6, 1e6)
            sp.setValue(1.0)
            sp.setToolTip("Multiplier applied to the scope reading. 1 = the scope channel "
                          "already reads amps (current probe). For a transducer enter A per V.")
            gl.addWidget(QtWidgets.QLabel("Phase %s current:" % PH[k]), k, 0)
            gl.addWidget(cb, k, 1); gl.addWidget(QtWidgets.QLabel("x"), k, 2); gl.addWidget(sp, k, 3)
            self.chI.append(cb); self.sclI.append(sp)
        left.addWidget(gi)

        self.gv = QtWidgets.QGroupBox("AC voltage channels  (channel / scale factor)")
        self.gv.setCheckable(True); self.gv.setChecked(True)
        gl = QtWidgets.QGridLayout(self.gv)
        self.chV, self.sclV = [], []
        for k in range(3):
            cb = QtWidgets.QComboBox(); cb.addItems(chs); cb.setCurrentText("CH%d" % (k + 6))
            sp = QtWidgets.QDoubleSpinBox(); sp.setDecimals(6); sp.setRange(-1e6, 1e6)
            sp.setValue(1.0)
            sp.setToolTip("Multiplier applied to the scope reading (1 = scope reads volts).")
            gl.addWidget(QtWidgets.QLabel("Phase %s voltage:" % PH[k]), k, 0)
            gl.addWidget(cb, k, 1); gl.addWidget(QtWidgets.QLabel("x"), k, 2); gl.addWidget(sp, k, 3)
            self.chV.append(cb); self.sclV.append(sp)
        left.addWidget(self.gv)

        ga = QtWidgets.QGroupBox("Acquisition")
        form = QtWidgets.QFormLayout(ga)
        self.acq_mode = QtWidgets.QComboBox()
        self.acq_mode.addItems(["Single acquisition (wait for trigger)",
                                "Transfer the waveform already on screen (stops the scope)"])
        form.addRow("Mode:", self.acq_mode)
        self.apply_h = QtWidgets.QCheckBox("Set sample rate and record length on the scope")
        form.addRow("", self.apply_h)
        self.fs = QtWidgets.QDoubleSpinBox(); self.fs.setDecimals(0)
        self.fs.setRange(1e3, 1e11); self.fs.setValue(2.5e6); self.fs.setSuffix(" S/s")
        self.fs.setGroupSeparatorShown(True)
        form.addRow("Sample rate:", self.fs)
        self.rl = QtWidgets.QSpinBox(); self.rl.setRange(1000, 1_000_000_000)
        self.rl.setValue(1_250_000); self.rl.setSuffix(" pts"); self.rl.setGroupSeparatorShown(True)
        form.addRow("Record length:", self.rl)
        self.dur_lbl = QtWidgets.QLabel()
        form.addRow("Record duration:", self.dur_lbl)
        self.fs.valueChanged.connect(self._upd_dur); self.rl.valueChanged.connect(self._upd_dur)
        self.apply_h.toggled.connect(self._upd_dur)
        self.trig_to = QtWidgets.QDoubleSpinBox(); self.trig_to.setRange(1, 3600)
        self.trig_to.setValue(30); self.trig_to.setSuffix(" s")
        form.addRow("Trigger wait timeout:", self.trig_to)
        self.restart = QtWidgets.QCheckBox("Put the scope back in RUN after the transfer")
        self.restart.setChecked(True)
        form.addRow("", self.restart)
        left.addWidget(ga)

        brow = QtWidgets.QHBoxLayout()
        self.readh_btn = QtWidgets.QPushButton("Read scope settings")
        self.readh_btn.clicked.connect(self._read_settings)
        self.acq_btn = QtWidgets.QPushButton("Acquire && transfer")
        self.acq_btn.setStyleSheet("font-weight: bold; font-size: 14px; padding: 8px;")
        self.acq_btn.clicked.connect(self._acquire)
        brow.addWidget(self.readh_btn); brow.addWidget(self.acq_btn, 1)
        left.addLayout(brow)
        brow = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Save waveform data..."); b.clicked.connect(self._save_data)
        b2 = QtWidgets.QPushButton("Load data file..."); b2.clicked.connect(self._load_data)
        brow.addWidget(b); brow.addWidget(b2)
        left.addLayout(brow)
        self.data_lbl = QtWidgets.QLabel("No data yet."); self.data_lbl.setWordWrap(True)
        left.addWidget(self.data_lbl)
        left.addStretch(1)

        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setMaximumWidth(500)
        lay.addWidget(lw)
        self.prev_fig = Figure(figsize=(7, 6), dpi=100)
        self.prev_canvas = FigureCanvas(self.prev_fig)
        pv = QtWidgets.QVBoxLayout()
        pv.addWidget(NavToolbar(self.prev_canvas, w)); pv.addWidget(self.prev_canvas, 1)
        lay.addLayout(pv, 1)
        self.tabs.addTab(w, "2. Acquire data")
        self._upd_dur()
        self._plot_preview()

    def _upd_dur(self, *_):
        self.fs.setEnabled(self.apply_h.isChecked())
        self.rl.setEnabled(self.apply_h.isChecked())
        d = self.rl.value() / self.fs.value()
        msg = "%.3f s" % d
        if d < 0.2:
            msg += "  (shorter than the 0.2 s Hann window!)"
        self.dur_lbl.setText(msg)

    def _read_settings(self):
        if not self.tr_:
            return
        tr = self.tr_

        def job(log, prog, cancel):
            tr.clear()
            return _q_num(tr, "HORizontal:SAMPLERate?"), int(_q_num(tr, "HORizontal:RECOrdlength?"))

        def ok(r):
            self.fs.setValue(r[0]); self.rl.setValue(r[1])
            self.log("Scope horizontal: %.4g S/s, %d points (%.3f s)" % (r[0], r[1], r[1] / r[0]))
        self._run(job, ok)

    def _acquire(self):
        if not self.tr_:
            self.error("Connect to the scope first (tab 1).")
            return
        cfg = dict(
            cur=[(cb.currentText() if cb.currentText() != "None" else None, sp.value())
                 for cb, sp in zip(self.chI, self.sclI)],
            vol=[(cb.currentText(), sp.value()) for cb, sp in zip(self.chV, self.sclV)],
            use_v=self.gv.isChecked(), mode=self.acq_mode.currentIndex() + 1,
            apply_h=self.apply_h.isChecked(), fs=self.fs.value(), rl=self.rl.value(),
            trig_timeout=self.trig_to.value(), restart=self.restart.isChecked(),
            idn=self.idn, model=getattr(self, "model", "MSO58B"))
        # phases B/C: "None" on B means C is ignored too (keep A,B,C order)
        cur = cfg["cur"]
        if cur[1][0] is None:
            cfg["cur"] = [cur[0], (None, 1), (None, 1)]
        used = [c for c, _ in cfg["cur"] if c] + ([c for c, _ in cfg["vol"]] if cfg["use_v"] else [])
        if len(set(used)) != len(used):
            self.error("The same scope channel is assigned twice. Check the channel mapping.")
            return
        tr = self.tr_
        self.log("-" * 50)
        self.log("Acquiring: currents %s; voltages %s" % (
            ", ".join(c for c, _ in cfg["cur"] if c),
            ", ".join(c for c, _ in cfg["vol"]) if cfg["use_v"] else "none"))
        self._run(lambda log, prog, cancel: acquire(tr, cfg, log, prog, cancel),
                  self._acquired, busy_text="Acquiring from the scope...")

    def _acquired(self, D):
        self._set_data(D)
        self.log("Capture complete: " + self._data_summary())

    def _set_data(self, D):
        self.D = D
        self.R = None
        self._clear_results()
        txt = self._data_summary()
        self.data_lbl.setText(txt)
        self.ana_data_lbl.setText(txt)
        self.run_btn.setEnabled(True)
        self._plot_preview()

    def _data_summary(self):
        D = self.D
        if D is None:
            return "No data yet."
        dt = np.median(np.diff(D["t"][:1000]))
        return ("%s\n%d samples, %.4g S/s, %.3f s; %d current channel(s), %s voltage channels"
                % (D["desc"], len(D["t"]), 1 / dt, D["t"][-1] - D["t"][0], D["I"].shape[1],
                   "3" if D["V"] is not None else "no"))

    def _plot_preview(self):
        fg = self.prev_fig
        fg.clear()
        a1 = fg.add_subplot(211); a2 = fg.add_subplot(212, sharex=a1)
        D = self.D
        if D is None:
            a1.set_title("Currents (no data yet)"); a2.set_title("Voltages (no data yet)")
        else:
            dt = np.median(np.diff(D["t"][:1000]))
            n = min(len(D["t"]), max(10, int(round(3 / self.freq.value() / dt))))
            step = max(1, n // 5000)
            sl = slice(0, n, step)
            for p in range(D["I"].shape[1]):
                a1.plot(D["t"][sl], D["I"][sl, p], color=PH_COLORS[p], lw=1, label="Phase %s" % PH[p])
            a1.set_title("Currents (first 3 periods)"); a1.set_ylabel("A"); a1.grid(True)
            a1.legend(loc="upper right", fontsize=8)
            if D["V"] is not None:
                for p in range(3):
                    a2.plot(D["t"][sl], D["V"][sl, p], color=PH_COLORS[p], lw=1,
                            label="Phase %s" % PH[p])
                a2.legend(loc="upper right", fontsize=8)
                a2.set_title("Voltages (first 3 periods)")
            else:
                a2.set_title("No voltage data")
            a2.set_ylabel("V"); a2.set_xlabel("Time [s]"); a2.grid(True)
        fg.tight_layout()
        self.prev_canvas.draw_idle()

    def _save_data(self):
        if self.D is None:
            self.error("There is no data to save yet.")
            return
        start = os.path.join(self.settings.value("lastDir", os.getcwd()), self.D["base_name"] + ".csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save waveform data", start,
            "CSV (readable by MCU_Harmonic_Analyzer.m) (*.csv);;NumPy archive, fast (*.npz)")
        if not path:
            return
        self.settings.setValue("lastDir", os.path.dirname(path))
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            save_data_file(path, self.D)
            self.log("Waveform data saved: " + path)
        except Exception as e:  # noqa: BLE001
            self.error(str(e))
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _load_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load waveform data", self.settings.value("lastDir", os.getcwd()),
            "Waveform data (*.csv *.txt *.npz);;All files (*.*)")
        if not path:
            return
        self.settings.setValue("lastDir", os.path.dirname(path))
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            D = load_data_file(path)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            self.error(str(e), "Could not load file")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._set_data(D)
        self.log("Loaded %s  (%s)" % (path, D["format"]))

    # ---------------------------------------------------------- analysis
    def _build_ana_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(w)
        left = QtWidgets.QVBoxLayout()
        gs = QtWidgets.QGroupBox("Analysis settings")
        form = QtWidgets.QFormLayout(gs)
        self.mcu = QtWidgets.QComboBox()
        self.mcu.addItem("MCU1  (max current 12.5 A)", 12.5)
        self.mcu.addItem("MCU2  (max current 6.25 A)", 6.25)
        form.addRow("MCU type:", self.mcu)
        self.limcol = QtWidgets.QComboBox()
        self.limcol.addItem("Three-phase equipment, 20 kVA or less", 2)
        self.limcol.addItem("Three-phase equipment, more than 20 kVA", 3)
        self.limcol.addItem("Single-phase equipment", 1)
        form.addRow("ABD 1.8 limit column:", self.limcol)
        self.freq = QtWidgets.QDoubleSpinBox(); self.freq.setRange(1, 1e5); self.freq.setValue(400)
        self.freq.setSuffix(" Hz")
        _form_row(form, "AC fundamental frequency:", self.freq,
                  "Usually 400 Hz. Use 385 or 415 for off-nominal tests (e.g. TCF101).")
        self.norm = QtWidgets.QComboBox()
        self.norm.addItems(["MCU nominal current (per-unit of rating)", "Measured fundamental current"])
        form.addRow("Harmonics relative to:", self.norm)
        self.hann = QtWidgets.QDoubleSpinBox(); self.hann.setDecimals(4)
        self.hann.setRange(1e-4, 100); self.hann.setValue(0.2); self.hann.setSuffix(" s")
        _form_row(form, "Hann window length:", self.hann,
                  "0.2 s gives 5 Hz frequency resolution (standard setting).")
        self.method = QtWidgets.QComboBox()
        self.method.addItems([METHOD_NAMES[1], METHOD_NAMES[2], METHOD_NAMES[3]])
        form.addRow("FFT method:", self.method)
        self.winnum = QtWidgets.QSpinBox(); self.winnum.setRange(1, 100000)
        _form_row(form, "Window number:", self.winnum,
                  "Single-window method: 1 = first Hann window of the record, 2 = second, ...")
        self.method.currentIndexChanged.connect(lambda i: self.winnum.setEnabled(i == 0))
        self.invert = QtWidgets.QCheckBox("Invert current polarity (probes installed backwards)")
        form.addRow(self.invert)
        self.filt = QtWidgets.QCheckBox("Report view: standard harmonics (2, 4, odd 3-59) + any failing")
        self.filt.setChecked(True)
        self.filt.toggled.connect(self._fill_harm_table)
        form.addRow(self.filt)
        left.addWidget(gs)

        self.ana_data_lbl = QtWidgets.QLabel("No data yet. Acquire from the scope or load a file.")
        self.ana_data_lbl.setWordWrap(True)
        gd = QtWidgets.QGroupBox("Data"); QtWidgets.QVBoxLayout(gd).addWidget(self.ana_data_lbl)
        left.addWidget(gd)
        self.run_btn = QtWidgets.QPushButton("Run analysis")
        self.run_btn.setStyleSheet("font-weight: bold; font-size: 15px; padding: 10px;")
        self.run_btn.clicked.connect(self._run_analysis)
        self.run_btn.setEnabled(False)
        left.addWidget(self.run_btn)
        self.rep_btn = QtWidgets.QPushButton("Save Excel report...")
        self.rep_btn.clicked.connect(self._save_report)
        self.rep_btn.setEnabled(False)
        left.addWidget(self.rep_btn)
        left.addStretch(1)
        lw = QtWidgets.QWidget(); lw.setLayout(left); lw.setMaximumWidth(470)
        lay.addWidget(lw)

        right = QtWidgets.QVBoxLayout()
        self.overall = QtWidgets.QLabel("Run the analysis to see results.")
        self.overall.setStyleSheet("font-size: 16px; font-weight: bold; padding: 6px;")
        right.addWidget(self.overall)
        right.addWidget(QtWidgets.QLabel("<b>Phase currents</b>"))
        self.sum_tbl = self._mk_table(["Phase", "Fundamental [A rms]", "Angle [deg]",
                                       "THD 2-40 [%]", "3rd harm. [A rms]", "Harmonics over limit"])
        self.sum_tbl.setMaximumHeight(130)
        right.addWidget(self.sum_tbl)
        right.addWidget(QtWidgets.QLabel("<b>Phase voltages</b>"))
        self.v_tbl = self._mk_table(["Phase", "Fundamental [V rms]", "Angle [deg]", "V - I angle [deg]"])
        self.v_tbl.setMaximumHeight(130)
        right.addWidget(self.v_tbl)
        right.addWidget(QtWidgets.QLabel("<b>Harmonics</b> (FAIL rows highlighted)"))
        self.h_tbl = self._mk_table([])
        right.addWidget(self.h_tbl, 1)
        lay.addLayout(right, 1)
        self.tabs.addTab(w, "3. Harmonic analysis")

    @staticmethod
    def _mk_table(cols):
        t = QtWidgets.QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        t.setAlternatingRowColors(True)
        return t

    def _build_plot_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.plot_tabs = QtWidgets.QTabWidget()
        lay.addWidget(self.plot_tabs)
        self.plot_tabs.addTab(QtWidgets.QLabel("Run the analysis to create the plots."), "Plots")
        self.tabs.addTab(w, "4. Plots")

    def _options(self):
        lim_col = self.limcol.currentData()
        _, lim_msg = abd_limits(lim_col)
        return dict(nominal=self.mcu.currentData(), mcu_name=self.mcu.currentText().strip(),
                    f_fund=self.freq.value(), use_nominal=self.norm.currentIndex() == 0,
                    hann_dur=self.hann.value(), method=self.method.currentIndex() + 1,
                    win_num=self.winnum.value(), invert=self.invert.isChecked(),
                    filter=self.filt.isChecked(), lim_col=lim_col, lim_msg=lim_msg)

    def _run_analysis(self):
        if self.D is None:
            self.error("Acquire data from the scope or load a data file first.")
            return
        o = self._options()
        D = dict(self.D)
        if o["invert"]:
            D["I"] = -D["I"]
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.log("-" * 50)
            self.log("Analysis (%s): %s" % (o["mcu_name"], o["lim_msg"]))
            R = analyze(D, o)
            figs = make_figures(D, R, o)
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            traceback.print_exc()
            self.error(str(e), "Analysis error")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.R, self.o, self.D_used, self.figs = R, o, D, figs
        self.log("Sample rate %.4g S/s, record %.3f s, %d window(s) of %d samples"
                 % (R["Fs"], R["N"] * R["Ts"], R["n_win"], R["N_hann"]))
        for p in range(R["nPh"]):
            self.log("  Phase %s: fundamental %.3f A rms, THD %.2f %%, %d harmonic(s) over limit"
                     % (PH[p], R["fund_rms"][p], 100 * R["thd"][p], R["n_fail"][p]))
        self.log(R["overall"])
        self._show_results()
        self.rep_btn.setEnabled(True)

    def _clear_results(self):
        self.overall.setText("Run the analysis to see results.")
        self.overall.setStyleSheet("font-size: 16px; font-weight: bold; padding: 6px;")
        for t in (self.sum_tbl, self.v_tbl, self.h_tbl):
            t.setRowCount(0)
        self.rep_btn.setEnabled(False)

    @staticmethod
    def _cell(v, fmt="%.4f"):
        if isinstance(v, str):
            it = QtWidgets.QTableWidgetItem(v)
        elif v is None or (isinstance(v, float) and not np.isfinite(v)):
            it = QtWidgets.QTableWidgetItem("")
        elif isinstance(v, (int, np.integer)):
            it = QtWidgets.QTableWidgetItem("%d" % v)
        else:
            it = QtWidgets.QTableWidgetItem(fmt % v)
        it.setTextAlignment(Qt.AlignCenter)
        return it

    def _show_results(self):
        R = self.R
        ok = R["overall"].startswith("PASS")
        self.overall.setText(R["overall"])
        self.overall.setStyleSheet("font-size: 16px; font-weight: bold; padding: 6px; color: white; "
                                   "background: %s; border-radius: 4px;" % ("#1b8a3a" if ok else "#c62828"))
        t = self.sum_tbl
        t.setRowCount(R["nPh"])
        for p in range(R["nPh"]):
            vals = [PH[p], R["fund_rms"][p], R["fund_ang"][p], 100 * R["thd"][p], R["h3_rms"][p],
                    int(R["n_fail"][p])]
            fm = ["", "%.4f", "%.2f", "%.3f", "%.5f", ""]
            for c, (v, f_) in enumerate(zip(vals, fm)):
                t.setItem(p, c, self._cell(v, f_ or "%s"))
        t = self.v_tbl
        if R["have_v"]:
            t.setRowCount(3)
            for p in range(3):
                vals = [PH[p], R["v_rms"][p], R["v_ang"][p],
                        float(wrap180(R["v_ang"][p] - R["fund_ang"][p]))]
                for c, (v, f_) in enumerate(zip(vals, ["", "%.3f", "%.2f", "%.2f"])):
                    t.setItem(p, c, self._cell(v, f_ or "%s"))
        else:
            t.setRowCount(0)
        self._fill_harm_table()

        # plots
        self.plot_tabs.clear()
        for fg, title in zip(self.figs, PLOT_TITLES):
            pw = QtWidgets.QWidget()
            pl = QtWidgets.QVBoxLayout(pw)
            cv = FigureCanvas(fg)
            pl.addWidget(NavToolbar(cv, pw)); pl.addWidget(cv, 1)
            self.plot_tabs.addTab(pw, title)

    def _fill_harm_table(self, *_):
        R = self.R
        t = self.h_tbl
        if R is None:
            return
        cols, data = R["table_cols"], R["table_data"]
        if self.filt.isChecked():
            mask = np.zeros(len(R["H"]), dtype=bool)
            mask[0] = True
            std = set([2, 4] + list(range(3, 60, 2)))
            for hh in range(2, min(75, len(R["H"])) + 1):
                mask[hh - 1] = hh in std or R["result"][hh - 1] == "FAIL"
        else:
            mask = np.ones(len(R["H"]), dtype=bool)
        idx = np.flatnonzero(mask)
        t.clear()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(idx))
        fail_bg = QtGui.QBrush(QtGui.QColor(255, 205, 205))
        for r, i in enumerate(idx):
            for c, col in enumerate(cols):
                v = data[c][i]
                if col == "Harmonic":
                    it = self._cell(int(v))
                elif col == "Frequency_Hz":
                    it = self._cell(float(v), "%.1f")
                elif col == "Result":
                    it = self._cell(str(v))
                else:
                    it = self._cell(float(v), "%.5f")
                if R["result"][i] == "FAIL":
                    it.setBackground(fail_bg)
                t.setItem(r, c, it)

    def _save_report(self):
        if self.R is None:
            return
        start = os.path.join(self.settings.value("lastDir", os.getcwd()),
                             self.D_used["base_name"] + "_harmonics.xlsx")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Excel report as", start,
                                                        "Excel workbook (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        self.settings.setValue("lastDir", os.path.dirname(path))
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # the report uses the filter setting chosen at analysis time; refresh it
            self.R["keep"] = self._keep_mask()
            msg = write_report(path, self.D_used, self.R, self.o, self.figs)
        except PermissionError:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.error("Cannot write \"%s\". Is it open in Excel? Close it and try again." % path)
            return
        except Exception as e:  # noqa: BLE001
            QtWidgets.QApplication.restoreOverrideCursor()
            traceback.print_exc()
            self.error(str(e), "Report error")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self.log(msg)
        self.log("Report saved: " + path)
        r = QtWidgets.QMessageBox.question(
            self, "Done", "Report saved:\n%s\n\n%s\n\nOpen the spreadsheet now?" % (path, self.R["overall"]))
        if r == QtWidgets.QMessageBox.Yes:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _keep_mask(self):
        R = self.R
        if not self.filt.isChecked():
            return np.ones(len(R["H"]), dtype=bool)
        mask = np.zeros(len(R["H"]), dtype=bool)
        mask[0] = True
        std = set([2, 4] + list(range(3, 60, 2)))
        for hh in range(2, min(75, len(R["H"])) + 1):
            mask[hh - 1] = hh in std or R["result"][hh - 1] == "FAIL"
        return mask

    # ---------------------------------------------------------- task runner
    def _run(self, fn, on_done, busy_text=None, on_fail=None):
        if self.task is not None and self.task.isRunning():
            self.error("Another operation is still running.")
            return
        task = Task(fn, self)
        dlg = None
        if busy_text:
            dlg = QtWidgets.QProgressDialog(busy_text, "Cancel", 0, 100, self)
            dlg.setWindowTitle("Scope")
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(0)
            dlg.setValue(0)
            dlg.setMinimumWidth(420)
            dlg.canceled.connect(task.cancel.set)
            task.prog.connect(lambda v, m: (dlg.setValue(min(v, 99)), m and dlg.setLabelText(m)))
        task.log.connect(self.log)

        def fin_ok(res):
            if dlg:
                dlg.reset()
            on_done(res)

        def fin_err(msg):
            if dlg:
                dlg.reset()
            if on_fail:
                on_fail(msg)
            else:
                self.error(msg)
        task.done.connect(fin_ok)
        task.failed.connect(fin_err)
        self.task = task
        task.start()

    # ---------------------------------------------------------- settings
    def _restore(self):
        s = self.settings
        self.iface.setCurrentIndex(int(s.value("iface", 0)))
        self.ip.setText(s.value("ip", "192.168.1.100"))
        self.port.setValue(int(s.value("port", 4000)))
        self.visa_lib.setCurrentIndex(int(s.value("visaLib", 0)))
        r = s.value("resource", "")
        if r:
            self.res.setEditText(r)

    def _save_settings(self):
        s = self.settings
        s.setValue("iface", self.iface.currentIndex())
        s.setValue("ip", self.ip.text())
        s.setValue("port", self.port.value())
        s.setValue("visaLib", self.visa_lib.currentIndex())
        s.setValue("resource", self.res.currentText())

    def closeEvent(self, ev):
        self._save_settings()
        if self.task is not None and self.task.isRunning():
            self.task.cancel.set()
            self.task.wait(3000)
        self._disconnect(quiet=True)
        super().closeEvent(ev)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
