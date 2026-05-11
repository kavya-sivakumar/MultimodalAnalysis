# =============================================================
# HRV Synchrony Pipeline
# ============================================================
# Research context:
#   Live performing arts may create shared physiological states
#   between audience members (or between performer - audience)
#
#
#   This pipeline quantifies *interpersonal HRV synchrony* from
#   dyadic recordings using:
#     1. Windowed cross-correlation
#     2. Phase Locking Value (PLV)
#     3. Wavelet coherence
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import pearsonr
import pywt
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------
# 1. SYNTHETIC DATA 
# ----------------------------------------------------------------
# Generate synthetic HRV (RR intervals in ms) for two participants
# load ECG → detect R-peaks → compute RR intervals
# using neurokit2: nk.ecg_process(ecg_signal, sampling_rate=1000)

def generate_synthetic_hrv(n_samples=300, base_rr=850, noise=30,
                            coupling_strength=0.0, seed=None):
    """
    Simulate RR interval series. Two participants share a common
    oscillatory component if coupling_strength > 0.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, n_samples / 4, n_samples)
    # Respiratory sinus arrhythmia (~0.25 Hz)
    rsa = 40 * np.sin(2 * np.pi * 0.25 * t)
    shared = 25 * np.sin(2 * np.pi * 0.1 * t)  # shared slow oscillation
    p1 = base_rr + rsa + rng.normal(0, noise, n_samples)
    p2 = (base_rr + coupling_strength * shared +
          (1 - coupling_strength) * rsa * 0.8 +
          rng.normal(0, noise, n_samples))
    return p1, p2

fs = 4  # Hz (RR series resampled to 4 Hz — standard for HRV)
n = 300

# Simulate two conditions: low vs. high synchrony (e.g., separate vs. shared concert)
rr1_low, rr2_low   = generate_synthetic_hrv(n, coupling_strength=0.1, seed=SEED)
rr1_high, rr2_high = generate_synthetic_hrv(n, coupling_strength=0.85, seed=SEED+1)

time_axis = np.arange(n) / fs  # seconds

# ----------------------------------------------------------------
# 2. PREPROCESSING — bandpass filter to HF-HRV band (0.15–0.4 Hz)
# ----------------------------------------------------------------
# Bandpass filter RR series

def bandpass_hrv(signal, fs=4, lowcut=0.04, highcut=0.4, order=3):
    """Filter RR series to the LF+HF band."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)

rr1_low_f  = bandpass_hrv(rr1_low)
rr2_low_f  = bandpass_hrv(rr2_low)
rr1_high_f = bandpass_hrv(rr1_high)
rr2_high_f = bandpass_hrv(rr2_high)

# ----------------------------------------------------------------
# 3. WINDOWED CROSS-CORRELATION
# ----------------------------------------------------------------
# Compute windowed cross-correlation between two RR series

def windowed_xcorr(s1, s2, window=40, step=10, max_lag=10):
    """
    Slide a window over two signals, compute peak cross-correlation
    at each window position.
    Returns: time points, peak correlations, lag at peak (samples)

    Fix: both seg1 and seg2 are sliced to the same fixed-length center
    region (length = window - 2*max_lag) so pearsonr always receives
    equal-length arrays regardless of lag.
    """
    times, peak_corrs, peak_lags = [], [], []
    seg_len = window - 2 * max_lag          # length of the comparison region
    lags    = np.arange(-max_lag, max_lag + 1)

    for start in range(0, len(s1) - window, step):
        seg1 = s1[start:start + window]
        seg2 = s2[start:start + window]

        corrs = []
        for lag in lags:
            # seg1: fixed center slice [max_lag : max_lag + seg_len]
            # seg2: shifted by lag     [max_lag+lag : max_lag+lag+seg_len]
            s1_slice = seg1[max_lag : max_lag + seg_len]
            s2_start = max_lag + lag
            s2_slice = seg2[s2_start : s2_start + seg_len]
            if len(s1_slice) == seg_len and len(s2_slice) == seg_len:
                corrs.append(pearsonr(s1_slice, s2_slice)[0])
            else:
                corrs.append(np.nan)

        corrs    = np.array(corrs)
        best_idx = np.nanargmax(np.abs(corrs))
        peak_corrs.append(corrs[best_idx])
        peak_lags.append(lags[best_idx])
        times.append((start + window / 2) / fs)

    return np.array(times), np.array(peak_corrs), np.array(peak_lags)

times_low,  xcorr_low,  lags_low  = windowed_xcorr(rr1_low_f,  rr2_low_f)
times_high, xcorr_high, lags_high = windowed_xcorr(rr1_high_f, rr2_high_f)

# ----------------------------------------------------------------
# 4. PHASE LOCKING VALUE (PLV)
# ----------------------------------------------------------------
# Phase synchrony via Hilbert transform

def compute_plv_windowed(s1, s2, window=40, step=10):
    """
    Compute instantaneous phase via Hilbert transform, then
    phase locking value (PLV) over sliding windows.
    PLV = |mean(exp(i * delta_phase))|, range [0, 1].
    """
    phase1 = np.angle(hilbert(s1))
    phase2 = np.angle(hilbert(s2))
    delta  = phase1 - phase2
    times, plvs = [], []
    for start in range(0, len(s1) - window, step):
        plv = np.abs(np.mean(np.exp(1j * delta[start:start + window])))
        plvs.append(plv)
        times.append((start + window / 2) / fs)
    return np.array(times), np.array(plvs)

times_plv_low,  plv_low  = compute_plv_windowed(rr1_low_f,  rr2_low_f)
times_plv_high, plv_high = compute_plv_windowed(rr1_high_f, rr2_high_f)

# PLV permutation baseline (shuffle one signal to get chance level)
def permutation_plv_baseline(s1, s2, n_perms=200, window=40, step=10):
    """Estimate chance-level PLV by time-shifting one signal."""
    rng = np.random.default_rng(SEED)
    null_plvs = []
    for _ in range(n_perms):
        shift = rng.integers(window, len(s2) - window)
        s2_shuffled = np.roll(s2, shift)
        _, plv_null = compute_plv_windowed(s1, s2_shuffled, window, step)
        null_plvs.append(np.mean(plv_null))
    return np.percentile(null_plvs, 95)

baseline_low  = permutation_plv_baseline(rr1_low_f,  rr2_low_f)
baseline_high = permutation_plv_baseline(rr1_high_f, rr2_high_f)

# ----------------------------------------------------------------
# 5. WAVELET COHERENCE
# ----------------------------------------------------------------
# Wavelet coherence across time and frequency

def wavelet_coherence(s1, s2, fs=4, scales=None):
    """
    Compute continuous wavelet transform coherence between two signals.
    Returns: time, frequencies, coherence matrix (freq x time)
    """
    if scales is None:
        scales = np.arange(2, 64)
    wavelet = "cmor1.5-1.0"  # complex Morlet
    coef1, freqs = pywt.cwt(s1, scales, wavelet, sampling_period=1/fs)
    coef2, _     = pywt.cwt(s2, scales, wavelet, sampling_period=1/fs)
    # Cross-spectrum and auto-spectra (smoothed)
    cross = np.convolve(np.ones(5)/5, (coef1 * np.conj(coef2)).mean(axis=0), mode="same")
    power1 = np.abs(coef1).mean(axis=0) ** 2
    power2 = np.abs(coef2).mean(axis=0) ** 2
    coherence_matrix = np.abs(coef1 * np.conj(coef2)) / (
        np.sqrt(np.abs(coef1)**2 * np.abs(coef2)**2) + 1e-10)
    t = np.arange(len(s1)) / fs
    return t, freqs, coherence_matrix

t_wc, freqs_wc, wc_low  = wavelet_coherence(rr1_low_f,  rr2_low_f)
_,    _,        wc_high = wavelet_coherence(rr1_high_f, rr2_high_f)

# ----------------------------------------------------------------
# 6. VISUALISATION
# ----------------------------------------------------------------

fig, axes = plt.subplots(4, 2, figsize=(14, 14))
fig.suptitle("Interpersonal HRV Synchrony\nLow Coupling (left) vs. High Coupling (right)",
             fontsize=14, fontweight="bold", y=1.01)

conditions = [
    (rr1_low,  rr2_low,  xcorr_low,  lags_low,  plv_low,  baseline_low,  wc_low,  "Low Coupling"),
    (rr1_high, rr2_high, xcorr_high, lags_high, plv_high, baseline_high, wc_high, "High Coupling"),
]

for col, (r1, r2, xcorr, lags, plv, baseline, wc, label) in enumerate(conditions):

    # Row 0: Raw RR series
    axes[0, col].plot(time_axis, r1, alpha=0.8, label="Participant 1", color="#2196F3")
    axes[0, col].plot(time_axis, r2, alpha=0.8, label="Participant 2", color="#FF5722")
    axes[0, col].set_title(f"{label} — RR Intervals")
    axes[0, col].set_xlabel("Time (s)")
    axes[0, col].set_ylabel("RR (ms)")
    axes[0, col].legend(fontsize=8)

    # Row 1: Windowed cross-correlation
    axes[1, col].plot(times_low if col == 0 else times_high, xcorr, color="#4CAF50")
    axes[1, col].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1, col].set_ylim(-1, 1)
    axes[1, col].set_title(f"Windowed Cross-Correlation  (mean={np.nanmean(xcorr):.2f})")
    axes[1, col].set_xlabel("Time (s)")
    axes[1, col].set_ylabel("r")

    # Row 2: PLV over time
    axes[2, col].plot(times_plv_low if col == 0 else times_plv_high, plv, color="#9C27B0")
    axes[2, col].axhline(baseline, color="red", linestyle="--", linewidth=1,
                         label=f"95% permutation baseline ({baseline:.2f})")
    axes[2, col].set_ylim(0, 1)
    axes[2, col].set_title(f"Phase Locking Value  (mean={np.mean(plv):.2f})")
    axes[2, col].set_xlabel("Time (s)")
    axes[2, col].set_ylabel("PLV")
    axes[2, col].legend(fontsize=7)

    # Row 3: Wavelet coherence heatmap
    im = axes[3, col].imshow(wc, aspect="auto", origin="lower",
                              extent=[0, n/fs, 0, len(freqs_wc)],
                              cmap="plasma", vmin=0, vmax=1)
    axes[3, col].set_title("Wavelet Coherence (time × frequency)")
    axes[3, col].set_xlabel("Time (s)")
    axes[3, col].set_ylabel("Scale")
    plt.colorbar(im, ax=axes[3, col], fraction=0.046)

plt.tight_layout()
import os, pathlib
OUT_DIR = pathlib.Path(r"C:/Users/91948/Desktop/Portfolio/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "hrv_synchrony_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figure saved to {out_path}")

# ----------------------------------------------------------------
# 7. SUMMARY STATISTICS TABLE
# ----------------------------------------------------------------


import pingouin as pg

summary = pd.DataFrame({
    "condition":          ["low_coupling", "high_coupling"],
    "mean_xcorr":         [np.nanmean(xcorr_low),  np.nanmean(xcorr_high)],
    "mean_plv":           [np.mean(plv_low),        np.mean(plv_high)],
    "plv_above_baseline": [(plv_low  > baseline_low).mean(),
                           (plv_high > baseline_high).mean()],
})

print("\n=== Synchrony Summary ===")
print(summary.to_string(index=False))

# Independent-samples t-test on mean PLV (as if two groups)
ttest = pg.ttest(plv_high, plv_low, paired=False)
print(f"\nt-test on PLV (high vs. low): t={ttest['T'].values[0]:.2f}, "
      f"p={ttest['p-val'].values[0]:.4f}, d={ttest['cohen-d'].values[0]:.2f}")


