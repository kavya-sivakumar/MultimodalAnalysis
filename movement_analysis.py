# =============================================================
# Movement & Wearable Sensor Synchrony Pipeline
# ============================================================
# Research context:
#   To analyze accelerometer data from wearable sensors
#   (Quantify movement synchrony between pairs of children during
#   live music)
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, welch
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------
# 1. SYNTHETIC ACCELEROMETER DATA
# ----------------------------------------------------------------
#    Simulate triaxial accelerometer for two children
#    Sampling rate: 50 Hz (typical for wrist wearables)
#    Conditions: "music" (synchronized movement) vs. "baseline" (free play)

FS_ACC = 50       # Hz
DURATION = 120    # seconds

def simulate_movement(condition, child_id, seed):
    """Simulate x/y/z acceleration (in g) for one child."""
    rng  = np.random.default_rng(seed)
    t    = np.linspace(0, DURATION, DURATION * FS_ACC)

    # Rhythmic bouncing component (matches music beat ~2 Hz)
    beat_freq = 2.0
    beat_amp  = 0.4 if condition == "music" else 0.1

    # Each child has a slightly different phase offset
    phase_offset = rng.uniform(0, 0.3) if condition == "music" else rng.uniform(0, 2 * np.pi)

    x = beat_amp * np.sin(2 * np.pi * beat_freq * t + phase_offset) + rng.normal(0, 0.15, len(t))
    y = beat_amp * np.cos(2 * np.pi * beat_freq * t + phase_offset) + rng.normal(0, 0.15, len(t))
    z = 1.0 + 0.2 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 0.1, len(t))  # gravity + sway

    return pd.DataFrame({"time": t, "x": x, "y": y, "z": z})

# Two children in each condition
acc_music_c1   = simulate_movement("music",    child_id=1, seed=SEED)
acc_music_c2   = simulate_movement("music",    child_id=2, seed=SEED + 1)
acc_free_c1    = simulate_movement("free",     child_id=1, seed=SEED + 2)
acc_free_c2    = simulate_movement("free",     child_id=2, seed=SEED + 3)

# ----------------------------------------------------------------
# 2. PREPROCESSING
# ----------------------------------------------------------------
# Compute vector magnitude, then bandpass filter

def vector_magnitude(df):
    """Euclidean norm of triaxial acceleration — removes orientation dependency."""
    return np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)

def bandpass(signal, fs=50, lowcut=0.5, highcut=10.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)

vm_music_c1 = bandpass(vector_magnitude(acc_music_c1))
vm_music_c2 = bandpass(vector_magnitude(acc_music_c2))
vm_free_c1  = bandpass(vector_magnitude(acc_free_c1))
vm_free_c2  = bandpass(vector_magnitude(acc_free_c2))

# ----------------------------------------------------------------
# 3. WINDOWED CROSS-CORRELATION
# ----------------------------------------------------------------


def windowed_xcorr(s1, s2, fs, window_sec=5, step_sec=1, max_lag_sec=0.5):
    window  = int(window_sec * fs)
    step    = int(step_sec   * fs)
    max_lag = int(max_lag_sec * fs)
    times, r_peaks = [], []
    for start in range(0, len(s1) - window, step):
        seg1, seg2 = s1[start:start+window], s2[start:start+window]
        lags = np.arange(-max_lag, max_lag + 1)
        corrs = [pearsonr(seg1[max_lag:-max_lag],
                          seg2[max_lag+l:window-max_lag+l])[0]
                 if 0 < max_lag + l < window - max_lag else np.nan
                 for l in lags]
        r_peaks.append(np.nanmax(np.abs(corrs)))
        times.append((start + window / 2) / fs)
    return np.array(times), np.array(r_peaks)

t_music, r_music = windowed_xcorr(vm_music_c1, vm_music_c2, FS_ACC)
t_free,  r_free  = windowed_xcorr(vm_free_c1,  vm_free_c2,  FS_ACC)

# ----------------------------------------------------------------
# 4. RECURRENCE QUANTIFICATION ANALYSIS (RQA)
# ----------------------------------------------------------------
# RQA captures non-linear coupling in movement timing

def cross_recurrence(s1, s2, threshold=0.15, embed_dim=3, delay=5):
    """
    Compute cross-recurrence matrix for two signals.
    Returns: recurrence matrix, %REC, diagonal line entropy (DET)
    """
    N = len(s1) - (embed_dim - 1) * delay

    def embed(x):
        return np.array([x[i:i + embed_dim * delay:delay] for i in range(N)])

    E1, E2 = embed(s1), embed(s2)
    dist   = np.sqrt(((E1[:, None] - E2[None, :])**2).sum(axis=-1))
    rmat   = dist < threshold

    pct_rec = rmat.mean() * 100

    # Diagonal line structure (determinism)
    diag_lengths = []
    for offset in range(-N // 2, N // 2):
        diag = np.diag(rmat, offset)
        if len(diag) >= 2:
            in_line, length = False, 0
            for v in diag:
                if v:
                    length += 1
                    in_line = True
                else:
                    if in_line and length >= 2:
                        diag_lengths.append(length)
                    in_line, length = False, 0
    det = (sum(l for l in diag_lengths if l >= 2) / max(rmat.sum(), 1)) * 100

    return rmat, pct_rec, det

# Use downsampled signal (RQA is O(N^2), keep N manageable)
ds = 10  # downsample factor
_, rqa_rec_music, rqa_det_music = cross_recurrence(vm_music_c1[::ds], vm_music_c2[::ds])
_, rqa_rec_free,  rqa_det_free  = cross_recurrence(vm_free_c1[::ds],  vm_free_c2[::ds])

print(f"\nRQA Results:")
print(f"  Music condition: %REC={rqa_rec_music:.1f}%, DET={rqa_det_music:.1f}%")
print(f"  Free play:       %REC={rqa_rec_free:.1f}%,  DET={rqa_det_free:.1f}%")

# ----------------------------------------------------------------
# 5. SLIDING-WINDOW FEATURE EXTRACTION FOR CLASSIFICATION
# ----------------------------------------------------------------
# Extract features per 5-second window for movement state classification

def extract_window_features(s, fs, window_sec=5, step_sec=2.5):
    """
    Extract time- and frequency-domain features from each window.
    Suitable for Random Forest classification.
    """
    window, step = int(window_sec * fs), int(step_sec * fs)
    rows = []
    for start in range(0, len(s) - window, step):
        seg = s[start:start + window]
        freqs, psd = welch(seg, fs=fs, nperseg=min(128, len(seg)))

        # Frequency bands
        delta_mask = (freqs >= 0.5) & (freqs < 1.0)
        beat_mask  = (freqs >= 1.5) & (freqs < 2.5)
        high_mask  = (freqs >= 3.0) & (freqs < 8.0)

        rows.append({
            "mean":       np.mean(seg),
            "std":        np.std(seg),
            "max":        np.max(np.abs(seg)),
            "rms":        np.sqrt(np.mean(seg**2)),
            "jerk":       np.mean(np.abs(np.diff(seg))),
            "psd_delta":  psd[delta_mask].sum() if delta_mask.any() else 0,
            "psd_beat":   psd[beat_mask].sum()  if beat_mask.any()  else 0,
            "psd_high":   psd[high_mask].sum()  if high_mask.any()  else 0,
            "spectral_centroid": np.sum(freqs * psd) / (np.sum(psd) + 1e-10),
        })
    return pd.DataFrame(rows)

features_music = extract_window_features(vm_music_c1, FS_ACC)
features_free  = extract_window_features(vm_free_c1,  FS_ACC)
features_music["label"] = "music"
features_free["label"]  = "free"
all_features = pd.concat([features_music, features_free], ignore_index=True)

# Random forest cross-validation
X = all_features.drop("label", axis=1).values
y = (all_features["label"] == "music").astype(int).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
cv_scores = cross_val_score(rf, X_scaled, y,
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            scoring="balanced_accuracy")
print(f"\nRF Movement State Classification (5-fold CV):")
print(f"  Balanced accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

rf.fit(X_scaled, y)
feat_names = all_features.drop("label", axis=1).columns
importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)

# ----------------------------------------------------------------
# 6. VISUALISATION
# ----------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Movement Synchrony Analysis\nMusic vs. Free Play",
             fontsize=14, fontweight="bold")

# A: Raw vector magnitude (first 10 sec)
ax = axes[0, 0]
t_show = np.linspace(0, 10, 10 * FS_ACC)
ax.plot(t_show, vm_music_c1[:len(t_show)], alpha=0.8, color="#2196F3", label="Child 1 (music)")
ax.plot(t_show, vm_music_c2[:len(t_show)], alpha=0.8, color="#FF5722", label="Child 2 (music)")
ax.set_title("Movement Signal (first 10 s)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Vector magnitude (g)")
ax.legend(fontsize=8)

# B: Windowed cross-correlation
ax = axes[0, 1]
ax.plot(t_music, r_music, color="#4CAF50",  label=f"Music  (mean={r_music.mean():.2f})")
ax.plot(t_free,  r_free,  color="#FF9800",  label=f"Free play (mean={r_free.mean():.2f})", alpha=0.8)
ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.8, label="Reference r=0.3")
ax.set_title("Windowed Cross-Correlation (Movement)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Peak |r|")
ax.legend(fontsize=8)
ax.set_ylim(0, 1)

# C: Power spectral density
ax = axes[1, 0]
for sig, label, color in [(vm_music_c1, "Music", "#2196F3"), (vm_free_c1, "Free play", "#FF9800")]:
    f, p = welch(sig, fs=FS_ACC, nperseg=256)
    mask = f <= 10
    ax.semilogy(f[mask], p[mask], label=label, color=color)
ax.axvline(2.0, color="red", linestyle="--", linewidth=1, label="Beat freq (2 Hz)")
ax.set_title("Power Spectral Density")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (g²/Hz)")
ax.legend(fontsize=8)

# D: Feature importances
ax = axes[1, 1]
importances.head(7).plot(kind="barh", ax=ax, color="#9C27B0", edgecolor="white")
ax.set_title("RF Feature Importance\n(Music vs. Free Play)")
ax.set_xlabel("Importance")
ax.invert_yaxis()

plt.tight_layout()
import os, pathlib as _pl
OUT_DIR = _pl.Path(r"C:/Users/91948/Desktop/Portfolio/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "movement_synchrony_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figure saved to {out_path}")

print(f"""
=== SUMMARY ===
  Music cross-correlation:     {r_music.mean():.3f}
  Free play cross-correlation: {r_free.mean():.3f}
  RQA %REC music:  {rqa_rec_music:.1f}% | free: {rqa_rec_free:.1f}%
  RQA  DET music:  {rqa_det_music:.1f}% | free: {rqa_det_free:.1f}%
""")