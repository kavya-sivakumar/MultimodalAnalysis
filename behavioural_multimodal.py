# ============================================================
# Behavioural Coding + Physiological Fusion Pipeline
# ============================================================
# Research context:
#   Do children show more prosocial behaviour following 
#   a performance art experience? 
#
#   This pipeline links observer-coded behavioural
#   events (sharing, helping, eye contact) to concurrent HRV,
#   using time-locked averaging and mixed-effects models.
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
import pingouin as pg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------
# 1. SYNTHETIC DATA — behavioural event log + continuous HRV
# ----------------------------------------------------------------
# Simulate a behavioural coding session
#
#
# Behavioural events coded:
#   "share"   — child offers toy / food to another child
#   "help"    — child assists another without being asked
#   "eye_contact" — sustained mutual gaze
#   "reject"  — child refuses to engage
#===================================================================

N_PARTICIPANTS = 30
FS_HRV = 4          # Hz
RECORDING_SEC = 300 # 5 minutes per participant

def simulate_participant(pid, condition, seed):
    rng = np.random.default_rng(seed)

    # Continuous HRV (RR in ms, resampled to 4 Hz)
    t = np.linspace(0, RECORDING_SEC, RECORDING_SEC * FS_HRV)
    rsa = 40 * np.sin(2 * np.pi * 0.25 * t)
    # Children in "arts" condition have slightly higher HRV
    base = 750 + (30 if condition == "arts" else 0)
    hrv = base + rsa + rng.normal(0, 25, len(t))

    # Behavioural events — arts condition has more prosocial events
    prosocial_rate = 0.025 if condition == "arts" else 0.012
    antisocial_rate = 0.005
    n_prosocial = rng.poisson(RECORDING_SEC * prosocial_rate)
    n_antisocial = rng.poisson(RECORDING_SEC * antisocial_rate)
    events_prosocial = rng.choice(["share", "help", "eye_contact"],
                                  size=n_prosocial)
    events_antisocial = rng.choice(["reject"], size=n_antisocial)

    onsets_pro = sorted(rng.uniform(10, RECORDING_SEC - 10, n_prosocial))
    onsets_anti = sorted(rng.uniform(10, RECORDING_SEC - 10, n_antisocial))

    rows = []
    for onset, etype in zip(onsets_pro, events_prosocial):
        rows.append({"participant_id": pid, "condition": condition,
                     "event_type": etype, "valence": "prosocial",
                     "onset_sec": onset, "duration_sec": rng.uniform(1, 4)})
    for onset, etype in zip(onsets_anti, events_antisocial):
        rows.append({"participant_id": pid, "condition": condition,
                     "event_type": etype, "valence": "antisocial",
                     "onset_sec": onset, "duration_sec": rng.uniform(0.5, 2)})

    return hrv, pd.DataFrame(rows)

# Generate data for all participants
all_hrv = {}
all_events = []
for i in range(N_PARTICIPANTS):
    cond = "arts" if i < 15 else "control"
    hrv, events = simulate_participant(pid=i, condition=cond, seed=SEED + i)
    all_hrv[i] = hrv
    all_events.append(events)

events_df = pd.concat(all_events, ignore_index=True)
print(f"Events DataFrame: {events_df.shape[0]} events across {N_PARTICIPANTS} participants")
print(events_df.groupby(["condition", "valence"])["event_type"].count())

# ----------------------------------------------------------------
# 2. INTER-RATER RELIABILITY — Cohen's Kappa
# ----------------------------------------------------------------
# Simulate two raters coding the same 50 clips; compute Cohen's kappa

n_clips = 50
categories = ["prosocial", "antisocial", "neutral"]
# Rater 1 "ground truth", Rater 2 agrees ~80% of the time
rater1 = np.random.choice(categories, size=n_clips, p=[0.5, 0.2, 0.3])
rater2 = np.where(np.random.rand(n_clips) < 0.80, rater1,
                  np.random.choice(categories, size=n_clips, p=[0.5, 0.2, 0.3]))

kappa = pg.intraclass_corr  # placeholder; use below for kappa
from sklearn.metrics import cohen_kappa_score
kappa_val = cohen_kappa_score(rater1, rater2)
print(f"\nInter-rater Cohen's Kappa: {kappa_val:.3f}  (> 0.6 = substantial agreement)")

# ----------------------------------------------------------------
# 3. TIME-LOCKED HRV AVERAGING (ERP-style)
# ----------------------------------------------------------------
# Extract HRV epochs around each behavioural event 
# Window: -5s to +15s relative to event onset

PRE_SEC  = 5
POST_SEC = 15
WIN_SAMPLES = (PRE_SEC + POST_SEC) * FS_HRV

epochs_prosocial, epochs_antisocial = [], []

for _, row in events_df.iterrows():
    pid = row["participant_id"]
    onset_sample = int(row["onset_sec"] * FS_HRV)
    start = onset_sample - PRE_SEC * FS_HRV
    end   = onset_sample + POST_SEC * FS_HRV
    if start >= 0 and end <= len(all_hrv[pid]):
        epoch = all_hrv[pid][start:end]
        if row["valence"] == "prosocial":
            epochs_prosocial.append(epoch)
        else:
            epochs_antisocial.append(epoch)

epochs_pro  = np.array(epochs_prosocial)
epochs_anti = np.array(epochs_antisocial)
t_epoch     = np.linspace(-PRE_SEC, POST_SEC, WIN_SAMPLES)

# Baseline-correct each epoch (subtract mean of pre-event window)
baseline_samples = PRE_SEC * FS_HRV
epochs_pro  = epochs_pro  - epochs_pro[:,  :baseline_samples].mean(axis=1, keepdims=True)
epochs_anti = epochs_anti - epochs_anti[:, :baseline_samples].mean(axis=1, keepdims=True)

print(f"\nEpochs extracted: {len(epochs_pro)} prosocial, {len(epochs_anti)} antisocial")

# ----------------------------------------------------------------
# 4. BUILD PARTICIPANT-LEVEL FEATURE TABLE
# ----------------------------------------------------------------
# One row per participant: behavioural counts + mean HRV features

participant_features = []
for pid in range(N_PARTICIPANTS):
    p_events = events_df[events_df["participant_id"] == pid]
    hrv_sig  = all_hrv[pid]
    cond     = p_events["condition"].iloc[0] if len(p_events) > 0 else "control"

    n_prosocial  = (p_events["valence"] == "prosocial").sum()
    n_antisocial = (p_events["valence"] == "antisocial").sum()
    prosocial_rate = n_prosocial / (RECORDING_SEC / 60)  # per minute

    # HRV features
    hrv_mean  = np.mean(hrv_sig)
    hrv_std   = np.std(hrv_sig)
    rmssd     = np.sqrt(np.mean(np.diff(hrv_sig)**2))  # RMSSD proxy

    participant_features.append({
        "participant_id":  pid,
        "condition":       cond,
        "n_prosocial":     n_prosocial,
        "n_antisocial":    n_antisocial,
        "prosocial_rate":  prosocial_rate,
        "hrv_mean":        hrv_mean,
        "hrv_std":         hrv_std,
        "rmssd":           rmssd,
    })

features_df = pd.DataFrame(participant_features)
print("\nParticipant feature table:")
print(features_df.groupby("condition")[["prosocial_rate", "rmssd"]].mean().round(2))

# ----------------------------------------------------------------
# 5. LINEAR MIXED-EFFECTS MODEL
# ----------------------------------------------------------------
#    LME: does condition predict prosocial rate?
#    Random intercept per participant (only condition as predictor used here)

# Reshape: one row per event (long format) for a proper LME
event_level = events_df.copy()
# Add HRV at event onset as a predictor
hrv_at_onset = []
for _, row in event_level.iterrows():
    pid = row["participant_id"]
    sample = int(row["onset_sec"] * FS_HRV)
    hrv_window = all_hrv[pid][max(0, sample-4*FS_HRV):sample]
    hrv_at_onset.append(np.mean(hrv_window) if len(hrv_window) > 0 else np.nan)

event_level["hrv_before"] = hrv_at_onset
event_level["is_prosocial"] = (event_level["valence"] == "prosocial").astype(int)

# LME: prosocial ~ condition + HRV_before + (1|participant)
lme = smf.mixedlm("hrv_before ~ C(condition) + is_prosocial",
                   data=event_level.dropna(),
                   groups=event_level.dropna()["participant_id"])
lme_result = lme.fit(reml=True)
print("\n=== Mixed-Effects Model: HRV ~ condition + prosocial ===")
print(lme_result.summary().tables[1])

# ----------------------------------------------------------------
# 6. RANDOM FOREST — predict condition from behavioural + HRV features
# ----------------------------------------------------------------
# Cross-validated RF classifier

X = features_df[["n_prosocial", "n_antisocial", "prosocial_rate",
                  "hrv_mean", "hrv_std", "rmssd"]].values
y = LabelEncoder().fit_transform(features_df["condition"])  # arts=0, control=1

rf = RandomForestClassifier(n_estimators=200, random_state=SEED)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scores = cross_val_score(rf, X, y, cv=cv, scoring="balanced_accuracy")

print(f"\nRandom Forest (5-fold CV):")
print(f"  Balanced accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

rf.fit(X, y)
feat_names = ["n_prosocial", "n_antisocial", "prosocial_rate", "hrv_mean", "hrv_std", "rmssd"]
importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
print(f"\nFeature importances:\n{importances.round(3).to_string()}")

# ----------------------------------------------------------------
# 7. VISUALISATION
# ----------------------------------------------------------------

# Palette mapping 
COND_PALETTE = {"arts": "#2196F3", "control": "#FF5722"}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Behavioural Coding + Physiological Fusion", fontsize=14,
             fontweight="bold")

# A: Time-locked HRV averages
ax = axes[0, 0]
if len(epochs_pro) > 0:
    mean_pro = epochs_pro.mean(axis=0)
    se_pro   = epochs_pro.std(axis=0) / np.sqrt(len(epochs_pro))
    ax.plot(t_epoch, mean_pro, color="#2196F3", label=f"Prosocial (n={len(epochs_pro)})")
    ax.fill_between(t_epoch, mean_pro - se_pro, mean_pro + se_pro,
                    alpha=0.2, color="#2196F3")

if len(epochs_anti) > 0:
    mean_anti = epochs_anti.mean(axis=0)
    se_anti   = epochs_anti.std(axis=0) / np.sqrt(len(epochs_anti))
    ax.plot(t_epoch, mean_anti, color="#FF5722", label=f"Antisocial (n={len(epochs_anti)})")
    ax.fill_between(t_epoch, mean_anti - se_anti, mean_anti + se_anti,
                    alpha=0.2, color="#FF5722")
else:
    ax.text(0.5, 0.5, "No antisocial epochs extracted\n(expected with low antisocial rate)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="gray", style="italic")

ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Event onset")
ax.axvspan(-PRE_SEC, 0, alpha=0.05, color="gray")
ax.set_xlabel("Time relative to event onset (s)")
ax.set_ylabel("Baseline-corrected RR (ms)")
ax.set_title("Time-locked HRV Averaging")
ax.legend(fontsize=8)

# B: Prosocial rate by condition
ax = axes[0, 1]
sns.boxplot(data=features_df, x="condition", y="prosocial_rate",
            hue="condition", palette=COND_PALETTE, legend=False, ax=ax)
sns.stripplot(data=features_df, x="condition", y="prosocial_rate",
              color="black", size=4, alpha=0.6, ax=ax)
ax.set_title("Prosocial Rate by Condition")
ax.set_ylabel("Events per minute")
ax.set_xlabel("")

# C: RMSSD by condition
ax = axes[1, 0]
sns.boxplot(data=features_df, x="condition", y="rmssd",
            hue="condition", palette=COND_PALETTE, legend=False, ax=ax)
sns.stripplot(data=features_df, x="condition", y="rmssd",
              color="black", size=4, alpha=0.6, ax=ax)
ax.set_title("RMSSD (HRV) by Condition")
ax.set_ylabel("RMSSD (ms)")
ax.set_xlabel("")

# D: Feature importances
ax = axes[1, 1]
importances.plot(kind="barh", ax=ax, color="#9C27B0", edgecolor="white")
ax.set_title("Random Forest Feature Importance\n(Predicting Arts vs. Control)")
ax.set_xlabel("Importance")
ax.invert_yaxis()

plt.tight_layout()
import os, pathlib as _pl
OUT_DIR = _pl.Path(r"C:/Users/91948/Desktop/Portfolio/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "behavioural_multimodal_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Figure saved to {out_path}")

