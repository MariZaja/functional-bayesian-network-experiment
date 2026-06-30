"""
Przygotowanie danych do modelu FBN.
Wejście:  e01_audio_prep.csv, e01_video_prep.csv, e01_eeg_prep.csv
Wyjście:  dane/audio.csv, dane/video.csv, dane/eeg.csv
          każdy plik: kolumny identyfikacyjne + PC1..PC20
          PC1-PC3:  komponenty LDA (max dyskryminowalność emocji)
          PC4-PC20: komponenty PCA (wariancja rezydualna)
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

N_LDA = 3
N_COMPONENTS = 3

os.makedirs("dane", exist_ok=True)

ID_COLS = ["trial_id", "window_id", "window_start_s", "window_end_s"]
EMOTION_MAP = {"Anger": 0, "Sadness": 1, "Happiness": 2, "Calm": 3}


def load_labels() -> pd.DataFrame:
    dq = pd.read_csv("prep/e01_data_quality_prep.csv")
    dq[["trial_id", "window_id"]] = (
        dq["window_id"].str.split("_", expand=True).astype(int)
    )
    dq["window_id"] += 1
    dq["emotion_int"] = dq["window_emotion"].map(EMOTION_MAP)
    return dq[["trial_id", "window_id", "emotion_int"]]


def apply_lda_pca(df, labels_df, id_cols, sentinel=None):
    feat_cols = [c for c in df.columns if c not in id_cols]

    merged = df.merge(labels_df, on=["trial_id", "window_id"], how="inner")
    y = merged["emotion_int"].values
    X = merged[feat_cols].values.astype(float)

    if sentinel is not None:
        X[X == sentinel] = np.nan
    if np.isnan(X).any():
        X = SimpleImputer(strategy="median").fit_transform(X)

    X = StandardScaler().fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=N_LDA)
    X_lda = lda.fit_transform(X, y)

    X_out = X_lda

    out = merged[id_cols].copy().reset_index(drop=True)
    for i in range(N_COMPONENTS):
        out[f"PC{i + 1}"] = X_out[:, i]

    return out


labels = load_labels()

# ── Audio ─────────────────────────────────────────────────────────────────────
df_audio = pd.read_csv("prep/e01_audio_prep.csv")
audio_out = apply_lda_pca(df_audio, labels, ID_COLS, sentinel=-201)
audio_out.to_csv("dane/audio.csv", index=False)
print(f"Audio → dane/audio.csv   {audio_out.shape}")

# ── Video ─────────────────────────────────────────────────────────────────────
df_video = pd.read_csv("prep/e01_video_prep.csv")
video_out = apply_lda_pca(df_video, labels, ID_COLS)
video_out.to_csv("dane/video.csv", index=False)
print(f"Video → dane/video.csv   {video_out.shape}")

# ── EEG ───────────────────────────────────────────────────────────────────────
df_eeg = pd.read_csv("prep/e01_eeg_prep.csv")

EEG_ID = {
    "('trial_id', '')":       "trial_id",
    "('window_id', '')":      "window_id",
    "('window_start_s', '')": "window_start_s",
    "('window_end_s', '')":   "window_end_s",
}
df_eeg = df_eeg.rename(columns=EEG_ID)
df_eeg["trial_id"]  += 1   # EEG jest 0-indexed; labels i pozostałe modality są 1-indexed
df_eeg["window_id"] += 1
eeg_out = apply_lda_pca(df_eeg, labels, ID_COLS)
eeg_out.to_csv("dane/eeg.csv", index=False)
print(f"EEG   → dane/eeg.csv     {eeg_out.shape}")