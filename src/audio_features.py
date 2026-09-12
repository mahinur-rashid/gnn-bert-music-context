"""Audio front-end: log-mel / chroma / MFCC, segmentation and chord estimation.

Implements the preprocessing pipeline of Section 3:

1. resample to 22'050 Hz,
2. extract log-mel (128 bins), chroma (12 bins) and MFCC features,
3. split each track into fixed overlapping windows (default 3 s / 1.5 s hop),
4. estimate a chord label per segment by template matching on chroma,
5. (optionally) return a fixed-size mel-spectrogram patch for the CNN baseline.

A node feature vector is the concatenation

    [ mfcc_mean(20) | mfcc_std(20) | chroma_mean(12) | contrast_mean(7)
      | centroid, bandwidth, rolloff, zcr, rms (5) | chord one-hot(25) ]  = 89 dims
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .config import CFG

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# --------------------------------------------------------------------------- #
# chord vocabulary
# --------------------------------------------------------------------------- #

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_NAMES: list[str] = ["N"] + [f"{p}:maj" for p in PITCH_CLASSES] + \
                                [f"{p}:min" for p in PITCH_CLASSES]
N_CHORDS = len(CHORD_NAMES)          # 25

N_MFCC = int(CFG["audio"]["n_mfcc"])
CONT_DIM = N_MFCC * 2 + 12 + 7 + 5   # 64 continuous dims
FEATURE_DIM = CONT_DIM + N_CHORDS    # 89


def chord_templates() -> np.ndarray:
    """[25, 12] binary triad templates (row 0 = 'no chord')."""
    T = np.zeros((N_CHORDS, 12), dtype=np.float32)
    for r in range(12):
        T[1 + r, [r, (r + 4) % 12, (r + 7) % 12]] = 1.0      # major triad
        T[13 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1.0     # minor triad
    T[1:] /= np.linalg.norm(T[1:], axis=1, keepdims=True)
    return T


_TEMPLATES = chord_templates()


def estimate_chords(chroma_seg: np.ndarray, energy: np.ndarray | None = None,
                    silence_q: float = 0.08) -> np.ndarray:
    """Segment-level chord ids by cosine matching against triad templates."""
    C = np.asarray(chroma_seg, dtype=np.float32)
    norm = np.linalg.norm(C, axis=1, keepdims=True) + 1e-8
    scores = (C / norm) @ _TEMPLATES[1:].T            # [S, 24]
    ids = scores.argmax(1) + 1
    if energy is not None:
        thr = np.quantile(energy, silence_q) if len(energy) > 4 else -np.inf
        ids = np.where(energy <= thr, 0, ids)
    return ids.astype(np.int64)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_audio(path: str | Path, sr: int | None = None, max_dur_s: float | None = None,
               offset: float = 0.0) -> np.ndarray:
    import librosa

    sr = sr or int(CFG["audio"]["sr"])
    max_dur_s = CFG["audio"]["max_dur_s"] if max_dur_s is None else max_dur_s
    y, _ = librosa.load(str(path), sr=sr, mono=True, offset=offset,
                        duration=max_dur_s if max_dur_s else None)
    if y.size == 0:
        raise ValueError(f"empty audio: {path}")
    return librosa.util.normalize(y)


# --------------------------------------------------------------------------- #
# frame-level features
# --------------------------------------------------------------------------- #


def frame_features(y: np.ndarray, sr: int, n_fft: int | None = None,
                   hop_length: int | None = None) -> dict[str, np.ndarray]:
    """All frame-level descriptors, computed once per track."""
    import librosa

    a = CFG["audio"]
    n_fft = n_fft or int(a["n_fft"])
    hop_length = hop_length or int(a["hop_length"])

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=int(a["n_mels"]))
    log_mel = librosa.power_to_db(mel, ref=np.max)

    return {
        "log_mel": log_mel.astype(np.float32),
        "mfcc": librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC).astype(np.float32),
        "chroma": librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).astype(np.float32),
        "contrast": librosa.feature.spectral_contrast(S=np.sqrt(S), sr=sr).astype(np.float32),
        "centroid": librosa.feature.spectral_centroid(S=np.sqrt(S), sr=sr).astype(np.float32),
        "bandwidth": librosa.feature.spectral_bandwidth(S=np.sqrt(S), sr=sr).astype(np.float32),
        "rolloff": librosa.feature.spectral_rolloff(S=np.sqrt(S), sr=sr).astype(np.float32),
        "zcr": librosa.feature.zero_crossing_rate(y, hop_length=hop_length).astype(np.float32),
        "rms": librosa.feature.rms(S=np.sqrt(S)).astype(np.float32),
        "hop_length": hop_length,
    }


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #


def segment_bounds(n_frames: int, sr: int, hop_length: int, win_s: float,
                   hop_s: float, max_segments: int) -> list[tuple[int, int]]:
    fps = sr / hop_length
    win = max(2, int(round(win_s * fps)))
    step = max(1, int(round(hop_s * fps)))
    bounds = [(s, min(s + win, n_frames)) for s in range(0, max(1, n_frames - win // 2), step)]
    bounds = [(a, b) for a, b in bounds if b - a >= max(2, win // 3)]
    if not bounds:
        bounds = [(0, n_frames)]
    return bounds[:max_segments]


def beat_segment_bounds(y: np.ndarray, sr: int, hop_length: int,
                        max_segments: int, beats_per_segment: int = 8) -> list[tuple[int, int]]:
    """Beat-synchronous segmentation (librosa beat tracker)."""
    import librosa

    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, trim=False)
    if len(beats) < beats_per_segment * 2:
        return []
    edges = list(beats[::beats_per_segment]) + [int(len(y) / hop_length)]
    bounds = [(int(edges[i]), int(edges[i + 1])) for i in range(len(edges) - 1)]
    bounds = [(a, b) for a, b in bounds if b - a >= 2]
    return bounds[:max_segments]


def aggregate_segments(ff: dict[str, np.ndarray],
                       bounds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool frame features inside each segment -> (continuous X, chroma, energy)."""
    rows, chromas, energies = [], [], []
    for a, b in bounds:
        mf = ff["mfcc"][:, a:b]
        ch = ff["chroma"][:, a:b]
        rows.append(np.concatenate([
            mf.mean(1), mf.std(1),
            ch.mean(1),
            ff["contrast"][:, a:b].mean(1),
            [ff["centroid"][0, a:b].mean(), ff["bandwidth"][0, a:b].mean(),
             ff["rolloff"][0, a:b].mean(), ff["zcr"][0, a:b].mean(),
             ff["rms"][0, a:b].mean()],
        ]).astype(np.float32))
        chromas.append(ch.mean(1))
        energies.append(float(ff["rms"][0, a:b].mean()))
    return np.stack(rows), np.stack(chromas), np.asarray(energies, dtype=np.float32)


def mel_patch(log_mel: np.ndarray, frames: int | None = None) -> np.ndarray:
    """Fixed-size [n_mels, frames] patch for the CNN baseline (B2)."""
    frames = frames or int(CFG["audio"]["mel_frames"])
    T = log_mel.shape[1]
    if T == frames:
        out = log_mel
    elif T > frames:                       # average-pool down to `frames`
        idx = np.linspace(0, T, frames + 1).astype(int)
        out = np.stack([log_mel[:, idx[i]:max(idx[i] + 1, idx[i + 1])].mean(1)
                        for i in range(frames)], axis=1)
    else:                                  # tile / pad short clips
        reps = int(np.ceil(frames / T))
        out = np.tile(log_mel, (1, reps))[:, :frames]
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# one-shot track extraction
# --------------------------------------------------------------------------- #


def audio_params(dataset: str | None = None) -> dict:
    """Audio settings with the per-dataset overrides from config.yaml applied."""
    a = dict(CFG["audio"])
    overrides = (a.pop("per_dataset", None) or {})
    if dataset and dataset in overrides:
        a.update(overrides[dataset])
    return a


def extract_track(path: str | Path, segmentation: str = "fixed",
                  want_mel: bool = True, max_dur_s: float | None = None,
                  win_s: float | None = None, hop_s: float | None = None,
                  max_segments: int | None = None) -> dict:
    """Full front-end for one audio file."""
    a = CFG["audio"]
    sr = int(a["sr"])
    win_s = float(a["win_s"]) if win_s is None else float(win_s)
    hop_s = float(a["hop_s"]) if hop_s is None else float(hop_s)
    max_segments = int(a["max_segments"]) if max_segments is None else int(max_segments)

    y = load_audio(path, sr, max_dur_s)
    ff = frame_features(y, sr)
    n_frames = ff["mfcc"].shape[1]

    bounds = []
    if segmentation == "beat":
        bounds = beat_segment_bounds(y, sr, ff["hop_length"], max_segments)
    if not bounds:
        bounds = segment_bounds(n_frames, sr, ff["hop_length"], win_s, hop_s, max_segments)

    X, chroma_seg, energy = aggregate_segments(ff, bounds)
    chords = estimate_chords(chroma_seg, energy)

    onehot = np.zeros((len(chords), N_CHORDS), dtype=np.float32)
    onehot[np.arange(len(chords)), chords] = 1.0

    out = {
        "x": np.concatenate([X, onehot], axis=1).astype(np.float32),   # [S, 89]
        "chords": chords,
        "n_segments": int(len(chords)),
        "duration_s": float(len(y) / sr),
        "bounds_s": np.asarray([(a0 * ff["hop_length"] / sr, b0 * ff["hop_length"] / sr)
                                for a0, b0 in bounds], dtype=np.float32),
    }
    if want_mel:
        out["mel"] = mel_patch(ff["log_mel"])
    return out


def chord_sequence_names(chords: np.ndarray) -> list[str]:
    return [CHORD_NAMES[int(c)] for c in chords]
