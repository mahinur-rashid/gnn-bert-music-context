"""Baselines required by Section 8 of the brief.

    B1  majority-class / random tag predictor
    B2  CNN on the mel-spectrogram (no graph, no text)   -> see gnn_model.MelCNN
    B3  BERT-only                                        -> see train_task1.py
    B4  PCA + MLP on hand-crafted audio features (optional)
"""
from __future__ import annotations

import numpy as np

from .evaluate import multilabel_metrics, singlelabel_metrics
from .utils import LOG


# --------------------------------------------------------------------------- #
# B1 -- random / majority predictors
# --------------------------------------------------------------------------- #


def random_multilabel_baseline(y_train: np.ndarray, y_test: np.ndarray,
                               seed: int = 42) -> dict[str, float]:
    """Predict each tag independently from its training prior."""
    rng = np.random.RandomState(seed)
    prior = y_train.mean(0)
    prob = rng.rand(len(y_test), y_train.shape[1]) * 0 + prior  # score == prior
    noisy = np.clip(prior + rng.normal(0, 1e-3, prob.shape), 0, 1)
    pred = (rng.rand(*noisy.shape) < prior).astype(float)
    m = multilabel_metrics(y_test, noisy, thresholds=0.5)
    # F1 from actually sampled predictions is more faithful for a random model
    from sklearn.metrics import f1_score

    m["macro_f1"] = float(f1_score(y_test, pred, average="macro", zero_division=0))
    m["micro_f1"] = float(f1_score(y_test, pred, average="micro", zero_division=0))
    m["samples_f1"] = float(f1_score(y_test, pred, average="samples", zero_division=0))
    return m


def majority_multilabel_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Always predict every tag whose training frequency exceeds 0.5."""
    prior = y_train.mean(0)
    prob = np.tile(prior, (len(y_test), 1))
    return multilabel_metrics(y_test, prob, thresholds=0.5)


def random_singlelabel_baseline(y_train: np.ndarray, y_test: np.ndarray,
                                n_classes: int, seed: int = 42) -> dict[str, float]:
    rng = np.random.RandomState(seed)
    prior = np.bincount(y_train, minlength=n_classes).astype(float)
    prior /= prior.sum()
    prob = np.tile(prior, (len(y_test), 1)) + rng.normal(0, 1e-6, (len(y_test), n_classes))
    m = singlelabel_metrics(y_test, prob)
    sampled = rng.choice(n_classes, size=len(y_test), p=prior)
    from sklearn.metrics import accuracy_score, f1_score

    m["accuracy"] = float(accuracy_score(y_test, sampled))
    m["macro_f1"] = float(f1_score(y_test, sampled, average="macro", zero_division=0))
    m["micro_f1"] = float(f1_score(y_test, sampled, average="micro", zero_division=0))
    return m


def majority_singlelabel_baseline(y_train: np.ndarray, y_test: np.ndarray,
                                  n_classes: int) -> dict[str, float]:
    maj = int(np.bincount(y_train, minlength=n_classes).argmax())
    from sklearn.metrics import accuracy_score, f1_score

    pred = np.full(len(y_test), maj)
    prior = np.bincount(y_train, minlength=n_classes).astype(float)
    prior /= prior.sum()
    prob = np.tile(prior, (len(y_test), 1))
    m = singlelabel_metrics(y_test, prob)
    m["accuracy"] = float(accuracy_score(y_test, pred))
    m["macro_f1"] = float(f1_score(y_test, pred, average="macro", zero_division=0))
    m["micro_f1"] = float(f1_score(y_test, pred, average="micro", zero_division=0))
    return m


# --------------------------------------------------------------------------- #
# B4 -- PCA + MLP on hand-crafted features
# --------------------------------------------------------------------------- #


def pca_mlp_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    multilabel: bool = False, n_components: int = 64, seed: int = 42,
) -> dict[str, float]:
    """PCA-whitened hand-crafted features -> small MLP (scikit-learn)."""
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_components = int(min(n_components, X_train.shape[1], max(2, len(X_train) - 1)))
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=400,
                      early_stopping=True, random_state=seed),
    )
    LOG.info("B4 PCA(%d)+MLP on %s -> %s", n_components, X_train.shape, y_train.shape)
    pipe.fit(X_train, y_train)

    if multilabel:
        prob = np.asarray(pipe.predict_proba(X_test))
        if prob.ndim == 3:  # list of per-label [n, 2] arrays
            prob = prob[:, :, 1].T
        return multilabel_metrics(y_test, prob)
    prob = pipe.predict_proba(X_test)
    return singlelabel_metrics(y_test, prob)
