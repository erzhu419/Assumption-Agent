"""
Distance metrics on Markov kernels.

All metrics take two |X|×|A| kernels and return a scalar.

Implemented:
  - Frobenius (baseline)
  - Spectral (Hausdorff on eigenvalues)
  - Fisher-Rao (per-row average, weighted by state prior)
  - KL divergence and asymmetry ratio (Perrone 2024)

Not in MVP (placeholders for later):
  - Blackwell order (LP feasibility)
  - Log-Euclidean on SPD
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _safe_row(p: np.ndarray) -> np.ndarray:
    p = p + 1e-10
    return p / p.sum()


def frobenius(K_a: np.ndarray, K_b: np.ndarray) -> float:
    return float(np.linalg.norm(K_a - K_b, "fro"))


def spectral(K_a: np.ndarray, K_b: np.ndarray) -> float:
    """Hausdorff distance on singular values (stable under non-square kernels)."""
    sa = np.sort(np.linalg.svd(K_a, compute_uv=False))
    sb = np.sort(np.linalg.svd(K_b, compute_uv=False))
    k = min(len(sa), len(sb))
    return float(np.linalg.norm(sa[:k] - sb[:k]))


def fisher_rao(K_a: np.ndarray, K_b: np.ndarray,
               state_weights: np.ndarray = None) -> float:
    """Mean Fisher-Rao distance across states."""
    n_states = K_a.shape[0]
    if state_weights is None:
        state_weights = np.ones(n_states) / n_states
    total = 0.0
    for x in range(n_states):
        p = _safe_row(K_a[x])
        q = _safe_row(K_b[x])
        bc = np.sum(np.sqrt(p * q))
        bc = np.clip(bc, 0.0, 1.0)
        total += state_weights[x] * 2.0 * np.arccos(bc)
    return float(total)


def kl_divergence(K_a: np.ndarray, K_b: np.ndarray,
                  state_weights: np.ndarray = None) -> float:
    """D_KL(K_a || K_b). Non-symmetric."""
    n_states = K_a.shape[0]
    if state_weights is None:
        state_weights = np.ones(n_states) / n_states
    total = 0.0
    for x in range(n_states):
        p = _safe_row(K_a[x])
        q = _safe_row(K_b[x])
        total += state_weights[x] * float(np.sum(p * np.log(p / q)))
    return float(total)


def kl_asymmetry(K_a: np.ndarray, K_b: np.ndarray) -> float:
    """log-ratio = log(D(a||b) / D(b||a)). Near 0 => symmetric."""
    d_ab = kl_divergence(K_a, K_b)
    d_ba = kl_divergence(K_b, K_a)
    return float(np.log((d_ab + 1e-10) / (d_ba + 1e-10)))


def all_metrics(K_a: np.ndarray, K_b: np.ndarray) -> Dict[str, float]:
    return {
        "frobenius": frobenius(K_a, K_b),
        "spectral":  spectral(K_a, K_b),
        "fisher":    fisher_rao(K_a, K_b),
        "kl_ab":     kl_divergence(K_a, K_b),
        "kl_ba":     kl_divergence(K_b, K_a),
        "kl_log_asym": kl_asymmetry(K_a, K_b),
    }
