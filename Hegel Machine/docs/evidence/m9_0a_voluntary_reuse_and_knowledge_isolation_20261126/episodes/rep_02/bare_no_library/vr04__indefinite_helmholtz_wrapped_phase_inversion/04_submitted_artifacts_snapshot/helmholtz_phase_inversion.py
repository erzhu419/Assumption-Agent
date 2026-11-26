"""Finite-volume Helmholtz inversion with calibrated complex receivers."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _polar(channel):
    return np.asarray(channel['amplitude_obs']) * np.exp(1j * np.asarray(channel['wrapped_phase_obs']))


def _phase(pilot):
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _polar(pilot))))
    return np.pi if phase == -np.pi else phase


def _path_laplacian(n):
    off = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return sparse.diags((off, degree, off), (-1, 0, 1), shape=(n, n))


def _response(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    face_length = np.zeros_like(n_sq, dtype=float)
    face_length[0, :] += hy
    face_length[-1, :] += hy
    face_length[:, 0] += hx
    face_length[:, -1] += hx
    diagonal = -area * k**2 * n_sq - 1j * k * problem['beta_imp'] * face_length
    matrix = (hy / hx * sparse.kron(_path_laplacian(nx), sparse.eye(ny))
              + hx / hy * sparse.kron(sparse.eye(nx), _path_laplacian(ny))
              + sparse.diags(diagonal.ravel())).tocsc()
    basis = np.asarray(problem['basis'])
    rhs = np.asfortranarray(area * basis.reshape(basis.shape[0], nx * ny).T, dtype=np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _robust_fit(design, observed, alpha):
    """Fit real coefficients; downweight the two quadratures jointly."""
    n, m = design.shape
    x = np.concatenate((design.real, design.imag), axis=0)
    y = np.concatenate((observed.real, observed.imag))
    reg = np.sqrt(alpha) * np.eye(m)

    def solve(weights, penalty=None):
        root = np.sqrt(np.tile(weights, 2))
        a, b = x * root[:, None], y * root
        if penalty is not None:
            a = np.vstack((a, penalty))
            b = np.concatenate((b, np.zeros(m)))
        return np.linalg.lstsq(a, b, rcond=None)[0]

    amplitudes = solve(np.ones(n))
    floor = max(np.median(np.abs(observed)) * 1e-12, np.finfo(float).tiny)
    # An L1-like start keeps large spikes from setting the final residual scale.
    for _ in range(25):
        radius = np.abs(design @ amplitudes - observed)
        scale = max(np.median(radius), floor)
        updated = solve(scale / np.maximum(radius, 0.15 * scale))
        if np.linalg.norm(updated - amplitudes) <= 1e-9 * max(np.linalg.norm(amplitudes), floor):
            amplitudes = updated
            break
        amplitudes = updated
    # Fix the scale before the ridge step so shrinkage cannot inflate the cutoff.
    cutoff = 4.0 * max(np.median(np.abs(design @ amplitudes - observed)), floor)
    radius = np.abs(design @ amplitudes - observed)
    weights = np.maximum(1.0 - (np.minimum(radius, cutoff) / cutoff)**2, 0.0)**2
    return solve(weights, reg)


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    phase = _phase(problem['pilot_calibration'])
    response = _response(problem)
    survey = problem['survey']
    ij = np.asarray(survey['sensor_ij'])
    design = response[ij[:, 0], ij[:, 1], :]
    observed = _polar(survey) * np.exp(-1j * phase)
    amplitudes = _robust_fit(design, observed, survey['tikhonov_alpha'])
    return {
        'source_amplitudes': amplitudes,
        'global_phase_rad': phase,
        'predicted_field_abs': np.abs(response @ amplitudes),
    }
