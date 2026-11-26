"""Finite-volume Helmholtz inversion from pilot-calibrated polar data."""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


def _face_laplacian(n):
    diagonal = np.full(n, 2.0)
    diagonal[0] -= 1.0
    diagonal[-1] -= 1.0
    return sparse.diags((-np.ones(n - 1), diagonal, -np.ones(n - 1)),
                        (-1, 0, 1), shape=(n, n), format='csc')


def _responses(problem):
    index_sq = np.asarray(problem['n_sq'])
    nx, ny = index_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    walls = np.zeros((nx, ny))
    walls[0, :] += hy
    walls[-1, :] += hy
    walls[:, 0] += hx
    walls[:, -1] += hx
    diagonal = -area * k**2 * index_sq - 1j * k * problem['beta_imp'] * walls
    matrix = ((hy / hx) * sparse.kron(_face_laplacian(nx), sparse.eye(ny))
              + (hx / hy) * sparse.kron(sparse.eye(nx), _face_laplacian(ny))
              + sparse.diags(diagonal.ravel()))
    basis = np.asarray(problem['basis']).reshape(-1, nx * ny)
    return splu(matrix.tocsc()).solve((area * basis.T).astype(np.complex128))


def _phasors(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _robust_real_fit(design, observed, alpha):
    # A sensor's real and imaginary components share an outlier weight.
    y = np.r_[observed.real, observed.imag]
    x = np.vstack((design.real, design.imag))
    penalty = np.sqrt(alpha) * np.eye(design.shape[1])

    def fit(weights):
        w = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((w[:, None] * x, penalty)),
                               np.r_[w * y, np.zeros(design.shape[1])], rcond=None)[0]

    coef = fit(np.ones(len(observed)))
    floor = max(np.finfo(float).tiny, 1e-12 * np.median(np.abs(observed)))
    # Convex, smooth approximation to absolute complex residuals for the start.
    for _ in range(25):
        radius = np.abs(design @ coef - observed)
        scale = max(np.median(radius), floor)
        nxt = fit(1 / np.hypot(1, radius / scale))
        change = np.linalg.norm(design @ (nxt - coef)) / np.sqrt(len(observed))
        coef = nxt
        if change <= 1e-5 * scale:
            break
    # Redescend once close: gross impulses no longer bias the final fit.
    for _ in range(25):
        radius = np.abs(design @ coef - observed)
        scale = max(np.median(radius), floor)
        weights = np.maximum(0, 1 - (radius / (4 * scale))**2)**2
        nxt = fit(weights)
        change = np.linalg.norm(design @ (nxt - coef)) / np.sqrt(len(observed))
        coef = nxt
        if change <= 1e-5 * scale:
            break
    return coef


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source strengths, oscillator phase, and field magnitudes."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    responses = _responses(problem)
    shape = np.shape(problem['n_sq'])
    survey = problem['survey']
    indices = np.ravel_multi_index(np.asarray(survey['sensor_ij']).T, shape)
    design = responses[indices]
    observed = np.exp(-1j * phase) * _phasors(survey)
    coef = _robust_real_fit(design, observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': coef, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ coef).reshape(shape)}
