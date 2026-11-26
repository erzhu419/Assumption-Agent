"""Invert real acoustic sources from phase-calibrated receiver measurements."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _path_laplacian(n):
    off = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return diags((off, degree, off), (-1, 0, 1), format='csc')


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'], dtype=float)
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    face_length = np.zeros_like(n_sq)
    face_length[0, :] += hy
    face_length[-1, :] += hy
    face_length[:, 0] += hx
    face_length[:, -1] += hx
    diagonal = -hx * hy * k**2 * n_sq - 1j * k * problem['beta_imp'] * face_length
    operator = (hy / hx * kron(_path_laplacian(nx), eye(ny), format='csc')
                + hx / hy * kron(eye(nx), _path_laplacian(ny), format='csc')
                + diags(diagonal.ravel(), format='csc'))
    basis = np.asarray(problem['basis'], dtype=float)
    rhs = (hx * hy * basis.reshape(basis.shape[0], nx * ny).T).astype(complex)
    return splu(operator).solve(rhs).reshape(nx, ny, basis.shape[0])


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _fit_sources(response, observed, alpha):
    # Both quadratures of a receiver must be downweighted together.
    n, m = response.shape
    design = np.concatenate((response.real, response.imag), axis=0)
    target = np.concatenate((observed.real, observed.imag))
    penalty = np.sqrt(alpha) * np.eye(m)
    floor = 64 * np.finfo(float).eps * np.max(np.abs(observed))

    def fit(weight, ridge=True):
        w = np.tile(np.sqrt(weight), 2)
        x, y = design * w[:, None], target * w
        if ridge and alpha > 0:
            x = np.vstack((x, penalty))
            y = np.concatenate((y, np.zeros(m)))
        return np.linalg.lstsq(x, y, rcond=None)[0]

    a = fit(np.ones(n), ridge=False)
    for _ in range(30):
        r = np.abs(response @ a - observed)
        cutoff = max(1.5 * np.median(r), floor, np.finfo(float).tiny)
        updated = fit(cutoff / np.maximum(r, cutoff), ridge=False)
        if np.linalg.norm(updated - a) <= 1e-10 * max(1., np.linalg.norm(a)):
            a = updated
            break
        a = updated

    # Refit on the consensus for full efficiency under small receiver noise.
    previous = None
    for _ in range(8):
        r = np.abs(response @ a - observed)
        keep = r <= max(4.0 * np.median(r), floor)
        if np.array_equal(keep, previous) or keep.sum() < m:
            break
        a = fit(keep)
        previous = keep
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Return the real source coefficients, shared phase, and field magnitude."""
    pilots = problem['pilot_calibration']
    known = np.asarray(pilots['known_field_re']) + 1j * np.asarray(pilots['known_field_im'])
    phase = float(np.angle(np.vdot(known, _polar(pilots))))
    if phase <= -np.pi:
        phase = float(np.pi)

    field_response = _responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = _polar(survey) * np.exp(-1j * phase)
    a = _fit_sources(field_response[i, j], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(field_response @ a)}
