"""Finite-volume Helmholtz source inversion from polar receiver telemetry."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_stiffness(n):
    # Path graph Laplacian: only interior faces contribute here.
    main = np.full(n, 2.0)
    main[[0, -1]] = 1.0
    off = -np.ones(n - 1)
    return diags((off, main, off), (-1, 0, 1), shape=(n, n), format="csc")


def _field_responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k0 = problem['k0']
    cell_area = hx * hy
    lap = (hy / hx * kron(_axis_stiffness(nx), eye(ny), format="csc")
           + hx / hy * kron(eye(nx), _axis_stiffness(ny), format="csc"))
    wall_length = np.zeros((nx, ny))
    wall_length[0, :] += hy
    wall_length[-1, :] += hy
    wall_length[:, 0] += hx
    wall_length[:, -1] += hx
    diagonal = -cell_area * k0**2 * n_sq - 1j * k0 * problem['beta_imp'] * wall_length
    operator = lap + diags(diagonal.ravel(), format="csc")
    basis = np.asarray(problem['basis'], dtype=np.complex128)
    rhs = cell_area * basis.reshape(basis.shape[0], nx * ny).T
    return splu(operator).solve(rhs).reshape(nx, ny, basis.shape[0])


def _complex_observations(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _robust_real_fit(response, observed, alpha):
    # Fit both quadratures with a single weight per receiver. The pilot has
    # already determined the oscillator phase, so the coefficients are real.
    design = np.concatenate((response.real, response.imag), axis=0)
    target = np.concatenate((observed.real, observed.imag))
    m = response.shape[1]
    penalty = np.sqrt(alpha) * np.eye(m)

    def fit(weights):
        root = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((root[:, None] * design, penalty)),
                               np.concatenate((root * target, np.zeros(m))), rcond=None)[0]

    a = fit(np.ones(len(observed)))
    floor = np.finfo(float).eps * max(1.0, np.max(np.abs(observed)))
    # Huber IRLS gets a stable residual scale before rejecting impulses.
    for _ in range(30):
        residual = np.abs(response @ a - observed)
        scale = max(np.median(residual), floor)
        updated = fit(np.minimum(1.0, 1.5 * scale / np.maximum(residual, floor)))
        if np.linalg.norm(updated - a) <= 1e-9 * max(1.0, np.linalg.norm(a)):
            a = updated
            break
        a = updated
    residual = np.abs(response @ a - observed)
    inliers = residual <= 4.0 * max(np.median(residual), floor)
    return fit(inliers) if np.count_nonzero(inliers) >= m else a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Estimate real source amplitudes, oscillator phase and field magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _field_responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = np.exp(-1j * phase) * _complex_observations(survey)
    a = _robust_real_fit(fields[i, j], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ a)}
