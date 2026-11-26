"""Finite-volume acoustic source inversion from polar receiver measurements."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _phasors(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(
        1j * np.asarray(telemetry['wrapped_phase_obs']))


def _helmholtz(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    area = hx * hy
    index = np.arange(nx * ny).reshape(nx, ny)
    left = np.r_[index[:-1, :].ravel(), index[:, :-1].ravel()]
    right = np.r_[index[1:, :].ravel(), index[:, 1:].ravel()]
    conductance = np.r_[np.full((nx - 1) * ny, hy / hx),
                        np.full(nx * (ny - 1), hx / hy)]
    boundary = np.zeros((nx, ny))
    boundary[0, :] += hy
    boundary[-1, :] += hy
    boundary[:, 0] += hx
    boundary[:, -1] += hx
    diagonal = (-area * k**2 * n_sq - 1j * k * problem['beta_imp'] * boundary).ravel()
    np.add.at(diagonal, left, conductance)
    np.add.at(diagonal, right, conductance)
    cells = index.ravel()
    matrix = coo_matrix((np.r_[diagonal, -conductance, -conductance],
                         (np.r_[cells, left, right], np.r_[cells, right, left])),
                        shape=(cells.size, cells.size)).tocsc()
    return matrix, area


def _robust_sources(design, observations, alpha):
    """Fit real coefficients with a shared weight per complex observation."""
    n, m = design.shape
    real_design = np.vstack((design.real, design.imag))
    real_obs = np.r_[observations.real, observations.imag]
    penalty = np.sqrt(alpha) * np.eye(m)

    def fit(weights):
        w = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((w[:, None] * real_design, penalty)),
                               np.r_[w * real_obs, np.zeros(m)], rcond=None)[0]

    coeff = fit(np.ones(n))
    floor = np.finfo(float).eps * max(np.max(np.abs(observations)), np.finfo(float).tiny)
    for _ in range(30):
        r = np.abs(observations - design @ coeff)
        cutoff = max(1.8 * np.median(r), floor)
        updated = fit(cutoff / np.maximum(r, cutoff))
        if np.linalg.norm(updated - coeff) <= 1e-10 * np.linalg.norm(updated):
            coeff = updated
            break
        coeff = updated

    # Gross impulses do not contribute to the final inlier regression.
    for _ in range(5):
        r = np.abs(observations - design @ coeff)
        inliers = r <= max(4.0 * np.median(r), floor)
        if inliers.sum() < m:
            break
        updated = fit(inliers)
        if np.array_equal(updated, coeff):
            break
        coeff = updated
    return coeff


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    pilot = problem['pilot_calibration']
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    matrix, area = _helmholtz(problem)
    basis = np.asarray(problem['basis'])
    m, nx, ny = basis.shape
    transfer = splu(matrix).solve(area * basis.reshape(m, nx * ny).T)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    design = transfer[i * ny + j]
    observations = _phasors(survey) * np.exp(-1j * phase)
    coeff = _robust_sources(design, observations, survey['tikhonov_alpha'])
    return {'source_amplitudes': coeff, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(transfer @ coeff).reshape(nx, ny)}
