"""Finite-volume Helmholtz inversion from pilot-referenced polar telemetry."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _source_fields(problem):
    """Compute one complex pressure field per source basis function."""
    n_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    grid = np.arange(nx * ny).reshape(nx, ny)
    left = np.concatenate((grid[:-1, :].ravel(), grid[:, :-1].ravel()))
    right = np.concatenate((grid[1:, :].ravel(), grid[:, 1:].ravel()))
    conductance = np.repeat([hy / hx, hx / hy], [(nx - 1) * ny, nx * (ny - 1)])

    diagonal = np.array(-area * k**2 * n_sq, dtype=np.complex128, order='C')
    impedance = -1j * k * problem['beta_imp']
    diagonal[0, :] += hy * impedance
    diagonal[-1, :] += hy * impedance
    diagonal[:, 0] += hx * impedance
    diagonal[:, -1] += hx * impedance
    diagonal = diagonal.ravel()
    np.add.at(diagonal, left, conductance)
    np.add.at(diagonal, right, conductance)
    ids = grid.ravel()
    matrix = coo_matrix(
        (np.concatenate((diagonal, -conductance, -conductance)),
         (np.concatenate((ids, left, right)), np.concatenate((ids, right, left)))),
        shape=(ids.size, ids.size),
    ).tocsc()
    rhs = np.asarray(area * basis.reshape(basis.shape[0], -1).T, dtype=np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _complex_observations(channel):
    return np.asarray(channel['amplitude_obs']) * np.exp(1j * np.asarray(channel['wrapped_phase_obs']))


def _robust_sources(design, data, alpha):
    """Fit real coefficients, weighting each complex receiver as one observation."""
    n, m = design.shape
    stacked = np.concatenate((design.real, design.imag), axis=0)
    obs = np.concatenate((data.real, data.imag))
    penalty = np.sqrt(alpha) * np.eye(m)
    rhs = np.concatenate((obs, np.zeros(m)))
    floor = np.finfo(float).eps * np.max(np.abs(data))
    weights = np.ones(n)
    a = np.zeros(m)
    for i in range(30):
        scale_rows = np.sqrt(np.tile(weights, 2))
        lhs = np.vstack((stacked * scale_rows[:, None], penalty))
        new_a = np.linalg.lstsq(lhs, rhs * np.concatenate((scale_rows, np.ones(m))), rcond=None)[0]
        if i >= 6 and np.linalg.norm(new_a - a) <= 1e-10 * (1 + np.linalg.norm(new_a)):
            return new_a
        a = new_a
        residual = np.abs(design @ a - data)
        sigma = max(np.median(residual) / np.sqrt(2 * np.log(2)), floor)
        if sigma == 0:
            return a
        if i < 5:
            # Bounded influence before rejecting impulses altogether.
            weights = 1 / np.maximum(1, residual / (1.5 * sigma))
        else:
            weights = np.maximum(0, 1 - (residual / (5 * sigma))**2)**2
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover signed real sources, global phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    survey = problem['survey']
    fields = _source_fields(problem)
    i, j = np.asarray(survey['sensor_ij']).T
    data = _complex_observations(survey) * np.exp(-1j * phase)
    amplitudes = _robust_sources(fields[i, j], data, survey['tikhonov_alpha'])
    return {'source_amplitudes': amplitudes, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ amplitudes)}
