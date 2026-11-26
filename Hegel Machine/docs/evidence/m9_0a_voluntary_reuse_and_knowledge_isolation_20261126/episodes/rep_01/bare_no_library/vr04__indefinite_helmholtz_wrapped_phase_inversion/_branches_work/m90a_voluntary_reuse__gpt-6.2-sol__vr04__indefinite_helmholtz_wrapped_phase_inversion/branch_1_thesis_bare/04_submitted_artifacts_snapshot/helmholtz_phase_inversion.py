"""Finite-volume Helmholtz source inversion from polar receiver data."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _path_laplacian(n):
    edges = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return diags((edges, degree, edges), (-1, 0, 1), shape=(n, n))


def _unit_source_fields(problem):
    n_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']

    # Kronecker order follows the C-order index i * ny + j.
    flux = ((hy / hx) * kron(_path_laplacian(nx), eye(ny))
            + (hx / hy) * kron(eye(nx), _path_laplacian(ny)))
    wall = np.zeros((nx, ny))
    wall[0, :] += hy
    wall[-1, :] += hy
    wall[:, 0] += hx
    wall[:, -1] += hx
    wave = -area * k0**2 * n_sq - 1j * k0 * problem['beta_imp'] * wall
    operator = (flux + diags(wave.ravel())).tocsc()
    rhs = np.asfortranarray(area * basis.reshape(basis.shape[0], -1).T,
                            dtype=np.complex128)
    return splu(operator).solve(rhs).reshape(nx, ny, -1)


def _complex_observations(group):
    return (np.asarray(group['amplitude_obs'])
            * np.exp(1j * np.asarray(group['wrapped_phase_obs'])))


def _fit_real_sources(response, data, alpha):
    """Robust complex regression, assigning one weight per receiver."""
    design = np.concatenate((response.real, response.imag), axis=0)
    target = np.concatenate((data.real, data.imag))
    ridge = np.sqrt(alpha) * np.eye(response.shape[1])
    zeros = np.zeros(response.shape[1])

    def fit(weights):
        w = np.tile(np.sqrt(weights), 2)
        return np.linalg.lstsq(np.vstack((w[:, None] * design, ridge)),
                               np.concatenate((w * target, zeros)), rcond=None)[0]

    def residual_and_scale(a):
        residual = np.abs(response @ a - data)
        # Rayleigh median for the radius of circular complex Gaussian noise.
        floor = np.finfo(float).eps * max(np.max(np.abs(data)), np.finfo(float).tiny)
        scale = max(np.median(residual) / np.sqrt(2 * np.log(2)), floor)
        return residual, scale

    a = fit(np.ones(data.size))
    # Huber steps give a stable starting point even when the initial fit is
    # contaminated; bisquare steps then reject the remaining large impulses.
    for redescending in (False, True):
        for _ in range(30):
            residual, scale = residual_and_scale(a)
            if redescending:
                ratio = np.minimum(residual / (4.685 * scale), 1.0)
                weights = (1 - ratio**2)**2
            else:
                cutoff = 1.5 * scale
                weights = cutoff / np.maximum(residual, cutoff)
            updated = fit(weights)
            converged = np.linalg.norm(updated - a) <= 1e-10 * (1 + np.linalg.norm(a))
            a = updated
            if converged:
                break
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Calibrate receiver phase and infer real source strengths and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_observations(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _unit_source_fields(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    data = np.exp(-1j * phase) * _complex_observations(survey)
    amplitudes = _fit_real_sources(fields[i, j], data, survey['tikhonov_alpha'])
    return {'source_amplitudes': amplitudes,
            'global_phase_rad': phase,
            'predicted_field_abs': np.abs(fields @ amplitudes)}
