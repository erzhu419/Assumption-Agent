"""Finite-volume Helmholtz inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _face_laplacian(n):
    # Each interior face contributes opposite fluxes to its two cells.
    off = -np.ones(n - 1)
    degree = np.zeros(n)
    degree[:-1] += 1
    degree[1:] += 1
    return diags((off, degree, off), (-1, 0, 1), shape=(n, n))


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']
    x_wall = np.zeros(nx)
    y_wall = np.zeros(ny)
    x_wall[0] += hy
    x_wall[-1] += hy
    y_wall[0] += hx
    y_wall[-1] += hx
    wall_length = x_wall[:, None] + y_wall[None, :]
    diagonal = -area * k0**2 * n_sq - 1j * k0 * problem['beta_imp'] * wall_length
    operator = ((hy / hx) * kron(_face_laplacian(nx), eye(ny))
                + (hx / hy) * kron(eye(nx), _face_laplacian(ny))
                + diags(diagonal.ravel()))
    basis = np.asarray(problem['basis'])
    rhs = area * basis.reshape(basis.shape[0], nx * ny).T
    return splu(operator.tocsc()).solve(rhs.astype(np.complex128))


def _complex_obs(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(1j * np.asarray(telemetry['wrapped_phase_obs']))


def _pilot_phase(pilot):
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_obs(pilot))))
    return np.pi if phase <= -np.pi else phase


def _fit_sources(response, observed, alpha):
    design = np.vstack((response.real, response.imag))
    target = np.concatenate((observed.real, observed.imag))
    penalty = np.sqrt(alpha) * np.eye(response.shape[1])
    rhs = np.r_[target, np.zeros(response.shape[1])]

    def fit(weights):
        root = np.sqrt(np.tile(weights, 2))
        return np.linalg.lstsq(np.vstack((design * root[:, None], penalty)),
                               np.r_[rhs[:target.size] * root, rhs[target.size:]],
                               rcond=None)[0]

    coef = fit(np.ones(observed.size))
    floor = np.finfo(float).eps * max(1.0, np.max(np.abs(observed)))
    # A receiver's real and imaginary residuals have one shared weight.
    # Huber first establishes a stable fit; bisquare then rejects impulses.
    for cutoff, redescend in ((1.8, False), (4.685, True)):
        for _ in range(35):
            magnitude = np.abs(response @ coef - observed)
            scale = max(np.median(magnitude) / np.sqrt(2 * np.log(2)), floor)
            radius = magnitude / (cutoff * scale)
            if redescend:
                weights = np.maximum(1 - radius**2, 0)**2
            else:
                weights = 1 / np.maximum(radius, 1)
            updated = fit(weights)
            if np.linalg.norm(updated - coef) <= 1e-9 * (1 + np.linalg.norm(updated)):
                coef = updated
                break
            coef = updated
    return coef


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source coefficients, oscillator phase, and pressure magnitude."""
    shape = np.shape(problem['n_sq'])
    survey = problem['survey']
    phase = _pilot_phase(problem['pilot_calibration'])
    responses = _responses(problem)
    ij = np.asarray(survey['sensor_ij'])
    indices = np.ravel_multi_index((ij[:, 0], ij[:, 1]), shape)
    observed = _complex_obs(survey) * np.exp(-1j * phase)
    coefficients = _fit_sources(responses[indices], observed, survey['tikhonov_alpha'])
    return {'source_amplitudes': coefficients,
            'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ coefficients).reshape(shape)}
