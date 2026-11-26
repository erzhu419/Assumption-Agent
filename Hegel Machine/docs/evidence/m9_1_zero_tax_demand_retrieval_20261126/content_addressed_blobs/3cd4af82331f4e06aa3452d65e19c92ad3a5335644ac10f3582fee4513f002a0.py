"""Finite-volume acoustic source inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import splu


def _axis_stiffness(n, weight):
    edge = np.full(n - 1, -weight)
    diagonal = np.zeros(n)
    diagonal[:-1] += weight
    diagonal[1:] += weight
    return diags((edge, diagonal, edge), (-1, 0, 1), format='csc')


def _basis_response(problem):
    medium = np.asarray(problem['n_sq'], dtype=float)
    basis = np.asarray(problem['basis'])
    nx, ny = medium.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    volume = hx * hy
    diagonal = (-volume * k**2 * medium).astype(np.complex128)
    flux = -1j * k * problem['beta_imp']
    diagonal[0, :] += hy * flux
    diagonal[-1, :] += hy * flux
    diagonal[:, 0] += hx * flux
    diagonal[:, -1] += hx * flux
    # C-order: y varies fastest, and each corner gets two wall fluxes.
    helmholtz = kronsum(_axis_stiffness(ny, hx / hy),
                        _axis_stiffness(nx, hy / hx), format='csc')
    helmholtz = helmholtz + diags(diagonal.ravel(), format='csc')
    rhs = volume * basis.reshape(len(basis), nx * ny).T
    response = splu(helmholtz).solve(rhs.astype(np.complex128))
    return response.reshape(nx, ny, len(basis))


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _robust_real_fit(response, observed, alpha):
    # Treat each complex residual as one observation when assigning weights.
    m = response.shape[1]
    ridge = np.sqrt(alpha) * np.eye(m)
    target = np.r_[observed.real, observed.imag, np.zeros(m)]
    matrix = np.vstack((response.real, response.imag, ridge))

    def fit(weights):
        root = np.sqrt(weights)
        scale = np.r_[root, root, np.ones(m)]
        return np.linalg.lstsq(matrix * scale[:, None], target * scale, rcond=None)[0]

    estimate = fit(np.ones(len(observed)))
    floor = np.finfo(float).eps * max(np.max(np.abs(observed)), np.finfo(float).tiny)
    for _ in range(50):
        residual = np.abs(observed - response @ estimate)
        cutoff = max(1.6 * np.median(residual), floor)
        updated = fit(cutoff / np.maximum(residual, cutoff))
        converged = np.linalg.norm(updated - estimate) <= 1e-9 * np.linalg.norm(updated)
        estimate = updated
        if converged:
            break

    # Refit the inlier channels without Huber shrinkage of their residuals.
    for _ in range(5):
        residual = np.abs(observed - response @ estimate)
        keep = residual <= max(3.5 * np.median(residual), floor)
        updated = fit(keep)
        converged = np.linalg.norm(updated - estimate) <= 1e-9 * np.linalg.norm(updated)
        estimate = updated
        if converged:
            break
    return estimate


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover real source coefficients, oscillator phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    response = _basis_response(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observed = _polar(survey) * np.exp(-1j * phase)
    coefficients = _robust_real_fit(response[i, j], observed, survey['tikhonov_alpha'])
    return {
        'source_amplitudes': coefficients,
        'global_phase_rad': phase,
        'predicted_field_abs': np.abs(response @ coefficients),
    }
