"""Finite-volume acoustic source inversion from calibrated polar receivers."""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _axis_flux(n, conductance):
    links = np.full(n - 1, conductance)
    diagonal = np.r_[links, 0.0] + np.r_[0.0, links]
    return diags((-links, diagonal, -links), (-1, 0, 1), shape=(n, n))


def _unit_source_fields(problem):
    index_squared = np.asarray(problem['n_sq'])
    nx, ny = index_squared.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    area = hx * hy

    # Each interior face contributes one symmetric flux pair. Corners
    # receive both exterior-face contributions to the radiation term.
    wall_length = np.zeros((nx, ny))
    wall_length[0, :] += hy
    wall_length[-1, :] += hy
    wall_length[:, 0] += hx
    wall_length[:, -1] += hx
    potential = -area * k**2 * index_squared - 1j * k * problem['beta_imp'] * wall_length
    operator = (kron(_axis_flux(nx, hy / hx), eye(ny), format='csc')
                + kron(eye(nx), _axis_flux(ny, hx / hy), format='csc')
                + diags(potential.ravel(), format='csc'))
    basis = np.asarray(problem['basis'])
    rhs = area * basis.reshape(len(basis), nx * ny).T
    fields = splu(operator).solve(rhs.astype(np.complex128))
    return fields.reshape(nx, ny, len(basis))


def _complex_readings(panel):
    return np.asarray(panel['amplitude_obs']) * np.exp(1j * np.asarray(panel['wrapped_phase_obs']))


def _fit_real_sources(design, observations, alpha):
    """Cauchy IRLS with one weight for both components of each receiver."""
    n, m = design.shape
    full_design = np.vstack((design.real, design.imag, np.sqrt(alpha) * np.eye(m)))
    target = np.r_[observations.real, observations.imag, np.zeros(m)]

    def fit(weights):
        root = np.sqrt(weights)
        rows = np.r_[root, root, np.ones(m)]
        return np.linalg.lstsq(full_design * rows[:, None], target * rows, rcond=None)[0]

    coefficients = fit(np.ones(n))
    floor = np.finfo(float).eps * max(np.max(np.abs(observations)), 1.0)
    # Re-estimate scale during the warm start so the initial, contaminated
    # least-squares residual does not set the final rejection threshold.
    for _ in range(8):
        residual = np.abs(design @ coefficients - observations)
        scale = max(3.0 * np.median(residual), floor)
        coefficients = fit(1.0 / (1.0 + (residual / scale)**2))

    # Fix the robust scale for the final objective and iterate to convergence.
    scale = max(3.0 * np.median(np.abs(design @ coefficients - observations)), floor)
    for _ in range(25):
        residual = np.abs(design @ coefficients - observations)
        updated = fit(1.0 / (1.0 + (residual / scale)**2))
        change = np.linalg.norm(updated - coefficients)
        coefficients = updated
        if change <= 1e-10 * max(1.0, np.linalg.norm(coefficients)):
            break
    return coefficients


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover the real source coefficients, pilot phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _complex_readings(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)

    fields = _unit_source_fields(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    observations = _complex_readings(survey) * np.exp(-1j * phase)
    amplitudes = _fit_real_sources(fields[i, j], observations, survey['tikhonov_alpha'])
    return {
        'source_amplitudes': amplitudes,
        'global_phase_rad': phase,
        'predicted_field_abs': np.abs(fields @ amplitudes),
    }
