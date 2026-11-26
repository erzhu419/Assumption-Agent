"""Finite-volume Helmholtz source inversion from calibrated polar telemetry."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(1j * np.asarray(data['wrapped_phase_obs']))


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k = problem['k0']
    idx = np.arange(nx * ny).reshape(nx, ny)
    left = np.concatenate((idx[:-1, :].ravel(), idx[:, :-1].ravel()))
    right = np.concatenate((idx[1:, :].ravel(), idx[:, 1:].ravel()))
    face = np.repeat([hy / hx, hx / hy], [(nx - 1) * ny, nx * (ny - 1)])
    diag = -area * k**2 * n_sq.astype(complex).ravel()
    diag += np.bincount(left, weights=face, minlength=idx.size)
    diag += np.bincount(right, weights=face, minlength=idx.size)
    diag = diag.reshape(idx.shape)
    impedance = -1j * k * problem['beta_imp']
    diag[0, :] += hy * impedance
    diag[-1, :] += hy * impedance
    diag[:, 0] += hx * impedance
    diag[:, -1] += hx * impedance
    flat = idx.ravel()
    matrix = coo_matrix((np.r_[diag.ravel(), -face, -face],
                         (np.r_[flat, left, right], np.r_[flat, right, left])),
                        shape=(idx.size, idx.size)).tocsc()
    basis = np.asarray(problem['basis'])
    rhs = area * basis.reshape(basis.shape[0], idx.size).T
    return splu(matrix).solve(rhs.astype(complex)).reshape(nx, ny, -1)


def _robust_sources(design, obs, alpha):
    # Treat real and imaginary parts of a receiver as one observation when weighting.
    n, m = design.shape
    x = np.stack((design.real, design.imag), axis=1)
    y = np.stack((obs.real, obs.imag), axis=1)
    eye = np.eye(m)

    def fit(weight):
        root = np.sqrt(weight)
        wx = (x * root[:, None, None]).reshape(-1, m)
        wy = (y * root[:, None]).ravel()
        return np.linalg.lstsq(np.vstack((wx, np.sqrt(alpha) * eye)),
                               np.r_[wy, np.zeros(m)], rcond=None)[0]

    rng = np.random.default_rng(39281)
    estimate = fit(np.ones(n))
    best = np.median(np.abs(design @ estimate - obs))
    for _ in range(40):
        sample = rng.choice(n, min(m + 2, n), replace=False)
        trial = np.linalg.lstsq(x[sample].reshape(-1, m), y[sample].ravel(), rcond=None)[0]
        score = np.median(np.abs(design @ trial - obs))
        if score < best:
            estimate, best = trial, score

    # For circular complex Gaussian noise the residual magnitude is Rayleigh;
    # its median equals sqrt(2 log(2)) times the component standard deviation.
    floor = np.finfo(float).eps * max(1., np.max(np.abs(obs)))
    for _ in range(25):
        residual = np.abs(design @ estimate - obs)
        cutoff = max(4.685 * np.median(residual) / np.sqrt(2 * np.log(2)), floor)
        weight = np.maximum(0., 1. - (residual / cutoff)**2)**2
        updated = fit(weight)
        if np.linalg.norm(updated - estimate) <= 1e-10 * max(1., np.linalg.norm(updated)):
            return updated
        estimate = updated
    return estimate


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Return real source coefficients, calibrated phase, and pressure magnitude."""
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    field_basis = _responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    obs = _polar(survey) * np.exp(-1j * phase)
    coefficients = _robust_sources(field_basis[i, j], obs, survey['tikhonov_alpha'])
    return {'source_amplitudes': coefficients,
            'global_phase_rad': phase,
            'predicted_field_abs': np.abs(field_basis @ coefficients)}
