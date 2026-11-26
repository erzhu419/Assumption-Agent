"""Finite-volume Helmholtz inversion from pilot and survey phasors."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _responses(problem):
    n_sq = np.asarray(problem['n_sq'])
    nx, ny = n_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    k = problem['k0']
    impedance = -1j * k * problem['beta_imp']
    cells = np.arange(nx * ny).reshape(nx, ny)
    diag = np.asarray(-hx * hy * k**2 * n_sq, dtype=np.complex128, order="C")
    diag[[0, -1], :] += hy * impedance
    diag[:, [0, -1]] += hx * impedance
    ends0, ends1, vals = [], [], []
    for a, b, w in ((cells[:-1, :], cells[1:, :], hy / hx),
                    (cells[:, :-1], cells[:, 1:], hx / hy)):
        a, b = a.ravel(), b.ravel()
        diag.ravel()[a] += w
        diag.ravel()[b] += w
        ends0.extend((a, b))
        ends1.extend((b, a))
        vals.extend((np.full(a.size, -w), np.full(b.size, -w)))
    matrix = coo_matrix((np.concatenate([diag.ravel(), *vals]),
                         (np.concatenate([cells.ravel(), *ends0]),
                          np.concatenate([cells.ravel(), *ends1]))),
                        shape=(cells.size, cells.size)).tocsc()
    basis = np.asarray(problem['basis'])
    rhs = (hx * hy * basis.reshape(basis.shape[0], -1).T).astype(np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _phasors(telemetry):
    return np.asarray(telemetry['amplitude_obs']) * np.exp(
        1j * np.asarray(telemetry['wrapped_phase_obs']))


def _fit(a, y, alpha):
    # Pair the real and imaginary equations so each receiver gets one weight.
    x = np.stack((a.real, a.imag), axis=1)
    z = np.stack((y.real, y.imag), axis=1)
    ridge = np.sqrt(alpha) * np.eye(a.shape[1])
    def solve(w):
        sw = np.sqrt(w)
        lhs = np.vstack(((x * sw[:, None, None]).reshape(-1, a.shape[1]), ridge))
        rhs = np.r_[(z * sw[:, None]).ravel(), np.zeros(a.shape[1])]
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    coeff = solve(np.ones(len(y)))
    floor = np.finfo(float).eps * max(np.max(np.abs(y)), np.finfo(float).tiny)
    for _ in range(25):
        residual = np.abs(y - a @ coeff)
        scale = max(np.median(residual), floor)
        new = solve(np.minimum(1.0, 1.5 * scale / np.maximum(residual, floor)))
        if np.linalg.norm(new - coeff) <= 1e-8 * max(np.linalg.norm(new), floor):
            coeff = new
            break
        coeff = new

    # Refit without shrinking the weight of clean receivers. For circular
    # complex Gaussian noise, the 3.5-median gate rejects negligible clean mass.
    residual = np.abs(y - a @ coeff)
    clean = residual <= 3.5 * max(np.median(residual), floor)
    return solve(clean)


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    responses = _responses(problem)
    pilot = problem['pilot_calibration']
    known = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(known, _phasors(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    received = _phasors(survey) * np.exp(-1j * phase)
    coeff = _fit(responses[i, j], received, survey['tikhonov_alpha'])
    return {'source_amplitudes': coeff, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ coeff)}
