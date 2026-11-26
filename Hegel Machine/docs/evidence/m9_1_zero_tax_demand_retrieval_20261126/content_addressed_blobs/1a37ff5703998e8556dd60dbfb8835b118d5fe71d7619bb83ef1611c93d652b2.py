"""Finite-volume Helmholtz inversion from polar receiver measurements."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def _responses(problem):
    """Return the complex pressure response to each unit source."""
    index_sq = np.asarray(problem['n_sq'])
    basis = np.asarray(problem['basis'])
    nx, ny = index_sq.shape
    hx, hy = problem['Lx'] / nx, problem['Ly'] / ny
    area = hx * hy
    k0 = problem['k0']
    ids = np.arange(nx * ny).reshape(nx, ny)
    cells = ids.ravel()
    left = np.concatenate((ids[:-1, :].ravel(), ids[:, :-1].ravel()))
    right = np.concatenate((ids[1:, :].ravel(), ids[:, 1:].ravel()))
    conductance = np.concatenate((np.full((nx - 1) * ny, hy / hx),
                                  np.full(nx * (ny - 1), hx / hy)))
    diagonal = np.array(-area * k0**2 * index_sq, dtype=np.complex128)
    flux = -1j * k0 * problem['beta_imp']
    diagonal[0, :] += hy * flux
    diagonal[-1, :] += hy * flux
    diagonal[:, 0] += hx * flux
    diagonal[:, -1] += hx * flux
    diagonal = diagonal.ravel()
    np.add.at(diagonal, left, conductance)
    np.add.at(diagonal, right, conductance)
    matrix = coo_matrix((np.concatenate((diagonal, -conductance, -conductance)),
                         (np.concatenate((cells, left, right)),
                          np.concatenate((cells, right, left)))),
                        shape=(cells.size, cells.size)).tocsc()
    rhs = np.asarray(area * basis.reshape(basis.shape[0], -1).T,
                     dtype=np.complex128)
    return splu(matrix).solve(rhs).reshape(nx, ny, basis.shape[0])


def _polar(data):
    return np.asarray(data['amplitude_obs']) * np.exp(
        1j * np.asarray(data['wrapped_phase_obs']))


def _fit(operator, target, alpha):
    """Weight each receiver as a pair of real and imaginary residuals."""
    n, m = operator.shape
    matrix = np.vstack((operator.real, operator.imag, np.sqrt(alpha) * np.eye(m)))
    observation = np.concatenate((target.real, target.imag, np.zeros(m)))

    def solve(weights):
        s = np.sqrt(weights)
        w = np.concatenate((s, s, np.ones(m)))
        return np.linalg.lstsq(matrix * w[:, None], observation * w, rcond=None)[0]

    a = solve(np.ones(n))
    floor = 32 * np.finfo(float).eps * np.max(np.abs(target))
    # Start with bounded-influence weights, then remove gross impulses entirely.
    for cutoff, hard_rejection in ((1.5, False), (5.0, True)):
        for _ in range(25):
            r = np.abs(operator @ a - target)
            scale = max(np.median(r) / np.sqrt(2.0 * np.log(2.0)), floor)
            if scale == 0:
                return a
            if hard_rejection:
                ratio = np.minimum(r / (cutoff * scale), 1.0)
                weights = (1.0 - ratio**2)**2
            else:
                weights = cutoff * scale / np.maximum(r, cutoff * scale)
            next_a = solve(weights)
            done = np.linalg.norm(next_a - a) <= 1e-10 * np.linalg.norm(a)
            a = next_a
            if done:
                break
    return a


def invert_helmholtz_sources_and_phase(problem: dict) -> dict:
    """Recover the real source amplitudes, common receiver phase, and field modulus."""
    pilot = problem['pilot_calibration']
    reference = np.asarray(pilot['known_field_re']) + 1j * np.asarray(pilot['known_field_im'])
    phase = float(np.angle(np.vdot(reference, _polar(pilot))))
    if phase <= -np.pi:
        phase = float(np.pi)
    responses = _responses(problem)
    survey = problem['survey']
    i, j = np.asarray(survey['sensor_ij']).T
    target = np.exp(-1j * phase) * _polar(survey)
    a = _fit(responses[i, j], target, survey['tikhonov_alpha'])
    return {'source_amplitudes': a, 'global_phase_rad': phase,
            'predicted_field_abs': np.abs(responses @ a)}
