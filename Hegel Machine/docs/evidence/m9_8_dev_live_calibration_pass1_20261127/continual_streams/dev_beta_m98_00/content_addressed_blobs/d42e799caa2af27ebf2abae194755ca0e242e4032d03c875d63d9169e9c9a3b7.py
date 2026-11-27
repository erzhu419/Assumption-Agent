"""Fit and invert the four-channel rosette in reference coordinates."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares

_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4
_PREV = (_CHANNEL - 1) % 4
_MONOMIALS = [terms for d in (2, 3)
              for terms in combinations_with_replacement(range(4), d)]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _cyclic_design(x, j):
    own, following, previous = x[:, j], x[:, _NEXT[j]], x[:, _PREV[j]]
    return np.column_stack((np.ones(len(x)), own, following, previous,
                            own * following, own ** 3))


def _full_design(x):
    return np.column_stack((_affine(x), *(np.prod(x[:, t], axis=1) for t in _MONOMIALS)))


def _robust_fit(designs, y):
    """Share weights across channels to reject simultaneous EMI spikes."""
    def solve(weights):
        w = np.sqrt(weights)
        return np.column_stack([
            np.linalg.lstsq(a * w[:, None], y[:, j] * w, rcond=None)[0]
            for j, a in enumerate(designs)
        ])

    weights = np.ones(len(y))
    floor = 1e-10 * max(1., np.median(np.max(np.abs(y), axis=1)))
    for _ in range(18):
        coeff = solve(weights)
        residual = np.column_stack([a @ coeff[:, j] for j, a in enumerate(designs)]) - y
        row_error = np.sqrt(np.mean(residual ** 2, axis=1))
        scale = max(np.median(row_error), floor)
        weights = 1. / (1. + (row_error / (2.5 * scale)) ** 2)
    clean = row_error <= 8. * scale
    if clean.sum() >= coeff.shape[0] and all(
            np.linalg.matrix_rank(a[clean]) == a.shape[1] for a in designs):
        coeff = solve(clean)
    return coeff, clean


class _Rosette:
    def __init__(self, x, y):
        sparse = [_cyclic_design(x, j) for j in _CHANNEL]
        coeff, clean = _robust_fit(sparse, y)
        self.coeff = np.zeros((5 + len(_MONOMIALS), 4))
        self.coeff[0] = coeff[0]
        self.coeff[1 + _CHANNEL, _CHANNEL] = coeff[1]
        self.coeff[1 + _NEXT, _CHANNEL] = coeff[2]
        self.coeff[1 + _PREV, _CHANNEL] = coeff[3]
        for j in _CHANNEL:
            self.coeff[5 + _MONOMIALS.index(tuple(sorted((j, _NEXT[j])))), j] = coeff[4, j]
            self.coeff[5 + _MONOMIALS.index((j, j, j)), j] = coeff[5, j]

    def predict(self, x):
        return _full_design(x) @ self.coeff

    def jacobian(self, x):
        jac = np.broadcast_to(self.coeff[1:5].T, (len(x), 4, 4)).copy()
        for terms, coeff in zip(_MONOMIALS, self.coeff[5:]):
            if not np.any(coeff):
                continue
            for i, channel in enumerate(terms):
                other = terms[:i] + terms[i + 1:]
                jac[:, :, channel] += np.prod(x[:, other], axis=1)[:, None] * coeff
        return jac

    def invert(self, raw):
        arr = np.asarray(raw, dtype=float)
        target = arr.reshape(-1, 4)
        x = np.linalg.solve(self.coeff[1:5].T, (target - self.coeff[0]).T).T
        tol = 2e-12 * (1. + np.max(np.abs(target), axis=1))
        for _ in range(18):
            residual = self.predict(x) - target
            pending = np.flatnonzero(np.max(np.abs(residual), axis=1) > tol)
            if not len(pending):
                break
            base = x[pending]
            delta = np.linalg.solve(self.jacobian(base), residual[pending, :, None])[:, :, 0]
            old_error = np.linalg.norm(residual[pending], axis=1)
            for shrink in range(10):
                trial = base - delta * (0.5 ** shrink)
                accept = np.linalg.norm(self.predict(trial) - target[pending], axis=1) < old_error
                x[pending[accept]] = trial[accept]
                keep = ~accept
                if not np.any(keep):
                    break
                pending, base, delta, old_error = (v[keep] for v in (pending, base, delta, old_error))
        for i in np.flatnonzero(np.max(np.abs(self.predict(x) - target), axis=1) > tol):
            x[i] = least_squares(lambda z: self.predict(z[None, :])[0] - target[i], x[i],
                                 jac=lambda z: self.jacobian(z[None, :])[0],
                                 gtol=1e-13, ftol=1e-13, xtol=1e-13).x
        return x.reshape(arr.shape)


def calibrate_upstream(study: dict) -> dict:
    """Return a calibrated copy of the study without changing its source data."""
    sweeps = study['sensor_calibration_sweeps']
    x = np.asarray(sweeps['ref_rosette'], dtype=float)
    y = np.asarray(sweeps['raw_rosette'], dtype=float)
    fit = _Rosette(x, y)
    calibrated = dict(study, y_obs_raw=fit.invert(study['y_obs_raw']))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        calibrated['reference_tracer_pulses'] = [
            dict(p, y_pulse_raw=None if p['y_pulse_raw'] is None else fit.invert(p['y_pulse_raw']))
            for p in pulses
        ]
    normalized = dict(sweeps, raw_rosette=sweeps['ref_rosette'].copy())
    banks = sweeps.get('single_bank_sweeps')
    if banks is not None:
        normalized['single_bank_sweeps'] = dict(banks, raw_readout=banks['ref_signal'].copy())
    calibrated['sensor_calibration_sweeps'] = normalized
    return calibrated
