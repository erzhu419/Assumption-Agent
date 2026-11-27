"""Robust four-channel rosette calibration in reference coordinates."""
from itertools import combinations_with_replacement

import numpy as np


_EYE = np.eye(4, dtype=int)
_POWERS = np.array([
    _EYE[list(indices)].sum(axis=0)
    for degree in range(4)
    for indices in combinations_with_replacement(range(4), degree)
])


# Four self-cubics and the adjacent products on the rosette ring.
_ADJACENT = [_EYE[i] + _EYE[(i + 1) % 4] for i in range(4)]
_RING = _POWERS[(np.sum(_POWERS, axis=1) <= 1) | np.any(_POWERS == 3, axis=1)
                | np.any(np.all(_POWERS[:, None, :] == _ADJACENT, axis=2), axis=1)]


def _features(x, powers):
    return np.prod(x[:, None, :] ** powers, axis=-1)


def _robust_fit(design, raw):
    weights = np.ones(len(raw))
    for _ in range(20):
        coeff = np.linalg.lstsq(design * weights[:, None], raw * weights[:, None], rcond=None)[0]
        residual = np.linalg.norm(design @ coeff - raw, axis=1)
        scale = max(np.median(residual), 1e-12)
        weights = 1 / np.hypot(1, residual / scale)
    return coeff, residual


def _fit(reference, raw):
    compact = _features(reference, _RING)
    _, residual = _robust_fit(compact, raw)
    scale = max(np.median(residual), 1e-12)
    clean = residual <= 8 * scale
    coeff, _, rank, _ = np.linalg.lstsq(compact[clean], raw[clean], rcond=None)
    signal_scale = np.median(np.linalg.norm(raw - np.median(raw, axis=0), axis=1))
    if rank == compact.shape[1] and scale <= max(1e-10, 1e-3 * signal_scale):
        return _RING, coeff

    # When the ring topology does not describe the sweep, allow other cubic terms.
    # Screen on the low-order fit first; a nearly interpolating cubic can make
    # ordinary measurement noise look like an outlier by collapsing its median.
    design = _features(reference, _POWERS)
    _, residual = _robust_fit(design[:, :5], raw)
    clean = residual <= 10 * max(np.median(residual), 1e-12)
    if np.count_nonzero(clean) < design.shape[1]:
        clean[:] = True
    coeff, _ = _robust_fit(design[clean], raw[clean])
    return _POWERS, coeff


def _invert(raw, powers, coeff):
    raw = np.asarray(raw, dtype=float)
    targets = raw.reshape(-1, 4)
    if not targets.size:
        return raw.copy()
    linear = np.zeros((4, 4))
    offset = np.zeros(4)
    for power, row in zip(powers, coeff):
        degree = power.sum()
        if degree == 0:
            offset = row
        elif degree == 1:
            linear[power.argmax()] = row
    x = np.linalg.solve(linear.T, (targets - offset).T).T
    derivative_powers = np.maximum(powers[None, :, :] - _EYE[:, None, :], 0)
    derivative_coeff = powers.T[:, :, None] * coeff[None, :, :]
    residual = _features(x, powers) @ coeff - targets
    for _ in range(30):
        norms = np.linalg.norm(residual, axis=1)
        active = norms > 1e-12 * (1 + np.linalg.norm(targets, axis=1))
        if not active.any():
            return x.reshape(raw.shape)
        z, target = x[active], targets[active]
        basis = np.prod(z[:, None, None, :] ** derivative_powers, axis=-1)
        jac = np.einsum('ndp,dpc->ncd', basis, derivative_coeff)
        step = np.linalg.solve(jac, residual[active, :, None])[:, :, 0]
        best_norm = norms[active].copy()
        best_x, best_residual = z.copy(), residual[active].copy()
        for k in range(12):
            proposal = z - step * (0.5 ** k)
            new_residual = _features(proposal, powers) @ coeff - target
            new_norm = np.linalg.norm(new_residual, axis=1)
            better = new_norm < best_norm
            best_x[better], best_residual[better] = proposal[better], new_residual[better]
            best_norm[better] = new_norm[better]
            if np.all(best_norm < norms[active]):
                break
        x[active], residual[active] = best_x, best_residual
    if np.any(np.linalg.norm(residual, axis=1) > 1e-10 * (1 + np.linalg.norm(targets, axis=1))):
        raise ValueError('Rosette inversion did not converge')
    return x.reshape(raw.shape)


def calibrate_upstream(study: dict) -> dict:
    sweeps = study['sensor_calibration_sweeps']
    reference = np.asarray(sweeps['ref_rosette'], dtype=float)
    raw = np.asarray(sweeps['raw_rosette'], dtype=float)
    powers, coeff = _fit(reference, raw)
    out = dict(study, y_obs_raw=_invert(study['y_obs_raw'], powers, coeff))
    pulses = study.get('reference_tracer_pulses')
    if pulses is not None:
        out['reference_tracer_pulses'] = [
            pulse if pulse.get('y_pulse_raw') is None else
            dict(pulse, y_pulse_raw=_invert(pulse['y_pulse_raw'], powers, coeff))
            for pulse in pulses
        ]
    normalized = dict(sweeps, raw_rosette=sweeps['ref_rosette'].copy())
    banks = sweeps.get('single_bank_sweeps')
    if banks is not None:
        normalized['single_bank_sweeps'] = dict(banks, raw_readout=banks['ref_signal'].copy())
    out['sensor_calibration_sweeps'] = normalized
    return out
