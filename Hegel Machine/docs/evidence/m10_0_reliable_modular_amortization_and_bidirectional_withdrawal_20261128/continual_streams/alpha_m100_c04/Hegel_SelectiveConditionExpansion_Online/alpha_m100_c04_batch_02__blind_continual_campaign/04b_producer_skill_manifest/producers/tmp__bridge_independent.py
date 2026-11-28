"""Calibrate the four-channel load-cell bridge before constitutive fitting."""

import numpy as np


def _basis(loads, channel):
    own = loads[:, channel]
    neighbor = loads[:, (channel + 1) % 4]
    return np.column_stack((np.ones(len(loads)), loads, own**3, own * neighbor))


def _fit_bridge(cal):
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    sweeps = cal['single_channel_sweeps']
    labs = np.asarray(sweeps['lab_id'])
    sweep_loads = np.eye(4)[labs] * np.asarray(sweeps['true_stress'])[:, None]
    sweep_raw = np.asarray(sweeps['cmd_stress'])
    n = len(loads)
    design = []
    for j in range(4):
        m = labs == j
        indices = np.r_[np.arange(n), n + np.flatnonzero(m)]
        design.append((_basis(np.vstack((loads, sweep_loads[m])), j),
                       np.r_[raw[:, j], sweep_raw[m]], indices))

    def regress(weights):
        coeff = []
        residual = np.zeros(len(weights))
        for a, b, idx in design:
            w = np.sqrt(weights[idx])
            c = np.linalg.lstsq(w[:, None] * a, w * b, rcond=None)[0]
            coeff.append(c)
            residual[idx] = np.maximum(residual[idx], np.abs(a @ c - b))
        return np.asarray(coeff), residual

    weights = np.ones(n + len(labs))
    for _ in range(20):
        _, r = regress(weights)
        scale = max(np.median(r), 1e-10)
        weights = 1 / (1 + (r / scale)**2)
    # Refit on the clean measurements, rejecting a bridge spike as a whole row.
    coeff, _ = regress(r <= 8 * scale)
    return coeff


def _forward(loads, coeff):
    bias, linear, cubic, cross = coeff[:, 0], coeff[:, 1:5], coeff[:, 5], coeff[:, 6]
    return bias + loads @ linear.T + cubic * loads**3 + cross * loads * np.roll(loads, -1, axis=1)


def _inverse(raw, coeff):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    linear = coeff[:, 1:5]
    x = np.linalg.solve(linear, (target - coeff[:, 0]).T).T
    j = np.arange(4)
    k = (j + 1) % 4
    for _ in range(24):
        r = _forward(x, coeff) - target
        err = np.max(np.abs(r), axis=1)
        if np.all(err < 1e-11):
            break
        jac = np.broadcast_to(linear, (len(x), 4, 4)).copy()
        jac[:, j, j] += 3 * coeff[:, 5] * x**2 + coeff[:, 6] * x[:, k]
        jac[:, j, k] += coeff[:, 6] * x
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        pending = err >= 1e-11
        for _ in range(12):
            if not np.any(pending):
                break
            idx = np.flatnonzero(pending)
            trial = x[idx] - step[idx]
            improved = np.max(np.abs(_forward(trial, coeff) - target[idx]), axis=1) < err[idx]
            x[idx[improved]] = trial[improved]
            pending[idx[improved]] = False
            step[idx[~improved]] *= 0.5
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return a dataset with load fields mapped to canonical channel loads."""
    cal = dataset['calibration_load_cells']
    coeff = _fit_bridge(cal)
    out = dict(dataset)
    for key in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(key)
        if part is not None:
            out[key] = dict(part)
            for field in ('past_channel_raw', 'hold_channel_raw'):
                out[key][field] = _inverse(part[field], coeff)
    sweeps = cal['single_channel_sweeps']
    out['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    return out
