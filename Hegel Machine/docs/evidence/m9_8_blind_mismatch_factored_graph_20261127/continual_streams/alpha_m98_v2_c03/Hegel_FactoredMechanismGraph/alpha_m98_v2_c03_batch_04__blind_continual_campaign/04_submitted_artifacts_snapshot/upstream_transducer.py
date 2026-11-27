"""Fit the load-cell bridge in forward coordinates and invert trial readings."""
from itertools import combinations

import numpy as np

_PAIRS = tuple(combinations(range(4), 2))


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x**2,
                            *(x[:, i] * x[:, j] for i, j in _PAIRS), x**3))


def _robust_seed(a, raw):
    # A few high-leverage spike rows can mask one another in a full fit.
    # Select a subset fit by its median residual on the full calibration set.
    rng = np.random.default_rng(21)
    n, p = a.shape

    def score(coef):
        return np.median(np.linalg.norm(raw - a @ coef, axis=1))

    best = np.linalg.lstsq(a, raw, rcond=None)[0]
    loss = score(best)
    for _ in range(96):
        ids = rng.choice(n, min(n, p + 5), replace=False)
        coef, _, rank, _ = np.linalg.lstsq(a[ids], raw[ids], rcond=None)
        if rank == p:
            candidate_loss = score(coef)
            if candidate_loss < loss:
                best, loss = coef, candidate_loss
    return best


def _fit_bridge(loads, raw):
    a = _basis(loads)
    # Spikes affect bridge readings, not the commanded calibration loads.
    coef = _robust_seed(a, raw)
    for _ in range(24):
        err = np.linalg.norm(raw - a @ coef, axis=1)
        scale = max(2 * np.median(err), 1e-12)
        weight = 1 / np.hypot(1, err / scale)
        coef = np.linalg.lstsq(a * weight[:, None], raw * weight[:, None], rcond=None)[0]
    err = np.linalg.norm(raw - a @ coef, axis=1)
    keep = err <= max(6 * np.median(err), 1e-11)
    clean_a, clean_y = a[keep], raw[keep]
    coef = np.linalg.lstsq(clean_a, clean_y, rcond=None)[0]
    residual = clean_y - clean_a @ coef
    stderr = np.sqrt(np.diag(np.linalg.pinv(clean_a.T @ clean_a)))[:, None]
    stderr = stderr * np.sqrt(np.sum(residual**2, axis=0) / max(1, len(clean_a) - a.shape[1]))
    for j in range(4):
        active = np.abs(coef[:, j]) > np.maximum(5 * stderr[:, j], 1e-12)
        active[1 + j] = True
        coef[~active, j] = 0
        coef[active, j] = np.linalg.lstsq(clean_a[:, active], clean_y[:, j], rcond=None)[0]
    return coef


def _inverse(raw, coef):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(coef[1:5].T, (y - coef[0]).T).T.copy()
    for _ in range(30):
        residual = _basis(x) @ coef - y
        error = np.linalg.norm(residual, axis=1)
        if np.all(error < 1e-11):
            break
        jac = np.broadcast_to(coef[1:5], (len(x), 4, 4)).copy()
        jac += 2 * x[:, :, None] * coef[None, 5:9]
        jac += 3 * x[:, :, None]**2 * coef[None, 15:19]
        for k, (i, j) in enumerate(_PAIRS, 9):
            jac[:, i, :] += x[:, j, None] * coef[k]
            jac[:, j, :] += x[:, i, None] * coef[k]
        step = np.linalg.solve(jac.transpose(0, 2, 1), residual[..., None])[..., 0]
        pending = np.flatnonzero(error >= 1e-11)
        for damping in 2.0**-np.arange(12):
            if not len(pending):
                break
            cand = x[pending] - damping * step[pending]
            better = np.linalg.norm(_basis(cand) @ coef - y[pending], axis=1) < error[pending]
            x[pending[better]] = cand[better]
            pending = pending[~better]
    return x.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Copy and calibrate the load fields expected by the downstream solver."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    coef = _fit_bridge(loads, np.asarray(cal['bridge_raw'], dtype=float))
    sweeps = cal['single_channel_sweeps']
    result = dict(dataset)
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=np.asarray(cal['true_channel_loads']).copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=np.asarray(sweeps['true_stress']).copy()),
    )
    for key in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(key)
        if block is not None:
            result[key] = dict(block, **{
                name: _inverse(block[name], coef)
                for name in ('past_channel_raw', 'hold_channel_raw')
            })
    return result
