"""Recover canonical channel loads from a cross-coupled bridge calibration."""
from itertools import combinations_with_replacement

import numpy as np


_TERMS = [()] + [term for degree in (1, 2, 3)
                  for term in combinations_with_replacement(range(4), degree)]
_POWERS = np.array([np.bincount(term, minlength=4) for term in _TERMS], dtype=int)


def _design(x, powers):
    return np.prod(x[:, None, :] ** powers, axis=-1)


def _fit(loads, raw):
    # Keep all quadratic interactions and the pure cubic bridge curvature.
    # Mixed cubics were not identifiable in the lower-load block audit.
    cols = np.flatnonzero((_POWERS.sum(axis=1) <= 2) |
                          (np.count_nonzero(_POWERS, axis=1) == 1))
    design = _design(loads, _POWERS[cols])
    n, p = design.shape
    # Select the fit with the smallest median row residual. Unlike an affine
    # prefilter, this separates bridge curvature from gross response spikes.
    rng = np.random.default_rng(0)
    best = np.inf
    for _ in range(96):
        rows = rng.choice(n, size=min(n, p + 4), replace=False)
        coef, _, rank, _ = np.linalg.lstsq(design[rows], raw[rows], rcond=None)
        if rank != p:
            continue
        error = np.linalg.norm(raw - design @ coef, axis=1)
        score = np.median(error)
        if score < best:
            best = score
            clean = error <= 6 * score + 1e-12
    for _ in range(6):
        coef = np.linalg.lstsq(design[clean], raw[clean], rcond=None)[0]
        error = np.linalg.norm(raw - design @ coef, axis=1)
        updated = error <= 6 * np.median(error) + 1e-12
        if np.array_equal(clean, updated):
            break
        clean = updated
    a, y = design[clean], raw[clean]
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    sigma = np.linalg.norm(y - a @ coef, axis=0) / np.sqrt(max(1, len(a) - p))
    leverage = np.linalg.norm(np.linalg.pinv(a), axis=1)
    keep = np.abs(coef) > 6 * leverage[:, None] * sigma + 1e-12
    keep[0] = True
    beta = np.zeros((len(_POWERS), raw.shape[1]))
    for i, support in enumerate(keep.T):
        beta[cols[support], i] = np.linalg.lstsq(a[:, support], y[:, i], rcond=None)[0]
    return beta, clean


def _invert(raw, beta):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4)
    used = np.any(beta != 0, axis=1)
    powers, coeff = _POWERS[used], beta[used]
    dp = np.maximum(powers[None, :, :] - np.eye(4, dtype=int)[:, None, :], 0)
    dcoeff = powers.T[:, :, None] * coeff[None, :, :]
    x = np.linalg.solve(beta[1:5].T, (target - beta[0]).T).T
    for _ in range(40):
        residual = _design(x, powers) @ coeff - target
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-11 * (1 + np.linalg.norm(target, axis=1))
        if not np.any(active):
            return x.reshape(raw.shape)
        z = x[active]
        jac = np.stack([_design(z, p) @ c for p, c in zip(dp, dcoeff)], axis=-1)
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        trial = z - step
        goal = target[active]
        bound = error[active]
        for _ in range(16):
            retry = np.linalg.norm(_design(trial, powers) @ coeff - goal, axis=1) > bound
            if not np.any(retry):
                break
            step[retry] *= .5
            trial[retry] = z[retry] - step[retry]
        x[active] = trial
    raise RuntimeError('Bridge calibration inverse did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'])
    beta, _ = _fit(loads, np.asarray(cal['bridge_raw']))
    out = dict(dataset)
    sweep = cal['single_channel_sweeps']
    out['calibration_load_cells'] = dict(
        cal, bridge_raw=loads.copy(),
        single_channel_sweeps=dict(sweep, cmd_stress=sweep['true_stress'].copy()),
    )
    for name in ('train', 'test', 'chamber_verification_trials'):
        section = dataset.get(name)
        if section is not None:
            out[name] = dict(section, **{
                key: _invert(section[key], beta)
                for key in ('past_channel_raw', 'hold_channel_raw') if key in section
            })
    return out
