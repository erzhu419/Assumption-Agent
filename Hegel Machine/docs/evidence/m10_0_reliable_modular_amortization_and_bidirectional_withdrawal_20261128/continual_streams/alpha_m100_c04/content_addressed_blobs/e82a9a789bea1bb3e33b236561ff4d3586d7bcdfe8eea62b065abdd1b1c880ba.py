"""Robust four-channel bridge calibration in the forward measurement coordinates."""
from itertools import combinations_with_replacement

import numpy as np


_TERMS = ((),) + tuple(t for degree in (1, 2, 3)
                       for t in combinations_with_replacement(range(4), degree))


def _design(x, terms=_TERMS):
    return np.column_stack([x[:, t].prod(axis=1) for t in terms])


def _fit_forward(x, raw):
    a = _design(x)
    b = np.linalg.lstsq(a, raw, rcond=None)[0]
    # A bridge spike corrupts several outputs together. Downweight the row,
    # not individual coordinates, using residuals in measurement space.
    for _ in range(50):
        r = np.linalg.norm(a @ b - raw, axis=1)
        w = 1.0 / np.hypot(1.0, r / max(2.0 * np.median(r), 1e-10))
        new = np.linalg.lstsq(a * w[:, None], raw * w[:, None], rcond=None)[0]
        delta = np.max(np.abs(new - b))
        b = new
        if delta < 1e-11:
            break
    r = np.linalg.norm(a @ b - raw, axis=1)
    keep = r <= max(8.0 * np.median(r), 1e-9)
    a, raw = a[keep], raw[keep]
    pinv = np.linalg.pinv(a)
    b = pinv @ raw
    if np.linalg.matrix_rank(a) == a.shape[1] and len(a) > a.shape[1]:
        # Retain the affine block. Drop nonlinear terms only when they cannot
        # be distinguished from calibration noise, then refit without shrinkage.
        sigma = np.linalg.norm(raw - a @ b, axis=0) / np.sqrt(len(a) - a.shape[1])
        stderr = np.linalg.norm(pinv, axis=1)[:, None] * sigma
        active = np.abs(b) > 4.0 * stderr
        active[:5] = True
        for j, sel in enumerate(active.T):
            b[~sel, j] = 0.0
            b[sel, j] = np.linalg.lstsq(a[:, sel], raw[:, j], rcond=None)[0]
    return b


def _make_inverse(b):
    active = np.any(b != 0.0, axis=1)
    terms = tuple(t for t, keep in zip(_TERMS, active) if keep)
    coef = b[active]
    deriv = []
    for j in range(4):
        cols, powers = [], []
        for i, t in enumerate(terms):
            if j in t:
                k = t.index(j)
                cols.append(coef[i] * t.count(j))
                powers.append(t[:k] + t[k + 1:])
        deriv.append((powers, np.asarray(cols).reshape(-1, 4)))

    def forward(x):
        return _design(x, terms) @ coef

    def jacobian(x):
        return np.stack([_design(x, t) @ c if t else np.zeros_like(x)
                         for t, c in deriv], axis=2)

    def newton(x, y):
        for _ in range(30):
            residual = forward(x) - y
            err = np.linalg.norm(residual, axis=1)
            work = err > 1e-11
            if not np.any(work):
                break
            rows = np.flatnonzero(work)
            jac = jacobian(x[rows])
            rhs = residual[rows, :, None]
            try:
                step = np.linalg.solve(jac, rhs)[:, :, 0]
            except np.linalg.LinAlgError:
                step = (np.linalg.pinv(jac) @ rhs)[:, :, 0]
            step /= np.maximum(1.0, np.linalg.norm(step, axis=1))[:, None]
            accepted = False
            for _ in range(12):
                z = x[rows] - step
                better = np.linalg.norm(forward(z) - y[rows], axis=1) < err[rows]
                x[rows[better]] = z[better]
                accepted |= bool(np.any(better))
                if np.all(better):
                    break
                rows, step = rows[~better], step[~better] * 0.5
            if not accepted:
                break
        return x

    def inverse(raw):
        raw = np.asarray(raw, dtype=float)
        y = raw.reshape(-1, 4)
        x = newton((y - b[0]) @ np.linalg.pinv(b[1:5]), y)
        # Where the affine starting point is outside Newton's basin, track
        # the root continuously from the unloaded bridge instead.
        bad = np.linalg.norm(forward(x) - y, axis=1) > 1e-8
        if np.any(bad):
            target = y[bad]
            z = np.zeros_like(target)
            for fraction in np.linspace(0.125, 1.0, 8):
                z = newton(z, b[0] + fraction * (target - b[0]))
            better = (np.linalg.norm(forward(z) - target, axis=1)
                      < np.linalg.norm(forward(x[bad]) - target, axis=1))
            x[np.flatnonzero(bad)[better]] = z[better]
        return x.reshape(raw.shape)

    return inverse


def calibrate_upstream(dataset: dict) -> dict:
    """Calibrate bridge readings without changing the supplied dataset."""
    cal = dataset['calibration_load_cells']
    true = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    inverse = np.copy if np.array_equal(raw, true) else _make_inverse(_fit_forward(true, raw))
    sweeps = cal['single_channel_sweeps']
    out = dict(dataset)
    out['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    for split in ('train', 'test', 'chamber_verification_trials'):
        block = dataset.get(split)
        if block is not None:
            out[split] = dict(block, past_channel_raw=inverse(block['past_channel_raw']),
                              hold_channel_raw=inverse(block['hold_channel_raw']))
    return out
