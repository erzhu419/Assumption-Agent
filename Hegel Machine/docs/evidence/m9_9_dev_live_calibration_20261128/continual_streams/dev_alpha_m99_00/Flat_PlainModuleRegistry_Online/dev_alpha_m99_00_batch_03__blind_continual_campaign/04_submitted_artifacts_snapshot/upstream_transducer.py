"""Recover engineering loads from the coupled four-channel bridge."""

import numpy as np


_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4
_PREV = (_CHANNEL - 1) % 4


def _basis(x):
    nxt = x[:, _NEXT]
    return np.stack((np.ones_like(x), x, nxt, x[:, _PREV], x * nxt, x**3), axis=-1)


def _fit_bridge(load, raw):
    design = _basis(load)

    def fit(weight):
        w = np.sqrt(weight)
        return np.stack([np.linalg.lstsq(design[:, j] * w[:, None], raw[:, j] * w,
                                         rcond=None)[0] for j in _CHANNEL])

    # A spike contaminates the entire row. Estimate its common weight from
    # residuals across all four channels rather than clipping raw amplitudes.
    weight = np.ones(len(load))
    for _ in range(20):
        coeff = fit(weight)
        error = np.linalg.norm(np.einsum('nck,ck->nc', design, coeff) - raw, axis=1)
        scale = max(np.median(error), 1e-10)
        updated = 1 / (1 + (error / (3 * scale))**2)
        if np.max(np.abs(updated - weight)) < 1e-5:
            break
        weight = updated
    return fit(error <= 6 * scale)


def _forward(load, coeff):
    return np.einsum('nck,ck->nc', _basis(load), coeff)


def _invert(raw, coeff):
    raw = np.asarray(raw, dtype=float)
    target = raw.reshape(-1, 4) - coeff[:, 0]
    linear = np.zeros((4, 4))
    linear[_CHANNEL, _CHANNEL] = coeff[:, 1]
    linear[_CHANNEL, _NEXT] = coeff[:, 2]
    linear[_CHANNEL, _PREV] = coeff[:, 3]
    x = np.linalg.solve(linear, target.T).T
    for _ in range(35):
        error = _forward(x, coeff) - raw.reshape(-1, 4)
        norm = np.max(np.abs(error), axis=1)
        active = norm > 1e-11
        if not active.any():
            return x.reshape(raw.shape)
        current, err = x[active], error[active]
        jac = np.broadcast_to(linear, (len(current), 4, 4)).copy()
        jac[:, _CHANNEL, _CHANNEL] += coeff[:, 4] * current[:, _NEXT] + 3 * coeff[:, 5] * current**2
        jac[:, _CHANNEL, _NEXT] += coeff[:, 4] * current
        step = np.linalg.solve(jac, err[..., None])[..., 0]
        proposed = current - step
        goal = raw.reshape(-1, 4)[active]
        # Damp Newton only on rows where the full update increases the error.
        for _ in range(16):
            back = np.max(np.abs(_forward(proposed, coeff) - goal), axis=1) > norm[active]
            if not back.any():
                break
            step[back] *= 0.5
            proposed[back] = current[back] - step[back]
        x[active] = proposed
    raise RuntimeError('Bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return the dataset in canonical load coordinates without mutating it."""
    cells = dataset['calibration_load_cells']
    coeff = _fit_bridge(np.asarray(cells['true_channel_loads'], dtype=float),
                        np.asarray(cells['bridge_raw'], dtype=float))
    out = dict(dataset)
    for name in ('train', 'test', 'chamber_verification_trials'):
        if dataset.get(name) is not None:
            split = dataset[name]
            out[name] = dict(split, **{key: _invert(split[key], coeff)
                                      for key in ('past_channel_raw', 'hold_channel_raw')})
    sweeps = cells['single_channel_sweeps']
    out['calibration_load_cells'] = dict(
        cells, bridge_raw=cells['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()))
    return out
