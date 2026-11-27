"""Calibrate a four-channel bridge in load coordinates."""
import numpy as np


_CHANNELS = np.arange(4)
_NEXT = (_CHANNELS + 1) % 4
_PREV = (_CHANNELS - 1) % 4


def _fit(load, raw):
    # A channel sees its own load, its two neighbors, a pair interaction
    # with the next channel, and a cubic self response.
    basis = np.stack((np.ones_like(load), load, load[:, _NEXT], load[:, _PREV],
                      load * load[:, _NEXT], load**3), axis=-1)
    center = np.median(raw, axis=0)
    scale = np.maximum(np.median(np.abs(raw - center), axis=0), 1e-12)
    # Screen the gross bridge excursions before they can distort the fit.
    eligible = np.max(np.abs(raw - center) / scale, axis=1) <= 12
    keep = eligible.copy()
    for _ in range(5):
        coef = np.column_stack([
            np.linalg.lstsq(basis[keep, j], raw[keep, j], rcond=None)[0]
            for j in _CHANNELS
        ])
        resid = np.max(np.abs(np.einsum('nji,ij->nj', basis, coef) - raw), axis=1)
        updated = eligible & (resid <= max(1e-10, 6 * np.median(resid[eligible])))
        if np.array_equal(updated, keep):
            break
        keep = updated
    bias, own, ahead, behind, pair, cubic = coef
    linear = np.zeros((4, 4))
    linear[_CHANNELS, _CHANNELS] = own
    linear[_NEXT, _CHANNELS] = ahead
    linear[_PREV, _CHANNELS] = behind
    return bias, linear, pair, cubic


def _forward(load, model):
    bias, linear, pair, cubic = model
    return bias + load @ linear + pair * load * load[:, _NEXT] + cubic * load**3


def _inverse(raw, model):
    bias, linear, pair, cubic = model
    obs = np.asarray(raw, dtype=float).reshape(-1, 4)
    load = np.linalg.solve(linear.T, (obs - bias).T).T.copy()
    for _ in range(20):
        residual = _forward(load, model) - obs
        if np.max(np.abs(residual), initial=0) <= 1e-11:
            break
        jac = np.broadcast_to(linear.T, (len(load), 4, 4)).copy()
        jac[:, _CHANNELS, _CHANNELS] += pair * load[:, _NEXT] + 3 * cubic * load**2
        jac[:, _CHANNELS, _NEXT] += pair * load
        step = np.linalg.solve(jac, residual[..., None])[..., 0]
        load -= step
    return load.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return calibrated splits and calibration metadata without mutating inputs."""
    cal = dataset['calibration_load_cells']
    truth = np.asarray(cal['true_channel_loads'], dtype=float)
    model = _fit(truth, np.asarray(cal['bridge_raw'], dtype=float))
    sweeps = cal['single_channel_sweeps']
    result = dict(dataset)
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    for name in ('train', 'test', 'chamber_verification_trials'):
        split = dataset.get(name)
        if split is not None:
            result[name] = dict(split, past_channel_raw=_inverse(split['past_channel_raw'], model),
                                hold_channel_raw=_inverse(split['hold_channel_raw'], model))
    return result
