"""Calibrate the coupled four-channel bridge before the constitutive fit."""

import numpy as np


_CHANNELS = np.arange(4)
_NEXT = (_CHANNELS + 1) % 4
_PREV = (_CHANNELS - 1) % 4


def _design(load):
    return np.stack((np.ones_like(load), load, load[:, _NEXT], load[:, _PREV],
                     load * load[:, _NEXT], load**3), axis=-1)


def _fit_bridge(load, raw):
    design = _design(load)
    weight = np.ones(len(load))

    def fit(weight):
        return np.stack([
            np.linalg.lstsq(design[:, i] * weight[:, None], raw[:, i] * weight,
                            rcond=None)[0]
            for i in _CHANNELS
        ])

    # One robust weight for the whole readout: the observed spikes affect all
    # four channels in a row. Refit on the recovered clean observations.
    for _ in range(30):
        coef = fit(weight)
        residual = np.linalg.norm(raw - np.einsum('nif,if->ni', design, coef), axis=1)
        scale = max(np.median(residual), 1e-12)
        weight = np.sqrt(np.minimum(1., scale / np.maximum(residual, 1e-12)))
    clean = residual <= 6 * scale
    if clean.sum() >= design.shape[-1] and all(
            np.linalg.matrix_rank(design[clean, i]) == design.shape[-1] for i in _CHANNELS):
        coef = fit(clean.astype(float))
    return coef


def _inverse(raw, coef):
    raw = np.asarray(raw)
    y = raw.reshape(-1, 4)
    bias, own, nxt, prev, product, cubic = coef.T
    linear = np.zeros((4, 4))
    linear[_CHANNELS, _CHANNELS] = own
    linear[_CHANNELS, _NEXT] = nxt
    linear[_CHANNELS, _PREV] = prev
    load = np.linalg.solve(linear, (y - bias).T).T
    for _ in range(24):
        residual = np.einsum('nif,if->ni', _design(load), coef) - y
        if np.max(np.abs(residual), initial=0.) < 1e-11:
            break
        jac = np.broadcast_to(linear, (len(load), 4, 4)).copy()
        jac[:, _CHANNELS, _CHANNELS] += product * load[:, _NEXT] + 3 * cubic * load**2
        jac[:, _CHANNELS, _NEXT] += product * load
        load -= np.linalg.solve(jac, residual[..., None])[..., 0]
    return load.reshape(raw.shape)


def calibrate_upstream(dataset: dict) -> dict:
    """Return the input schema with bridge fields in true load coordinates."""
    cal = dataset['calibration_load_cells']
    coef = _fit_bridge(np.asarray(cal['true_channel_loads']), np.asarray(cal['bridge_raw']))
    out = dataset.copy()
    for name in ('train', 'test', 'chamber_verification_trials'):
        part = dataset.get(name)
        if part is not None:
            out[name] = dict(part, **{
                key: _inverse(part[key], coef)
                for key in ('past_channel_raw', 'hold_channel_raw')
            })
    sweeps = cal['single_channel_sweeps']
    out['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    return out
