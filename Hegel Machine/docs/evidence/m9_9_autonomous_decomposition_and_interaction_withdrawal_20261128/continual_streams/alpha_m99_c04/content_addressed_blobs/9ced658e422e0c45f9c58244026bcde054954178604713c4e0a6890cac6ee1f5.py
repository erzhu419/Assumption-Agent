"""Robust calibration of the four coupled load-cell bridges."""
import numpy as np

_CHANNEL = np.arange(4)
_NEXT = (_CHANNEL + 1) % 4


def _fit_bridge(loads, raw):
    # Each channel has a cubic response and couples to its cyclic neighbor.
    affine = np.column_stack((np.ones(len(loads)), loads))
    interaction = loads * loads[:, _NEXT]
    design = [np.column_stack((affine, loads[:, j] ** 3, interaction[:, j]))
              for j in _CHANNEL]

    def fit(weights):
        return np.column_stack([
            np.linalg.lstsq(a * weights[:, None], raw[:, j] * weights,
                            rcond=None)[0]
            for j, a in enumerate(design)
        ])

    weights = np.ones(len(loads))
    for _ in range(16):
        coef = fit(weights)
        error = np.linalg.norm(
            np.column_stack([a @ coef[:, j] for j, a in enumerate(design)]) - raw,
            axis=1,
        )
        scale = max(2.0 * np.median(error), 1e-12)
        weights = 1.0 / np.hypot(1.0, error / scale)
    # Give the clean rows equal weight after identifying gross bridge spikes.
    keep = error <= 6.0 * scale
    if all(np.linalg.matrix_rank(a[keep]) == a.shape[1] for a in design):
        coef = fit(keep.astype(float))
    return coef


def _forward(loads, coef):
    return (coef[0] + loads @ coef[1:5] + coef[5] * loads ** 3
            + coef[6] * loads * loads[..., _NEXT])


def _inverse(raw, coef):
    shape = raw.shape
    target = raw.reshape(-1, 4)
    linear = coef[1:5].T
    x = np.linalg.solve(linear, (target - coef[0]).T).T.copy()
    for _ in range(40):
        residual = _forward(x, coef) - target
        error = np.linalg.norm(residual, axis=1)
        active = error > 1e-11
        if not np.any(active):
            return x.reshape(shape)
        z = x[active]
        jac = np.broadcast_to(linear, (len(z), 4, 4)).copy()
        jac[:, _CHANNEL, _CHANNEL] += 3.0 * coef[5] * z ** 2 + coef[6] * z[:, _NEXT]
        jac[:, _CHANNEL, _NEXT] += coef[6] * z
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        trial = z - step
        for _ in range(20):
            worse = np.linalg.norm(_forward(trial, coef) - target[active], axis=1) >= error[active]
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = z[worse] - step[worse]
        x[active] = trial
    raise RuntimeError('Load-cell bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return the dataset in canonical load coordinates without mutating it."""
    cal = dataset['calibration_load_cells']
    loads = cal['true_channel_loads']
    coef = _fit_bridge(loads, cal['bridge_raw'])
    sweeps = cal['single_channel_sweeps']
    result = dict(dataset)
    result['calibration_load_cells'] = dict(
        cal, bridge_raw=loads.copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    for name in ('train', 'test', 'chamber_verification_trials'):
        group = dataset.get(name)
        if group is not None:
            result[name] = dict(group, **{
                key: _inverse(group[key], coef)
                for key in ('past_channel_raw', 'hold_channel_raw')
            })
    return result
