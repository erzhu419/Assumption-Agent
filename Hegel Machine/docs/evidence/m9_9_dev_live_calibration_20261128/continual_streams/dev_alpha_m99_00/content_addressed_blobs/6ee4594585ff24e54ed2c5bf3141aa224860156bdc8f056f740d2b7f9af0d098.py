"""Fit and invert the coupled four-channel load-cell bridge."""
import numpy as np

_I, _J = np.triu_indices(4, 1)


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, x*x, x[:, _I]*x[:, _J], x**3))


def _fit(load, raw):
    a = _basis(load)
    b = np.linalg.lstsq(a, raw, rcond=None)[0]
    for _ in range(30):
        r = np.linalg.norm(a @ b - raw, axis=1)
        w = np.minimum(1., max(0.7*np.median(r), 1e-12)/np.maximum(r, 1e-12))
        b = np.linalg.lstsq(w[:, None]*a, w[:, None]*raw, rcond=None)[0]
    r = np.linalg.norm(a @ b - raw, axis=1)
    clean = r <= max(6*np.median(r), 1e-10)
    a, raw = a[clean], raw[clean]
    inv = np.linalg.pinv(a)
    b = inv @ raw
    # Compact each output separately; the broad fit leaves missing couplings testable.
    df = len(a) - a.shape[1]
    if df > 0:
        sigma = np.linalg.norm(a @ b - raw, axis=0) / np.sqrt(df)
        se = np.linalg.norm(inv, axis=1)[:, None]*sigma
        for k in range(4):
            keep = np.abs(b[:, k]) > 6*se[:, k]
            keep[:5] = True
            b[:, k] = 0.
            b[keep, k] = np.linalg.lstsq(a[:, keep], raw[:, k], rcond=None)[0]
    return b


def _invert(raw, b):
    shape = raw.shape
    y = np.asarray(raw, dtype=float).reshape(-1, 4)
    x = np.linalg.solve(b[1:5].T, (y-b[0]).T).T
    for _ in range(40):
        r = _basis(x) @ b - y
        norm = np.linalg.norm(r, axis=1)
        if not np.any(norm > 1e-11):
            return x.reshape(shape)
        jac = (b[1:5] + 2*x[:, :, None]*b[5:9]
               + 3*x[:, :, None]**2*b[15:19])
        for k, (i, j) in enumerate(zip(_I, _J)):
            jac[:, i] += x[:, j, None]*b[9+k]
            jac[:, j] += x[:, i, None]*b[9+k]
        step = np.linalg.solve(jac.transpose(0, 2, 1), r[..., None])[..., 0]
        step[norm <= 1e-11] = 0.
        trial = x - step
        for _ in range(16):
            worse = np.linalg.norm(_basis(trial) @ b - y, axis=1) > norm
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = x[worse] - step[worse]
        x = trial
    raise RuntimeError('Load-cell bridge inverse did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Return a copy with bridge measurements in canonical load coordinates."""
    cal = dataset['calibration_load_cells']
    load = np.asarray(cal['true_channel_loads'], dtype=float)
    raw = np.asarray(cal['bridge_raw'], dtype=float)
    b = None if np.array_equal(raw, load) else _fit(load, raw)
    result = dict(dataset)
    result['calibration_load_cells'] = {**cal, 'bridge_raw': cal['true_channel_loads'].copy()}
    sweeps = cal['single_channel_sweeps']
    result['calibration_load_cells']['single_channel_sweeps'] = {
        **sweeps, 'cmd_stress': sweeps['true_stress'].copy()}
    for name in ('train', 'test', 'chamber_verification_trials'):
        split = dataset.get(name)
        if split is not None and b is not None:
            result[name] = {**split, **{field: _invert(split[field], b)
                                       for field in ('past_channel_raw', 'hold_channel_raw')}}
    return result
