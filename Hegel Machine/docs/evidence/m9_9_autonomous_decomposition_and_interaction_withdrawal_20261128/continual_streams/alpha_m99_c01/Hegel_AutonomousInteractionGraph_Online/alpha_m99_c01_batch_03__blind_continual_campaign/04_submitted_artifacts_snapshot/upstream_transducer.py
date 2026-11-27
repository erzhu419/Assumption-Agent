"""Recover canonical channel loads from the calibrated bridge response."""

import numpy as np


_NEXT = np.array([1, 2, 3, 0])
_CHANNELS = np.arange(4)


def _basis(loads):
    return np.stack((loads, loads**3, loads * loads[:, _NEXT]), axis=2)


def _fit_bridge(loads, bridge):
    # Each output has a common affine basis, plus its own cubic and adjacent
    # channel product. Use the entire bridge residual to reject spike rows.
    terms = _basis(loads)
    common = np.column_stack((np.ones(len(loads)), terms[:, :, 0]))
    designs = [np.column_stack((common, terms[:, j, 1:])) for j in range(4)]

    def fit(w):
        return np.column_stack([
            np.linalg.lstsq(a * w[:, None], bridge[:, j] * w, rcond=None)[0]
            for j, a in enumerate(designs)
        ])

    def residual(b):
        return np.linalg.norm(np.column_stack([
            a @ b[:, j] - bridge[:, j] for j, a in enumerate(designs)
        ]), axis=1)

    b = fit(np.ones(len(loads)))
    for _ in range(20):
        r = residual(b)
        b = fit(1 / np.hypot(max(1e-7, np.median(r)), r))
    r = residual(b)
    good = r <= max(1e-6, 8 * np.median(r))
    return fit(good) if all(np.linalg.matrix_rank(a[good]) == a.shape[1] for a in designs) else b


def _forward(x, b):
    return b[0] + x @ b[1:5] + b[5] * x**3 + b[6] * x * x[:, _NEXT]


def _inverse(raw, b):
    raw = np.asarray(raw, dtype=float)
    y = raw.reshape(-1, 4)
    x = np.linalg.solve(b[1:5].T, (y - b[0]).T).T.copy()
    for _ in range(40):
        residual = _forward(x, b) - y
        err = np.linalg.norm(residual, axis=1)
        active = err > 1e-11 * (1.0 + np.linalg.norm(y, axis=1))
        if not np.any(active):
            return x.reshape(raw.shape)
        xa = x[active]
        ya = y[active]
        jac = np.broadcast_to(b[1:5].T, (len(xa), 4, 4)).copy()
        jac[:, _CHANNELS, _CHANNELS] += 3 * b[5] * xa**2 + b[6] * xa[:, _NEXT]
        jac[:, _CHANNELS, _NEXT] += b[6] * xa
        step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
        trial = xa - step
        for _ in range(20):
            worse = np.linalg.norm(_forward(trial, b) - ya, axis=1) >= err[active]
            if not np.any(worse):
                break
            step[worse] *= 0.5
            trial[worse] = xa[worse] - step[worse]
        x[active] = trial
    raise ValueError('Bridge inversion did not converge')


def calibrate_upstream(dataset: dict) -> dict:
    """Calibrate bridge observations without changing the input dataset."""
    cal = dataset['calibration_load_cells']
    loads = np.asarray(cal['true_channel_loads'], dtype=float)
    b = _fit_bridge(loads, np.asarray(cal['bridge_raw'], dtype=float))
    sweeps = cal['single_channel_sweeps']
    out = dict(dataset)
    out['calibration_load_cells'] = dict(
        cal, bridge_raw=cal['true_channel_loads'].copy(),
        single_channel_sweeps=dict(sweeps, cmd_stress=sweeps['true_stress'].copy()),
    )
    for name in ('train', 'test', 'chamber_verification_trials'):
        group = dataset.get(name)
        if group is not None:
            out[name] = dict(group, **{
                field: _inverse(group[field], b)
                for field in ('past_channel_raw', 'hold_channel_raw')
            })
    return out
