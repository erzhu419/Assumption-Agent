import numpy as np
from itertools import combinations_with_replacement
_TERMS = np.array([[k.count(j) for j in range(4)] for d in range(4) for k in combinations_with_replacement(range(4),d)])

def _poly(x, exponents):
    return np.prod(x[..., None, :]**exponents, axis=-1)

def _bridge(cal):
    loads = np.asarray(cal['true_channel_loads'])
    raw = np.asarray(cal['bridge_raw'])
    sweeps = cal['single_channel_sweeps']
    coefficients = np.zeros((len(_TERMS), 4))
    for j in range(4):
        m = np.asarray(sweeps['lab_id']) == j
        isolated = np.asarray(sweeps['true_stress'])[m,None] * (np.arange(4) == j)
        x = np.vstack((loads, isolated))
        y = np.r_[raw[:, j], np.asarray(sweeps['cmd_stress'])[m]]
        keep = np.isfinite(y) & (np.abs(y - np.nanmedian(y)) < 6)
        a, y = _poly(x[keep], _TERMS), y[keep]
        coef = np.linalg.lstsq(a, y, rcond=None)[0]
        support = np.abs(coef) > 0.002
        coefficients[support, j] = np.linalg.lstsq(a[:,support], y, rcond=None)[0]
    active = np.any(coefficients != 0, axis=1)
    return _TERMS[active], coefficients[active]

def _forward(x, bridge):
    exponents, coefficients = bridge
    return _poly(x, exponents) @ coefficients

def _invert(raw, bridge):
    raw = np.asarray(raw)
    exponents, coefficients = bridge
    linear = (exponents.sum(axis=1) == 1)
    jac0 = (exponents[linear].T @ coefficients[linear]).T
    loads = (raw - coefficients[exponents.sum(axis=1) == 0].sum(axis=0)) @ np.linalg.inv(jac0).T
    axes = np.eye(4, dtype=int)
    derivatives = [np.maximum(exponents - e, 0) for e in axes]
    for _ in range(30):
        error = _forward(loads, bridge) - raw
        if np.max(abs(error)) < 1e-11:
            break
        jac = np.stack([_poly(loads, derivatives[j]) @ (coefficients * exponents[:,j,None]) for j in range(4)],axis=-1)
        step = np.linalg.solve(jac, error[...,None])[...,0]
        trial = loads - step
        for _ in range(12):
            bad = np.max(abs(_forward(trial,bridge)-raw),axis=-1,keepdims=True) > np.max(abs(error),axis=-1,keepdims=True)
            if not np.any(bad):
                break
            step = np.where(bad,step/2,step)
            trial = loads-step
        loads = trial
    return loads

def _loads(part, bridge):
    return tuple(_invert(part[key],bridge).sum(axis=-1) for key in ('past_channel_raw','hold_channel_raw'))
