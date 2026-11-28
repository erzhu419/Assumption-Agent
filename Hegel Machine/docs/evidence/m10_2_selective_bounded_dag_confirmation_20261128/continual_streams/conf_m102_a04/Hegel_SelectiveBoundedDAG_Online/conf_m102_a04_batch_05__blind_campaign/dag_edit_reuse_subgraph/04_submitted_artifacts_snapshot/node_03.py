import numpy as np


def _fit(design, target):
    n = len(target)
    coef = np.linalg.lstsq(design[:n], target, rcond=1e-9)[0]
    return design[n:] @ coef


def run_node(inputs: dict) -> dict:
    w = inputs['node_00']['calibrated']
    labels = inputs['node_01']
    target, basis = labels['targets'], labels['basis']
    baseline = inputs['node_02']['fit']
    c, d = np.split(w, 2, axis=1)
    base = np.column_stack((np.ones(len(w)), w))
    paired = np.sum(c*d, axis=1)
    shifted = c[:, 0]*d[:, 1]
    design = np.column_stack((base, paired, shifted))
    y = _fit(design, target[:, 0])
    amps = np.linalg.lstsq(basis, target[:, 1:].T, rcond=1e-9)[0].T
    state = _fit(design, amps) @ basis.T
    y = np.where(np.isfinite(y), y, baseline[:, 0])
    state = np.where(np.isfinite(state), state, baseline[:, 1:])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
