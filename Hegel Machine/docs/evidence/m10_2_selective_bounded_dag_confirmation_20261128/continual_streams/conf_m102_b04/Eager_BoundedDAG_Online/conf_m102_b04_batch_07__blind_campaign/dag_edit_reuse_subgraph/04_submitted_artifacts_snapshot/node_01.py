"""Predict scalar and state heads with a smooth panel-informed kernel."""
import numpy as np


def _gram(q, ref, length):
    dot = q @ ref.T
    d2 = np.maximum((q*q).sum(1)[:, None] + (ref*ref).sum(1)[None, :] - 2*dot, 0)
    return np.exp(-d2/length), dot


def run_node(inputs: dict) -> dict:
    study, panel = inputs['__input__'], inputs['node_00']
    tr = study['train']
    n = len(tr['x'])
    x = np.concatenate((tr['x'], study['test']['x']))
    # The frozen cubic basis starts with a constant and the four panel coordinates.
    u = panel['raw'][:, 1:1+x.shape[1]]
    z = np.asarray(tr['traj'])
    time_basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    targets = np.column_stack((tr['y'], z @ time_basis.T))
    k, g = _gram(x, x[:n], 3.)
    p, _ = _gram(u, u[:n], 4.)
    k += (1 + g/4)**2 + .01*p
    fit = np.linalg.solve(k[:n] + .0003*np.eye(n), targets)
    pred = k[n:] @ fit
    y, state = pred[:, 0], pred[:, 1:] @ time_basis
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
