"""Separate calibrated endpoint and low-rank trajectory readouts."""
from itertools import combinations_with_replacement
import numpy as np
from sklearn.linear_model import Lasso


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _structures(z, g):
    cyclic = z * np.roll(g, 1, axis=1)
    affine = np.column_stack((np.ones(len(z)), g, z, cyclic))
    core = np.column_stack((affine, g[:, :2] * g[:, 2:]))
    values = np.column_stack((z, g))
    terms = []
    for degree in (2, 3):
        for inds in combinations_with_replacement(range(values.shape[1]), degree):
            terms.append(np.prod(values[:, inds], axis=1))
    for i in range(values.shape[1]):
        for j in range(i):
            for sign in (-1, 1):
                h = values[:, i] + sign * values[:, j]
                for weight in (0.5, 1.0, 2.0):
                    x = weight * h
                    terms.extend((np.sin(x), np.cos(x), np.tanh(x), x / (1 + x*x)))
    for x in values.T:
        for t in values.T:
            for weight in (0.5, 1.0, 2.0):
                terms.append(x / (1 + weight * t*t))
    return affine, core, np.column_stack(terms)


def _endpoint(core, terms, train_idx, test_idx, y):
    fit = _fit(core[train_idx], np.column_stack((y, terms[train_idx])))
    target = y - core[train_idx] @ fit[:, 0]
    residual = terms - core @ fit[:, 1:]
    scale = np.maximum(residual[train_idx].std(axis=0), 1e-6)
    model = Lasso(alpha=0.001, max_iter=6000, tol=0.0003, fit_intercept=False)
    model.fit(residual[train_idx] / scale, target)
    return core[test_idx] @ fit[:, 0] + model.predict(residual[test_idx] / scale)


def run_node(inputs: dict) -> dict:
    observed = inputs['__input__']['train']
    calibrated = inputs['node_00']
    z = np.asarray(calibrated['z'], float)
    i, j = calibrated['train_idx'], calibrated['test_idx']
    g = np.asarray(inputs['node_01']['g'], float)
    y = np.asarray(observed['y'], float).reshape(-1)
    traj = np.asarray(observed['traj'], float)
    affine, core, terms = _structures(z, g)
    yp = _endpoint(core, terms, i, j, y)
    _, _, vt = np.linalg.svd(traj, full_matrices=False)
    modes = vt[:3]
    tp = (affine[j] @ _fit(affine[i], traj @ modes.T)) @ modes
    diagnostic = np.array([yp.mean(), yp.std(), tp[:, -1].mean(), tp[:, -1].std()])
    return {'y_hat': yp, 'state_pred': tp, 'diagnostics_vec': diagnostic}
