"""Fit a quadratic scalar readout and a joint linear trajectory readout."""
import numpy as np


def _predict(train, test, target, degree, ridge):
    scale = np.maximum(train.std(axis=0), 0.1) * np.sqrt(train.shape[1])
    train, test = train / scale, test / scale
    gram = (1 + train @ train.T)**degree
    gram.flat[::len(train)+1] += ridge
    center = target.mean(axis=0)
    return center + (1 + test @ train.T)**degree @ np.linalg.solve(gram, target - center)


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_00']
    b_train, b_test = upstream['b_train'], upstream['b_test']
    y = _predict(b_train, b_test, upstream['y_train'], 2, 0.03)
    train = np.column_stack((b_train, upstream['c_train']))
    test = np.column_stack((b_test, upstream['c_test']))
    traj = _predict(train, test, upstream['traj_train'], 1, 0.003)
    terminal = traj[:, -1]
    diagnostics = np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])
    return {'y_hat': y, 'state_pred': traj, 'diagnostics_vec': diagnostics}
