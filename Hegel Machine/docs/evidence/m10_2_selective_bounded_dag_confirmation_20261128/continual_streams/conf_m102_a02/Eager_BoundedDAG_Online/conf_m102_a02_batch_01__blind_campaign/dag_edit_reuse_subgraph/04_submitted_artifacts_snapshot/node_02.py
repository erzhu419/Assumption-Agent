"""Apply the calibrated endpoint and trajectory readouts."""
from itertools import combinations
import numpy as np


def _features(z, frequency):
    w = np.sin(frequency * z)**3
    pairs = [w[:, i]*w[:, j] for i, j in combinations(range(z.shape[1]), 2)]
    return np.column_stack((np.ones(len(z)), w, z, w*w, *pairs))


def run_node(inputs: dict) -> dict:
    a = np.asarray(inputs['node_00']['a_test'], float)
    fit = inputs['node_01']
    z = (a[:, 1:] - fit['offsets']) / fit['gains']
    fy, fs = fit['frequencies']
    y = (_features(z, fy) @ fit['endpoint_coef']).ravel()
    states = _features(z, fs) @ fit['trajectory_coef']
    final = states[:, -1]
    return {'y_hat': y, 'state_pred': states,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
