"""Fit the scalar readout and two temporal modes using the calibrated kernels."""
import numpy as np
from scipy.linalg import solve


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    features = inputs['node_00']
    readout = solve(features['readout_train'], np.asarray(train['y'], dtype=float), assume_a='pos')
    traj = np.asarray(train['traj'], dtype=float)
    modes = features['modes']
    blocks = []
    for K, mode in zip(features['state_train'], modes):
        coef = solve(K, traj @ mode, assume_a='pos')
        blocks.append(np.outer(coef, mode))
    return {'readout_coef': readout, 'state_coef': np.concatenate(blocks, axis=0)}
