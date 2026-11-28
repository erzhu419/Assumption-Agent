"""Estimate separate state and scalar readout coefficients."""
import numpy as np


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    features = inputs['node_00']
    state = features['z']
    traj = np.asarray(train['traj'], float)
    a = features['readout_train']
    y = np.asarray(train['y'], float)
    scale = np.maximum(np.sqrt(np.mean(a*a, axis=0)), 1e-10)
    u, s, vh = np.linalg.svd(a/scale, full_matrices=False)
    readout = (vh.T @ ((s/(s*s + .001))*(u.T @ y)))/scale
    return {'state_coef': np.linalg.lstsq(state, traj, rcond=1e-10)[0], 'readout_coef': readout}
