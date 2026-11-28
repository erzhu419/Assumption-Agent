"""Predict scalar and time-series labels in the calibrated cubic coordinates."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _kernel(x, q):
    delta = -.02*(x[:, None, :] - q[None, :, :])**2
    return .5*(np.exp(delta.mean(axis=2)) + np.exp(delta).mean(axis=2))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    features = inputs['node_00']
    x = features['cubic_train']
    q = features['cubic_test']
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-12)
    x, q = (x-mean)/scale, (q-mean)/scale
    y = np.asarray(study['train']['y'], float).reshape(-1)
    z = np.asarray(study['train']['traj'], float)
    values = np.column_stack((y,z))
    center = values.mean(axis=0)
    k = _kernel(x,x)
    k.flat[::len(k)+1] += 3e-5
    coeff = cho_solve(cho_factor(k, check_finite=False), values-center, check_finite=False)
    pred = center + _kernel(q,x) @ coeff
    yh, zh = pred[:,0], pred[:,1:]
    return {'y_hat': yh, 'state_pred': zh,
            'diagnostics_vec': np.array([yh.mean(), yh.std(), zh[:,-1].mean(), zh[:,-1].std()])}
