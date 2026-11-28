"""Scalar kernel ridge regression on the frozen spatial and operator features."""
import numpy as np
from scipy.linalg import solve


def _unit_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features, dtype=float), where=norms > 0.0)


def run_node(inputs: dict) -> dict:
    p = inputs['node_00']
    st, qt = p['st'], p['qt']
    train_unit, query_unit = _unit_rows(st), _unit_rows(qt)
    gram = p['K'] + 3.0 * (st @ st.T) + 0.3 * (train_unit @ train_unit.T) ** 3
    cross = p['Kq'] + 3.0 * (qt @ st.T) + 0.3 * (query_unit @ train_unit.T) ** 3
    weights = solve(gram + 0.001 * np.eye(len(gram)), p['y_centered'], assume_a='pos')
    return {'y_hat': cross @ weights + p['y_mean']}
