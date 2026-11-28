import numpy as np
from scipy.optimize import least_squares


def _basis(u, q):
    return np.column_stack((np.ones(len(u)), u, np.tanh(q[0]*u), u/(1+q[1]*u*u)))


def run_node(inputs: dict) -> dict:
    p = inputs['node_02']
    u, v, z = p['b_u'], p['b_v'], p['z']

    def fit(q):
        a = _basis(u, q)
        return a, np.linalg.lstsq(a, v, rcond=None)[0]

    def residual(q):
        a, w = fit(q)
        return (a @ w - v).ravel()

    q = least_squares(residual, [3.5, 6.], bounds=([.1, .1], [8., 15.])).x
    _, w = fit(q)
    return {'z': z, 'b': _basis(z, q) @ w}
