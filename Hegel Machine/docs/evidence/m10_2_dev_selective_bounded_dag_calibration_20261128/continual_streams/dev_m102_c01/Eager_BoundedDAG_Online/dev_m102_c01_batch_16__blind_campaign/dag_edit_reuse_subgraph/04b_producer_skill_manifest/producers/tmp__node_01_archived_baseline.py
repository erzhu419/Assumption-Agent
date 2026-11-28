import numpy as np
from scipy.linalg import solve


def run_node(inputs: dict) -> dict:
    p = inputs['node_00']
    sy, qy = p['sy'], p['qy']
    gram = p['K'] + 0.3 * (sy @ sy.T)
    cross = p['Kq'] + 0.3 * (qy @ sy.T)
    dual = solve(gram + 0.001 * np.eye(len(gram)), p['y_centered'], assume_a='pos')
    return {'y_hat': p['y_mean'] + cross @ dual}
