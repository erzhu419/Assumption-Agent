import numpy as np
from scipy.linalg import solve


def run_node(inputs: dict) -> dict:
    """Fit the scalar readout independently of the trajectory branch."""
    p = inputs['node_00']
    kernel, cross_kernel = p['K'], p['Kq']
    train, query = p['sy'], p['qy']
    scale = np.median(np.diag(kernel))
    # Use one training-derived scale for both blocks of the squared kernel.
    gram = kernel + 0.30 * (train @ train.T) + (kernel / scale)**2
    cross = cross_kernel + 0.30 * (query @ train.T) + (cross_kernel / scale)**2
    gram.flat[::len(gram) + 1] += 0.008
    y = p['y_centered']
    # Profile an unpenalized intercept by solving for both right-hand sides.
    u, v = solve(gram, np.column_stack((y, np.ones_like(y))), assume_a='pos').T
    offset = u.sum() / v.sum()
    return {'y_hat': (p['y_mean'] + offset) + cross @ (u - offset * v)}
