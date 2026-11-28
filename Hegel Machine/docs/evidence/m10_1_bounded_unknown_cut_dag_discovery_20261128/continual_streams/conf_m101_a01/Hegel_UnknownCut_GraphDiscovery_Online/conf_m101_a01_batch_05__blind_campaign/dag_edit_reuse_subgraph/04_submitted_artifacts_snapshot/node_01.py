import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, i] * x[:, j]
                            for i in range(x.shape[1]) for j in range(i, x.shape[1]))))


def _response_design(x, direction):
    h = x @ (direction / max(np.linalg.norm(direction), 1e-12))
    h2 = h * h
    return np.column_stack((_quadratic(x), x * h2[:, None], h2 * h2))


def _response(x, xt, y):
    quadratic = _fit(_quadratic(x), y)[1 + x.shape[1]:]
    matrix = np.zeros((x.shape[1], x.shape[1]))
    for c, (i, j) in zip(quadratic, zip(*np.triu_indices(x.shape[1]))):
        matrix[i, j] = matrix[j, i] = c if i == j else c / 2
    _, axes = np.linalg.eigh(matrix)

    def residual(direction):
        a = _response_design(x, direction)
        return a @ _fit(a, y) - y

    # Use the finite polynomial limit of the gate, without cancelling nearly
    # collinear columns or extrapolating from large fitted coefficients.
    starts = [scale * axis for scale in (0.45, 1.3) for axis in (axes[:, 0], axes[:, -1])]
    best = min((least_squares(residual, d, bounds=(-6, 6), max_nfev=100, ftol=1e-7)
                for d in starts), key=lambda r: r.cost)
    return _response_design(xt, best.x) @ _fit(_response_design(x, best.x), y)


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    x, xt = inputs['node_00']['q_tr'], inputs['node_00']['q_te']
    y_hat = _response(x, xt, train['y'])
    state_pred = _quadratic(xt) @ _fit(_quadratic(x), train['traj'])
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
