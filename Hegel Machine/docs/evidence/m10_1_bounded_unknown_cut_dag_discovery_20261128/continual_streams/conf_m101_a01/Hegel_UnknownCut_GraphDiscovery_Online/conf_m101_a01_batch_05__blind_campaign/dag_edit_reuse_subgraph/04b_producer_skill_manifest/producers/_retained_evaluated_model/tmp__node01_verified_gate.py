import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, i] * x[:, j]
                            for i in range(x.shape[1]) for j in range(i, x.shape[1]))))


def _response_design(x, p):
    h = p[0] + x @ p[1:]
    return np.column_stack((_quadratic(x), _affine(x) * np.tanh(h)[:, None]))


def _response(x, xt, y):
    def residual(p):
        a = _response_design(x, p)
        return a @ _fit(a, y) - y

    rng = np.random.default_rng(495)
    starts = [np.r_[0., v] for v in 1.7 * np.eye(x.shape[1])]
    starts += [rng.normal(size=x.shape[1] + 1) * 1.7 for _ in range(4)]
    best = min((least_squares(residual, p, bounds=(-7, 7), max_nfev=125, ftol=2e-7)
                for p in starts), key=lambda r: r.cost)
    return _response_design(xt, best.x) @ _fit(_response_design(x, best.x), y)


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    x, xt = inputs['node_00']['q_tr'], inputs['node_00']['q_te']
    y_hat = _response(x, xt, train['y'])
    state_pred = _quadratic(xt) @ _fit(_quadratic(x), train['traj'])
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
