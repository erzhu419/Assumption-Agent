import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _panel_operator(panel, z):
    u, v = panel['u'], panel['v']

    def design(x, p):
        return np.column_stack((np.ones(len(x)), np.sin(p[0] * x) + p[1] * x))

    def residual(p):
        a = design(u, p)
        return (a @ _fit(a, v) - v).ravel()

    p = least_squares(residual, [2.8, -2.5], bounds=([1., -6.], [4.5, -.5])).x
    return design(z, p) @ _fit(design(u, p), v)


def _scalar_readout(g, y):
    n = len(y)
    base = np.column_stack((np.ones(len(g)), g, g[:, 0] * g[:, 1] * g[:, 3]))
    q = np.linalg.qr(base[:n], mode='reduced')[0]

    def project(a):
        return a - q @ (q.T @ a)

    r = project(y)

    def residual(p):
        a = project(np.sin(g[:n] @ p[:4] + p[4]))
        return a * (a @ r / max(a @ a, 1e-14)) - r

    starts = np.random.default_rng(2).normal(size=(12, g.shape[1] + 1))
    fits = (least_squares(residual, start, max_nfev=200) for start in starts)
    p = min(fits, key=lambda result: result.cost).x
    a = np.column_stack((base, np.sin(g @ p[:4] + p[4])))
    return a[n:] @ _fit(a[:n], y)


def run_node(inputs: dict) -> dict:
    data = inputs['node_00']
    y, traj = data['y'], data['traj']
    n = len(y)
    z = np.concatenate((data['z_tr'], data['z_te']), axis=0)
    g = _panel_operator(data['panel_b'], z)
    y_hat = _scalar_readout(g, y)
    a = np.column_stack((np.ones(len(z)), z, g))
    state_pred = a[n:] @ _fit(a[:n], traj)
    final = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])
    return dict(y_hat=y_hat, state_pred=state_pred, diagnostics_vec=diagnostics)
