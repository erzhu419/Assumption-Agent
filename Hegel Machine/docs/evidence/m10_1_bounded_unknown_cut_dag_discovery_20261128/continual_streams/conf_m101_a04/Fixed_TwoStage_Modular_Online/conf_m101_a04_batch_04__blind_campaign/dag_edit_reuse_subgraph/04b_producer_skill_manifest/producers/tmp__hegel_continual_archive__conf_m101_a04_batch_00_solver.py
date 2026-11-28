import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    mask = np.ones(len(u), dtype=bool)
    for _ in range(8):
        a, b = u[mask], v[mask]
        ac, bc = a-a.mean(0), b-b.mean(0)
        scale = (ac*bc).sum(0)/(ac*ac).sum(0)
        shift = b.mean(0)-a.mean(0)*scale
        r = np.linalg.norm(u*scale+shift-v, axis=1)
        mask = r <= max(5*np.median(r), .0006)
    return shift, scale



def _panel_basis(x, frequency, sine):
    features = [np.ones(len(x)), x]
    if sine:
        features.extend((np.sin(frequency*x), np.sin(2*frequency*x)))
    else:
        features.append(np.cos(frequency*x))
    return np.column_stack(features)


def _operator(u, v, x, sine):
    def fit(frequency):
        f = _panel_basis(u, frequency, sine)
        coef = np.linalg.lstsq(f, v, rcond=None)[0]
        return f, coef

    def residual(frequency):
        f, coef = fit(frequency)
        return np.mean((f@coef-v)**2)

    bounds = (2.7, 3.8) if sine else (1., 1.5)
    frequency = minimize_scalar(residual, bounds=bounds, method='bounded').x
    _, coef = fit(frequency)
    return _panel_basis(x, frequency, sine)@coef


def _readout_basis(x):
    mixed = np.column_stack([x[:, i]*x[:, j] for i, j in
                             combinations_with_replacement(range(x.shape[1]), 2)])
    return np.column_stack((np.ones(len(x)), x, .4*mixed))


def solve_batch(study: dict) -> dict:
    train = study['train']
    offset, scale = _calibrate(study['panel_a'])
    raw = np.concatenate((train['x'], study['test']['x']))
    x = (raw-offset)/scale
    c = study['panel_c']
    b = _operator(c['u'], c['v1'], x, True)
    d = _operator(c['u'], c['v2'], x, False)
    f = _readout_basis(np.column_stack((b, d)))
    n = len(train['x'])
    a, xt = f[:n], f[n:]
    gram = a@a.T
    cross = xt@a.T
    eye = np.eye(n)
    yh = cross@np.linalg.solve(gram+.0003*eye, train['y'])
    state = cross@np.linalg.solve(gram+.001*eye, train['traj'])
    end = state[:, -1]
    return {'y_hat': yh, 'state_pred': state,
            'diagnostics_vec': np.array([yh.mean(), yh.std(), end.mean(), end.std()])}
