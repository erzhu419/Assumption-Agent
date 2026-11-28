"""Calibrated scalar and trajectory readouts for the frozen sensor encoder."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _panel_map(panel):
    u, v = (np.asarray(panel[key], float) for key in ('u', 'v'))
    affine = _fit(_affine(u), v)

    def basis(x, k):
        return np.column_stack((_affine(x), np.sin(k*x)))

    def loss(k):
        a = basis(u, k)
        return np.mean((a @ _fit(a, v) - v)**2)

    k = minimize_scalar(loss, bounds=(2., 4.), method='bounded').x
    coef = _fit(basis(u, k), v)
    return affine, lambda x: basis(x, k) @ coef


def _scalar(x, z, xt, zt, y):
    i, j = np.triu_indices(x.shape[1], 1)
    level = max(float(np.max(np.abs(y))), 1e-12)
    result = []
    for form, initial, limits in [('rational', .34, (.03, 3.)), ('sine', 3., (1., 6.))]:
        for mixed in (False, True):
            def design(x, z, k):
                w = x**3/(1+k*x**4) if form == 'rational' else np.sin(k*x)
                cross = (x[:, i]*z[:, j], z[:, i]*x[:, j]) if mixed else (z[:, i]*z[:, j],)
                return np.column_stack((_affine(x), w, *cross))

            def profile(p):
                k, ratio = p
                a = design(x, z, k)
                cap = level * ratio
                coef = _fit(a, np.arctanh(y/cap))
                return cap, a, coef

            def error(p):
                cap, a, coef = profile(p)
                return cap*np.tanh(a@coef)-y

            opt = least_squares(error, [initial, 1.35], bounds=([limits[0], 1.001], [limits[1], 8.]), max_nfev=75)
            cap, _, coef = profile(opt.x)
            result.append(cap*np.tanh(design(xt, zt, opt.x[0]) @ coef))
    return np.mean(result, axis=0)


def _trajectory(x, z, xt, zt, traj):
    def features(x, k):
        return np.column_stack((_affine(x), np.sin(k*x)))

    def loss(k):
        a = features(x, k)
        return np.mean((a @ _fit(a, traj) - traj)**2)

    k = minimize_scalar(loss, bounds=(2., 4.), method='bounded').x
    adjusted = _affine(z)
    corrected = _affine(zt) @ _fit(adjusted, traj)
    flexible = features(xt, k) @ _fit(features(x, k), traj)
    return .5*(corrected + flexible)


def run_node(inputs: dict) -> dict:
    study, encoded = inputs['__input__'], inputs['node_00']
    b, correct = _panel_map(study['panel_b'])
    pre, pret = [np.linalg.solve(b[1:].T, (np.asarray(encoded[key], float)-b[0]).T).T for key in ('z', 'zt')]
    y = _scalar(pre, correct(pre), pret, correct(pret), np.asarray(study['train']['y'], float))
    state = _trajectory(pre, correct(pre), pret, correct(pret), np.asarray(study['train']['traj'], float))
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
