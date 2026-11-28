"""Calibrate opposite-coordinate pairs and fit endpoint and temporal readouts."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import minimize_scalar


def _least_squares(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibration(panel):
    u, v = (np.asarray(panel[key], float) for key in ('u', 'v'))
    displacement = np.linalg.norm(v - u, axis=1)
    med = np.median(displacement)
    mad = np.median(np.abs(displacement - med))
    keep = displacement <= med + 8 * max(1.4826 * mad, 1e-3)
    u, v = u[keep], v[keep]
    pairs = [(u[:, i], u[:, (i + 2) % u.shape[1]]) for i in range(u.shape[1])]

    def profile(w, coefficients=False):
        blocks = [np.column_stack((np.ones(len(p)), p, q, np.sin(w*p), np.sin(w*q)))
                  for p, q in pairs]
        c = np.stack([_least_squares(a, v[:, j]) for j, a in enumerate(blocks)])
        if coefficients:
            return c
        return sum(np.sum((a @ c[j] - v[:, j])**2) for j, a in enumerate(blocks))

    w = minimize_scalar(profile, bounds=(1.7, 3.1), method='bounded').x
    return w, profile(w, True)


def _invert(x, w, c):
    z = x.copy()
    i = np.arange(x.shape[1])
    j = (i + 2) % x.shape[1]
    jac = np.zeros((len(x), x.shape[1], x.shape[1]))
    for _ in range(30):
        p, q = z, z[:, j]
        r = c[:, 0] + c[:, 1]*p + c[:, 2]*q + c[:, 3]*np.sin(w*p) + c[:, 4]*np.sin(w*q) - x
        jac[:, i, i] = c[:, 1] + c[:, 3]*w*np.cos(w*p)
        jac[:, i, j] = c[:, 2] + c[:, 4]*w*np.cos(w*q)
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= np.clip(step, -0.5, 0.5)
        if not step.size or np.max(np.abs(step)) < 1e-12:
            break
    return z


def _features(z, frequency):
    terms = [z[:, i]*z[:, j] for i, j in combinations_with_replacement(range(z.shape[1]), 2)]
    return np.column_stack((np.ones(len(z)), z, *terms, np.tanh(frequency*z), np.sin(frequency*z)))


def run_node(inputs: dict) -> dict:
    study, raw = inputs['__input__'], inputs['node_00']
    x_train, x_test = (np.asarray(raw[k], float)[:, 1:] for k in ('a', 'a_test'))
    w, c = _calibration(study['panel_a'])
    z = _invert(np.concatenate((x_train, x_test)), w, c)
    a = _features(z, float(study['telemetry_vec'][0]))
    n = len(x_train)
    train = study['train']
    y, traj = np.asarray(train['y'], float), np.asarray(train['traj'], float)
    endpoint = _least_squares(a[:n], y)
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state = _least_squares(a[:n], traj @ modes.T) @ modes
    return {'test_features': a[n:], 'coefficients': np.column_stack((endpoint, state))}
