import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ols(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibrate(panel, frequency_hint):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    j = np.arange(u.shape[1])
    pair = (j + 2) % u.shape[1]

    def regress(freq, mask=None):
        coeff = []
        residual = []
        for k, p in zip(j, pair):
            x = np.column_stack((np.ones(len(u)), u[:, k],
                                 np.sin(freq[0] * u[:, k]),
                                 np.sin(freq[1] * u[:, p])))
            y = v[:, k]
            if mask is None:
                c = _ols(x, y)
                for _ in range(24):
                    w = (1 + ((x @ c - y) / .008)**2)**(-.25)
                    c = _ols(x * w[:, None], y * w)
            else:
                keep = mask[:, k]
                c = _ols(x[keep], y[keep])
            coeff.append(c)
            residual.append(x @ c - y)
        return np.array(coeff).T, np.column_stack(residual)

    initial = (max(2.8, frequency_hint), 2.4)
    bounds = ([1.3, 1.3], [4.5, 4.5])
    fit = least_squares(lambda f: regress(f)[1].ravel(), initial,
                        bounds=bounds, loss='soft_l1', f_scale=.01)
    mask = np.abs(regress(fit.x)[1]) < .10
    refined = least_squares(lambda f: regress(f, mask)[1][mask], fit.x,
                            bounds=bounds)
    return refined.x, regress(refined.x, mask)[0]


def _invert(x, frequency, coefficients):
    b, g, a, c = coefficients
    f, h = frequency
    j = np.arange(x.shape[1])
    pair = (j + 2) % x.shape[1]
    z = (x - b) / g
    for _ in range(30):
        residual = b + g*z + a*np.sin(f*z) + c*np.sin(h*z[:, pair]) - x
        if np.max(np.abs(residual)) < 1e-11:
            break
        jac = np.zeros((len(z), len(j), len(j)))
        jac[:, j, j] = g + f*a*np.cos(f*z)
        jac[:, j, pair] = h*c*np.cos(h*z[:, pair])
        z -= np.clip(np.linalg.solve(jac, residual[..., None])[..., 0], -.5, .5)
    return z


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return x[:, i] * x[:, j]


def _normalize(w, n, mode):
    if mode == 0:
        return w
    w = w - w[:n].mean(axis=0)
    if mode == 1:
        return w
    if mode == 2:
        return w / np.maximum(w[:n].std(axis=0), .1)
    _, s, v = np.linalg.svd(w[:n], full_matrices=False)
    return (w @ v.T) / np.maximum(s / np.sqrt(n), .07)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    n = len(train)
    f, coefficients = _calibrate(study['panel_a'], float(study['telemetry_vec'][-1]))
    z = _invert(np.concatenate((train, test)), f, coefficients)
    panel = study['panel_c']
    u = np.asarray(panel['u'], dtype=float)
    y1, y2 = panel['v1'], panel['v2']

    def intercept(a):
        return np.column_stack((np.ones(len(a)), a))

    Z = _quadratic(z)
    U = _quadratic(u)
    support = np.linalg.norm(_ols(intercept(U), y1)[1:], axis=1)
    keep = support > max(.002, .003 * support.max())
    P = intercept(Z[:, keep]) @ _ols(intercept(U[:, keep]), y1)

    def design(freq):
        return intercept(np.cos(freq*u))

    def objective(freq):
        A = design(freq)
        return np.sum((A @ _ols(A, y2) - y2)**2)

    q = minimize_scalar(objective, bounds=(1.35, 3.55), method='bounded').x
    H = np.cos(q*z)
    C = intercept(H) @ _ols(design(q), y2)
    panel_b = study['panel_b']
    rotation = _ols(intercept(panel_b['u']), panel_b['v'])[1:]
    B = z @ rotation
    R = np.linalg.solve(rotation.T, z.T).T
    features = {'p': P, 'c': C, 'h': H, 'd': np.sin(q*z), 'm': Z[:, keep],
                'q': Z, 'r': _quadratic(R), 'b': _quadratic(B), 'i': z,
                'f': np.cos(q*R), 'g': np.cos(q*B), 's': P+C, 't': P-C,
                'e': z*z, 'k': P*C}
    blocks = [_normalize(features[key], n, 2) * (1 if i == 0 else 0.5)
              for i, key in enumerate('mht')]
    return {'features': 2 * np.column_stack(blocks), 'n_train': n}
