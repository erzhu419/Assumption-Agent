import numpy as np
from scipy.optimize import least_squares


def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def one(z):
    return np.c_[np.ones(len(z)), z]


def _design(u, j):
    a, b = u[:, j], u[:, (j + 1) % u.shape[1]]
    return np.column_stack((np.ones(len(u)), u, a * b, a**3))


def _calibration(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    design = [_design(u, j) for j in range(u.shape[1])]
    affine = one(u)
    r = np.linalg.norm(v - affine @ lin(affine, v), axis=1)
    keep = r <= max(3 * np.median(r), 1e-3)
    if keep.sum() < design[0].shape[1]:
        keep[:] = True
    for _ in range(8):
        coeff = np.column_stack([lin(f[keep], v[keep, j]) for j, f in enumerate(design)])
        fit = np.column_stack([f @ coeff[:, j] for j, f in enumerate(design)])
        r = np.linalg.norm(v - fit, axis=1)
        refined = r <= max(5 * np.median(r), 1e-3)
        if np.array_equal(keep, refined) or refined.sum() < design[0].shape[1]:
            break
        keep = refined
    return coeff


def _invert(raw, coeff):
    d = raw.shape[1]
    i = np.arange(d)
    j = (i + 1) % d
    matrix = coeff[1:d+1]
    offset, pair, cubic = coeff[0], coeff[-2], coeff[-1]
    linear_guess = np.linalg.lstsq(matrix.T, (raw - offset).T, rcond=1e-9)[0].T

    def jac(z):
        out = matrix.T.copy()
        out[i, i] += pair * z[j] + 3 * cubic * z**2
        out[i, j] += pair * z[i]
        return out

    result = np.empty_like(raw)
    for k, (x, initial) in enumerate(zip(raw, linear_guess)):
        def residual(z):
            return offset + z @ matrix + pair * z * z[j] + cubic * z**3 - x

        fit = least_squares(residual, initial, jac=jac, gtol=1e-11, ftol=1e-11, xtol=1e-11)
        if np.linalg.norm(fit.fun) > 1e-7:
            other = least_squares(residual, np.zeros(d), jac=jac, gtol=1e-11, ftol=1e-11, xtol=1e-11)
            if np.linalg.norm(other.fun) < np.linalg.norm(fit.fun):
                fit = other
        result[k] = fit.x
    return result


def _panel_maps(panel, z):
    u = np.asarray(panel['u'], dtype=float)

    # Both responses share their nonlinear shape parameters across coordinates.
    def softsign(w, q):
        return one(w / (1 + q[0]*abs(w)) + q[1]*w)

    def curvature(w, q):
        w2 = w*w
        return np.c_[one(w), w2, w2/(1+q[0]*w2)]

    def fit(features, target, initial, bounds, **options):
        def residual(q):
            f = features(u, q)
            return (f @ lin(f, target) - target).ravel()
        q = least_squares(residual, initial, bounds=bounds, **options).x
        return features(z, q) @ lin(features(u, q), target)

    c = fit(softsign, panel['v1'], [.85, -.01], ([.02, -1], [10, 1]))
    d = fit(curvature, panel['v2'], [.25], (.08, .65), gtol=1e-13)
    return np.c_[c, d]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.asarray(study['train']['x'], dtype=float)
    n = len(x)
    raw = np.vstack((x, study['test']['x']))
    z = _invert(raw, _calibration(study['panel_a']))
    w = _panel_maps(study['panel_c'], z)
    w = (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1)
    return {'features': w, 'n_train': n}
