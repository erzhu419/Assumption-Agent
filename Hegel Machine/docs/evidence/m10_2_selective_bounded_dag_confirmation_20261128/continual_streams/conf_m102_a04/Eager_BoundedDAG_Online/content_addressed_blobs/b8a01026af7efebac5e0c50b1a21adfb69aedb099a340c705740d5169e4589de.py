import numpy as np
from scipy.optimize import least_squares


def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def one(z):
    return np.c_[np.ones(len(z)), z]


def _calibrate(panel, measured):
    """Fit the panel's forward response, then invert it row by row."""
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    # Grossly corrupted rows have large joint residuals even to an affine fit.
    affine = one(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(4):
        beta = lin(affine[keep], v[keep])
        error = np.linalg.norm(affine @ beta - v, axis=1)
        med = np.median(error)
        spread = 1.4826 * np.median(np.abs(error - med))
        good = error <= med + 6 * max(spread, .01)
        if good.sum() < 2 * u.shape[1] + 2 or np.array_equal(good, keep):
            break
        keep = good
    u, v = u[keep], v[keep]

    def design(z, frequency):
        return np.c_[one(z), np.sin(frequency * z)]

    def residual(freq):
        f = design(u, freq[0])
        return (f @ lin(f, v) - v).ravel()

    frequency = least_squares(residual, [2.2], bounds=([1.2], [3.5]),
                              gtol=1e-12).x[0]
    intercept, *weights = lin(design(u, frequency), v)
    linear, harmonic = np.split(np.asarray(weights), 2, axis=0)
    z = np.linalg.solve(linear.T, (measured - intercept).T).T
    for _ in range(20):
        angle = frequency * z
        error = z @ linear + np.sin(angle) @ harmonic + intercept - measured
        if np.max(np.abs(error), initial=0.) < 1e-11:
            break
        jacobian = linear + (frequency * np.cos(angle))[:, :, None] * harmonic
        step = np.linalg.solve(jacobian.transpose(0, 2, 1), error[..., None])[..., 0]
        z -= step / np.maximum(1., np.max(np.abs(step), axis=1, keepdims=True) / .35)
    return z


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
    train_x = np.asarray(study['train']['x'], dtype=float)
    measured = np.vstack((train_x, study['test']['x']))
    z = _calibrate(study['panel_a'], measured)
    w = _panel_maps(study['panel_c'], z)
    n = len(train_x)
    features = (w - w[:n].mean(axis=0)) / np.maximum(w[:n].std(axis=0), .1)
    return {'features': features, 'n_train': np.asarray(n, dtype=np.intp)}
