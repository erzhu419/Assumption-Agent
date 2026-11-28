import numpy as np
from scipy.optimize import least_squares, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _next(x):
    return np.roll(x, -1, axis=-1)


def _calibration(panel):
    u, v = panel['u'], panel['v']
    n = len(u)
    following = _next(u)
    design = np.stack((np.ones_like(u), u, following, u*following, u**3), axis=-1)
    mask = np.ones(n, dtype=bool)
    for _ in range(12):
        w = np.column_stack([_ls(design[mask, j], v[mask, j]) for j in range(u.shape[1])])
        err = np.linalg.norm(np.einsum('njk,kj->nj', design, w)-v, axis=1)
        keep = err <= max(1e-8, 4*np.median(err[mask]))
        if keep.sum() < 8 or np.array_equal(mask, keep):
            break
        mask = keep
    return w


def _invert(readings, w):
    bias, gain, coupling, mixed, cubic = w
    n = len(bias)
    i = np.arange(n)
    j = (i+1) % n

    def forward(x):
        neighbor = x[j]
        return bias + gain*x + (coupling+mixed*x)*neighbor + cubic*x**3

    def jacobian(x):
        jac = np.diag(gain + mixed*x[j] + 3*cubic*x*x)
        jac[i, j] = coupling + mixed*x
        return jac

    latent = np.empty_like(readings, dtype=float)
    for k, row in enumerate(readings):
        initial = np.clip((row-bias)/gain, -1.5, 1.5)
        residual = lambda x: forward(x)-row
        fit = root(residual, initial, jac=jacobian, tol=1e-10)
        if np.linalg.norm(fit.fun, np.inf) > 1e-8:
            fit = least_squares(residual, fit.x, jac=jacobian, gtol=1e-12)
        latent[k] = fit.x
    return latent


def _panel_features(panel, latent):
    u = panel['u']
    w_product = _ls(_affine(u*_next(u)), panel['v1'])
    observed_cosine = panel['v2']

    def residual(freq):
        a = _affine(np.cos(freq[0]*u))
        return (a @ _ls(a, observed_cosine) - observed_cosine).ravel()

    # Profile out the linear mixing, then refine the shared frequency.
    grid = np.linspace(0.25, 5.0, 39)
    start = min(grid, key=lambda f: np.linalg.norm(residual((f,))))
    freq = least_squares(residual, [start], bounds=(0.05, 8.0)).x[0]
    w_cosine = _ls(_affine(np.cos(freq*u)), observed_cosine)
    return _affine(latent*_next(latent)) @ w_product, _affine(np.cos(freq*latent)) @ w_cosine


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train, test = study['train'], study['test']
    n = len(train['x'])
    latent = _invert(np.concatenate((train['x'], test['x'])), _calibration(study['panel_a']))
    p, r = _panel_features(study['panel_c'], latent)
    common = _affine(np.column_stack((p, r, np.einsum('ij,ij->i', p, r))))
    off_i, off_j = np.nonzero(~np.eye(p.shape[1], dtype=bool))
    features = np.column_stack((common, p[:, off_i]*r[:, off_j]))
    return {
        'train_features': features[:n].copy(),
        'test_features': features[n:].copy(),
        'regression': common[n:] @ _ls(common[:n], train['y']),
    }
