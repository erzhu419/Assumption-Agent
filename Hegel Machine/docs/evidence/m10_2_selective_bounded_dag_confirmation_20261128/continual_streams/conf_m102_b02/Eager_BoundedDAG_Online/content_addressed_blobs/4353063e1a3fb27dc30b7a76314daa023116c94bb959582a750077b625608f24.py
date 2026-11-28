"""Recover latent coordinates from calibration panels for the frozen kernel."""
import numpy as np
from scipy.optimize import minimize_scalar


def _design(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(a, v):
    return np.linalg.lstsq(a, v, rcond=1e-10)[0]


def _latent(panel, x):
    a, v = _design(panel['u']), np.asarray(panel['v'])
    keep = np.ones(len(a), dtype=bool)
    for _ in range(8):
        w = _fit(a[keep], v[keep])
        e = np.linalg.norm(a @ w - v, axis=1)
        good = e < max(.001, 3 * np.median(e))
        if np.array_equal(keep, good) or np.count_nonzero(good) < a.shape[1]:
            break
        keep = good
    return np.linalg.solve(w[1:].T, (x - w[0]).T).T


def _panel_b(panel, z):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])

    def loss(q):
        f = _design(np.cos(q * u))
        return np.mean((f @ _fit(f, v) - v)**2)

    grid = np.linspace(.12, 7., 90)
    i = np.argmin([loss(q) for q in grid])
    q = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),
                        method='bounded', options={'xatol': 1e-8}).x
    c = np.cos(q * z)
    return _design(c) @ _fit(_design(np.cos(q * u)), v), c


from itertools import product
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import erf


def _groups(z, b, c):
    blocks = []
    for x in (z, b, c):
        blocks.append(x)
        for q in (1., 1.5, 2., 2.5, 3., 4.):
            blocks.extend((np.tanh(q*x), np.sin(q*x), erf(q*x)))
        blocks.extend(x**p for p in (2, 3, 4, 5, 6))
        blocks.append(np.column_stack((np.sin(2*x), np.sin(4*x),
                                       np.cos(2*x), np.cos(4*x))))
    axes = {'z': z, 'b': b, 'c': c, 't': np.tanh(2*b), 's': np.sin(2.6*z)}
    for a, x in axes.items():
        for d, v in axes.items():
            if a <= d:
                blocks.append(np.column_stack([x[:, i]*v[:, j]
                              for i, j in product(range(x.shape[1]), range(v.shape[1]))
                              if a != d or i < j]))
        blocks.append(x)
        blocks.extend(x[:, j:j+1] for j in range(x.shape[1]))
    return [f for f in blocks if f.shape[1]]


def _weights(blocks, n, targets):
    """Fit nonnegative group scales by regularized training-only leave-one-out."""
    a = np.asarray([f[:n] @ f[:n].T for f in blocks]).reshape(len(blocks), -1)
    eye = np.eye(n)
    y = targets / np.maximum(targets.std(axis=0), 1e-8)
    if y.shape[1] > 1:
        # Give the whole trajectory approximately the same weight as the scalar.
        y[:, 1:] *= targets[:, 1:].std(axis=0) / max(targets[:, 1:].std(), 1e-8)
        y[:, 1:] /= np.sqrt(y.shape[1]-1)
    penalty = .015

    def objective(u):
        w = np.exp(u)
        gram = (w @ a).reshape(n, n) + 1. + .004*eye
        inverse = solve(gram, eye, assume_a='pos', check_finite=False)
        alpha = inverse @ y
        diagonal = inverse.diagonal()
        residual = alpha / diagonal[:, None]
        da = 2*residual / diagonal[:, None]
        dq = -2*np.sum(residual**2, axis=1) / diagonal
        dg = -(inverse @ da) @ alpha.T - (inverse*dq) @ inverse
        return np.sum(residual**2) + penalty*np.sum(w), w*(a @ dg.ravel() + penalty)

    result = minimize(objective, np.full(len(blocks), -4.), jac=True,
                      bounds=[(-12., 4.)]*len(blocks), method='L-BFGS-B',
                      options={'maxiter': 170})
    return np.exp(result.x)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    tr, te = np.asarray(train['x']), np.asarray(study['test']['x'])
    n = len(tr)
    z = _latent(study['panel_a'], np.vstack((tr, te)))
    b, c = _panel_b(study['panel_b'], z)
    blocks = [(f-f[:n].mean(axis=0)) / np.maximum(f[:n].std(axis=0), 1e-7)
              / np.sqrt(f.shape[1]) for f in _groups(z, b, c)]
    targets = np.column_stack((train['y'], train['traj']))
    weights = _weights(blocks, n, targets)
    # An offset makes the frozen quadratic kernel's linear term dominant:
    # (1 + offset + g/(2*(1+offset)))**2 = intercept + g + small remainder.
    dim = 1 + sum(f.shape[1] for f in blocks)
    offset = 16.
    features = np.column_stack([np.full(len(z), np.sqrt(dim*offset))] +
                 [f*np.sqrt(dim*w/(2*(1+offset))) for f, w in zip(blocks, weights)])
    return {'train_features': features[:n], 'test_features': features[n:]}
