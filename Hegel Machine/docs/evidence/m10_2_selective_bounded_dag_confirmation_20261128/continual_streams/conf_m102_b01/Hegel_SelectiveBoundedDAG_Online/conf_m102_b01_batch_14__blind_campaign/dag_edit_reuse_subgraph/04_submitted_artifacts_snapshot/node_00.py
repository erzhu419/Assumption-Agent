import numpy as np
from scipy.optimize import minimize_scalar


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _profile(loss, lo, hi):
    grid = np.linspace(lo, hi, 49)
    i = int(np.argmin([loss(g) for g in grid]))
    return minimize_scalar(loss, method='bounded',
                           bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]),
                           options={'xatol': 1e-9}).x


def _invert_a(panel, x):
    u, v = panel['u'], panel['v']
    d = _aff(u)
    fit = _ls(d, v)
    for _ in range(5):
        err = np.linalg.norm(d @ fit - v, axis=1)
        keep = np.argsort(err)[:max(6, int(.875*len(u)))]
        fit = _ls(d[keep], v[keep])
    err = np.linalg.norm(d @ fit - v, axis=1)
    keep = err <= max(4*np.median(err), .02)
    u, v = u[keep], v[keep]
    n = u.shape[1]
    j = np.arange(n)
    k = (j+2) % n

    def basis(z, g):
        sn = np.sin(g*z)
        return np.stack((np.ones_like(z), z, z[:, k], sn, sn[:, k], z**3), axis=-1)

    def fit_at(g, coeff=False):
        a = basis(u, g)
        q = np.array([_ls(a[:, i], v[:, i]) for i in j])
        return q if coeff else np.mean((np.sum(a*q, axis=-1)-v)**2)

    g = _profile(fit_at, .8, 4.5)
    q = fit_at(g, True)
    z = (x-q[:, 0])/q[:, 1]
    jac = np.zeros((len(x), n, n))
    for _ in range(30):
        r = np.sum(basis(z, g)*q, axis=-1)-x
        err = np.sum(r*r, axis=1)
        if np.max(err, initial=0.) < 1e-24:
            break
        cs = g*np.cos(g*z)
        jac[:, j, j] = q[:, 1]+q[:, 3]*cs+3*q[:, 5]*z*z
        jac[:, j, k] = q[:, 2]+q[:, 4]*cs[:, k]
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        step /= np.maximum(1, np.max(abs(step), axis=1, keepdims=True)/.5)
        pending = err > 1e-24
        for _ in range(15):
            trial = z-step
            rr = np.sum(basis(trial, g)*q, axis=-1)-x
            take = pending & (np.sum(rr*rr, axis=1) < err)
            z[take] = trial[take]
            pending &= ~take
            if not np.any(pending):
                break
            step *= .5
    return z


def _map_b(panel, z):
    u, v = panel['u'], panel['v']

    def basis(x, g):
        return _aff(np.cos(g*x))

    def loss(g):
        d = basis(u, g)
        return np.mean((d @ _ls(d, v)-v)**2)

    g = _profile(loss, .7, 4.5)
    return basis(z, g) @ _ls(basis(u, g), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    z = _invert_a(study['panel_a'], x)
    w = _map_b(study['panel_b'], z)
    # A shared monotone coordinate stretch conditions the frozen head's
    # saturating features without changing panel-B symmetry or mixing channels.
    return {'w': w + 0.27 * w**3}
