"""Panel-conditioned regression for scalar and trajectory readouts."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _panel(panel, trim=0):
    u = _affine(panel['u'])
    v = panel['v']
    keep = np.arange(len(u))
    for _ in range(5 if trim else 0):
        c = np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]
        keep = np.argsort(np.linalg.norm(u @ c - v, axis=1))[:len(u)-trim]
    return np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]


def _universal(k, kt, z, zt, y, noise):
    p, pt = _affine(z), _affine(zt)
    f = cho_factor(k + noise*np.eye(len(z)))
    ip, iy = cho_solve(f, p), cho_solve(f, y)
    c = np.linalg.solve(p.T @ ip, p.T @ iy)
    return pt @ c + kt @ (iy-ip @ c)


def _scalar_kernel(x, xt, z, zt, y):
    def parts(a, b, u, v):
        distance = cdist(u, v, 'sqeuclidean')
        raw = .4*np.exp(-.23*cdist(a, b, 'sqeuclidean'))
        r = np.sqrt(.30*distance)
        return (
            raw + np.exp(-.13*distance) + .05*(1+u @ v.T/4)**2,
            raw + (1+r+r*r/3)*np.exp(-r),
            raw + 1/(1+.025*distance),
        )

    train = parts(x, x, z, z)
    test = parts(xt, x, zt, z)
    return sum(weight*_universal(k, kt, z, zt, y, .003)
               for weight, k, kt in zip((.4, .3, .3), train, test))


def _matern(z, zt, target, poly):
    p = _affine(z)
    n, dof = len(z), len(z)-p.shape[1]
    eye = np.eye(n)
    d = (z[:, None]-z[None, :])**2
    q = poly*(1+z @ z.T/4)**2
    ref = np.log(.13)

    def obj(h):
        weights, noise = np.exp(h[:4]), np.exp(h[4])
        r = np.sqrt(5*(d @ weights))
        em = np.exp(-r)
        k = (1+r+r*r/3)*em + q
        f = cho_factor(k+noise*eye)
        ip = cho_solve(f, p)
        ff = cho_factor(p.T @ ip)
        v = cho_solve(f, target)
        v -= ip @ cho_solve(ff, p.T @ v)
        rss = target @ v
        penalty = h[:4]-ref
        loss = (dof*np.log(rss/dof) + 2*np.log(f[0].diagonal()).sum()
                + 2*np.log(ff[0].diagonal()).sum() + .35*(penalty @ penalty))
        g = cho_solve(f, eye)-ip @ cho_solve(ff, ip.T)-dof/rss*np.outer(v, v)
        slope = -5/6*(1+r)*em
        grad = np.einsum('ij,ijk,ij,k->k', g, d, slope, weights)
        return loss, np.r_[grad+.7*penalty, noise*np.trace(g)]

    h = minimize(obj, np.r_[np.full(4, ref), np.log(.003)], jac=True,
                 method='L-BFGS-B', bounds=[(-5, 1.5)]*4+[(-12, -2)]).x
    weights, noise = np.exp(h[:4]), np.exp(h[4])

    def kernel(a, b):
        r = np.sqrt(5*cdist(a*np.sqrt(weights), b*np.sqrt(weights), 'sqeuclidean'))
        return (1+r+r*r/3)*np.exp(-r) + poly*(1+a @ b.T/4)**2

    return _universal(kernel(z,z), kernel(zt,z), z, zt, target, noise)


def _components(study):
    train = study['train']
    x, xt = train['x'], study['test']['x']
    a, b = _panel(study['panel_a'], 3), _panel(study['panel_b'])

    def transform(u):
        return _affine(_affine(u) @ a) @ b

    w, wt = transform(x), transform(xt)
    z, zt = (w-w.mean(0))/w.std(0), (wt-w.mean(0))/w.std(0)
    scalar = _scalar_kernel(x, xt, z, zt, train['y'])
    traj = train['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    amps = traj @ modes.T
    trajectory = np.array([np.column_stack([_matern(z, zt, col, poly) for col in amps.T]) @ modes
                           for poly in (0., .05)])
    return scalar, trajectory


def solve_batch(study: dict) -> dict:
    y_hat, trajectories = _components(study)
    state_pred = trajectories.mean(axis=0)
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
