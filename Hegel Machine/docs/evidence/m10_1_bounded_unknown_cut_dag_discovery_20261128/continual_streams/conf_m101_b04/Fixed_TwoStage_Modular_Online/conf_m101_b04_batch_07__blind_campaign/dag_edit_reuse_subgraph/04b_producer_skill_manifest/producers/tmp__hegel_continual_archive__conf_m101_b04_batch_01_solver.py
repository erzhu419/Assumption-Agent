"""Calibrated kernel readouts for scalar response and time trajectory."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel, robust=False):
    u = _linear(panel['u'])
    v = panel['v']
    keep = np.arange(len(u))
    if robust:
        for _ in range(8):
            c = np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]
            err = np.linalg.norm(v - u @ c, axis=1)
            keep = np.argsort(err)[:len(u) - len(u)//5]
        keep = np.flatnonzero(err < 10*np.median(err))
    return np.linalg.lstsq(u[keep], v[keep], rcond=None)[0]


def _kernel_predict(k, kt, target, noise):
    f = cho_factor(k + noise*np.eye(len(k)))
    return kt @ cho_solve(f, target)


def _matern(z, zt, y):
    delta = (z[:, None] - z[None, :])**2
    dt = (zt[:, None] - z[None, :])**2
    p, pt = _linear(z), _linear(zt)
    eye = np.eye(len(z))

    def kernel(d, theta):
        r = np.sqrt(5*(d @ np.exp(theta[:4])))
        return (1+r+r*r/3)*np.exp(-r)

    def objective(theta, predict=False):
        f = cho_factor(kernel(delta, theta) + np.exp(theta[4])*eye)
        ip, iy = cho_solve(f, p), cho_solve(f, y)
        g = p.T @ ip
        beta = np.linalg.solve(g, p.T @ iy)
        alpha = iy - ip @ beta
        if predict:
            return pt @ beta + kernel(dt, theta) @ alpha
        return ((len(y)-p.shape[1])*np.log((y-p@beta) @ alpha)
                + 2*np.log(np.diag(f[0])).sum() + np.linalg.slogdet(g)[1]
                + 1.5*np.sum((theta[:4]+1.5)**2))

    optimum = minimize(objective, [-1.5]*4+[-6.], method='L-BFGS-B',
                       bounds=[(-5, 2)]*4+[(-12, -2)])
    return objective(optimum.x, predict=True)


def _trajectory(x, xt, z, zt, train):
    scale = np.sqrt(np.mean(x*x, axis=0))
    xs, xst = x/scale, xt/scale
    def kernel(a, b, c, d):
        return np.exp(-.2*cdist(c, d, 'sqeuclidean')) + (1+a@b.T/4)**2
    am = _kernel_predict(kernel(xs, xs, z, z), kernel(xst, xs, zt, z), train, .002)
    # The public trajectories are low rank; discard independent higher time modes.
    mean = train.mean(axis=0)
    modes = np.linalg.svd(train-mean, full_matrices=False)[2][:2]
    return mean + ((am-mean) @ modes.T) @ modes


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = train['x'], study['test']['x']
    a = _calibration(study['panel_a'], robust=True)
    b = _calibration(study['panel_b'])
    xa, xat = _linear(x) @ a, _linear(xt) @ a
    xab, xabt = _linear(xa) @ b, _linear(xat) @ b
    def rbf(l, r):
        return np.exp(-.2*cdist(l, r, 'sqeuclidean'))
    y = train['y']
    y_hat = .55*_matern(xa, xat, y) + .45*_kernel_predict(rbf(xab, xab), rbf(xabt, xab), y, .002)
    ai, bi = np.linalg.inv(a[1:]), np.linalg.inv(b[1:])
    def inverse(z):
        return ((z-b[0]) @ bi-a[0]) @ ai
    state = _trajectory(x, xt, inverse(x), inverse(xt), train['traj'])
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
