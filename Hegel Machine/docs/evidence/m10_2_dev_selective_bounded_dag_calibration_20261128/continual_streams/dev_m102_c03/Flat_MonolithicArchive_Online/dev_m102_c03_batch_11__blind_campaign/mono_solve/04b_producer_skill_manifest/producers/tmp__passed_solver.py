"""Panel-calibrated two-mode predictor."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _soft(x):
    return x / (1 + np.abs(x))


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    a = _aff(u)
    err = np.linalg.norm(v - a @ _ls(a, v), axis=1)
    keep = err < max(3*np.median(err), 0.2)
    if keep.sum() < 20:
        keep = err <= np.quantile(err, .85)
    def design(z, rate):
        w = np.roll(z, 2, axis=-1)
        return np.stack((np.ones_like(z), z, w, z*w, z*z, w*w, z**3,
                         np.sin(rate*w)), axis=-1)
    def fit(rate):
        a = design(u[keep], rate)
        return np.array([_ls(a[:, j], v[keep, j]) for j in range(4)])
    def forward(z, rate, coef):
        return np.einsum('...ij,ij->...i', design(z, rate), coef)
    for i in range(3):
        def objective(rate):
            return np.mean((forward(u[keep], rate, fit(rate))-v[keep])**2)
        rate = minimize_scalar(objective, bounds=(1.2, 3.5), method='bounded').x
        coef = fit(rate)
        err = np.linalg.norm(v-forward(u, rate, coef), axis=1)
        new = err < max(.04, 4*np.median(err[keep]))
        if new.sum() < 20 or np.array_equal(new, keep) or i == 2:
            break
        keep = new
    affine = _ls(a[keep], v[keep]); init = _ls(affine[1:].T, (x-affine[0]).T).T
    return np.array([least_squares(lambda p: forward(p, rate, coef)-row,
                                   z0, gtol=1e-11).x for row, z0 in zip(x, init)])


def _branch(panel, z):
    u, v = panel['u'], panel['v1']
    def design(x, rate):
        return np.column_stack((_aff(x), _soft(rate*x)))
    def residual(rate):
        a = design(u, rate[0]); return (a @ _ls(a, v)-v).ravel()
    rate = least_squares(residual, [.83], bounds=(.2, 2.)).x[0]
    return design(z, rate) @ _ls(design(u, rate), v)


def _cross(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def _basis(times, traj):
    def design(r):
        return -np.expm1(-times[:, None]*r)
    def residual(r):
        a = design(r); return (a @ _ls(a, traj.T)-traj.T).ravel()
    r = least_squares(residual, [.4, 1.8], bounds=([.12, 1.], [.9, 4.])).x
    return np.linalg.qr(design(r), mode='reduced')[0]


def _readout(features, target, n):
    f = (features-features[:n].mean(0))/np.maximum(features[:n].std(0), 1e-10)
    k = 1 + f @ f[:n].T / f.shape[1]
    eig, vec = np.linalg.eigh(k[:n]); eig = np.maximum(eig, 0.)
    best = np.full(2, np.inf)
    out = np.empty((len(f)-n, target.shape[1]))
    for alpha in [1e-9, 1e-7, 1e-5, 1e-4, .001, .005, .02, .1]:
        div = 1/(eig+alpha)
        weights = vec @ (div[:, None]*(vec.T @ target))
        err = np.mean((weights / ((vec**2) @ div)[:, None])**2, axis=0)
        for j, sel in enumerate((slice(0, 1), slice(1, None))):
            if err[sel].sum() < best[j]:
                best[j] = err[sel].sum()
                out[:, sel] = k[n:] @ weights[:, sel]
    return out


def solve_batch(study: dict) -> dict:
    train = study['train']; n = len(train['x'])
    z = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    a = _branch(study['panel_c'], z)
    b = np.exp(-.877**2*z*z)
    features = np.column_stack((a, b, _cross(a, b), _cross(z, a), _cross(z, b), z))
    Q = _basis(study['eval_times'], train['traj'])
    targets = np.column_stack((train['y'], train['traj'] @ Q))
    pred = _readout(features, targets, n)
    y, state = pred[:, 0], pred[:, 1:] @ Q.T
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()]))
