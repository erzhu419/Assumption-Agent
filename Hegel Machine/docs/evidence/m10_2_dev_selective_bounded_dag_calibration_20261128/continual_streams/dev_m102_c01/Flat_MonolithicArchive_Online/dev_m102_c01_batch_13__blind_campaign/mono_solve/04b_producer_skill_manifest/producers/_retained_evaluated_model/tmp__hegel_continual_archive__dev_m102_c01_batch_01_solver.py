"""Batch prediction with a calibrated spatial kernel and shared temporal modes."""
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist


def _calibration(panel):
    u, v = map(np.asarray, (panel['u'], panel['v']))
    i, j = np.triu_indices(len(u), 1)
    du = u[i] - u[j]
    slopes = np.median((v[i] - v[j]) / np.where(abs(du) > 1e-12, du, np.inf), axis=0)
    offsets = np.median(v - u * slopes, axis=0)
    for _ in range(3):
        r = np.linalg.norm(v - u * slopes - offsets, axis=1)
        med = np.median(r)
        keep = r <= max(6 * med, 0.0005)
        if keep.sum() < 5:
            break
        for k in range(u.shape[1]):
            slopes[k], offsets[k] = np.linalg.lstsq(np.c_[u[keep, k], np.ones(keep.sum())], v[keep, k], rcond=None)[0]
    return np.where(abs(slopes) > 1e-6, slopes, 1.), offsets


def _kernel(x, q, slope):
    i, j = np.triu_indices(x.shape[1])
    return (1. + x @ q.T + .30 * ((x[:, i] * x[:, j]) @ (q[:, i] * q[:, j]).T)
            + 1.5 * np.exp(-cdist(x, q, 'sqeuclidean') / 6.)
            + 1.5 * np.exp(-cdist(x / slope, q / slope, 'sqeuclidean') / 6.))


def _features(z, kind, k):
    if kind == 'tanh':
        return np.c_[np.ones(len(z)), np.tanh(k*z)]
    return np.c_[np.ones(len(z)), z, z*z/(1. + k*z*z)]


def _operator(u, v, kind, bounds):
    def loss(k):
        f = _features(u, kind, k)
        U, _, _ = np.linalg.svd(f, full_matrices=False)
        e = v - U @ (U.T @ v)
        return np.mean((e / np.maximum(1. - (U*U).sum(axis=1, keepdims=True), 1e-8))**2)
    k = minimize_scalar(loss, bounds=bounds, method='bounded').x
    w = np.linalg.lstsq(_features(u, kind, k), v, rcond=None)[0]
    return lambda z: _features(z, kind, k) @ w


def _quadratic(z):
    i, j = np.triu_indices(z.shape[1])
    return np.c_[z, z[:, i]*z[:, j]]


def _predict(K, Kq, target, ridge):
    mu = target.mean(axis=0)
    return Kq @ solve(K + ridge*np.eye(len(K)), target - mu, assume_a='pos') + mu


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], dtype=float)
    q = np.asarray(study['test']['x'], dtype=float)
    y, t = map(np.asarray, (train['y'], train['traj']))
    slope, offset = _calibration(study['panel_a'])
    b, c = study['panel_b'], study['panel_c']
    u = b['u']
    w = np.linalg.lstsq(np.c_[np.ones(len(u)), u], b['v'], rcond=None)[0]
    bmap = lambda z: z @ w[1:] + w[0]
    c1 = _operator(c['u'], c['v1'], 'tanh', (.3, 3.))
    c2 = _operator(c['u'], c['v2'], 'rational', (.015, .8))
    sy, qy = (c1(bmap((z-offset)/slope)) for z in (x, q))
    st, qt = (_quadratic(c2(bmap(z*slope+offset))) for z in (x, q))
    K, Kq = _kernel(x, x, slope), _kernel(q, x, slope)
    yh = _predict(K + .3*(sy @ sy.T), Kq + .3*(qy @ sy.T), y, .001)
    _, _, modes = np.linalg.svd(t, full_matrices=False)
    modes = modes[:2]
    traj = _predict(K + st @ st.T, Kq + qt @ st.T, t @ modes.T, .001) @ modes
    return {'y_hat': yh, 'state_pred': traj,
            'diagnostics_vec': np.array([yh.mean(), yh.std(), traj[:, -1].mean(), traj[:, -1].std()])}

from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _gp(parts, n, y):
    p = np.asarray(parts)
    a = p[:, :n, :n].copy()
    b = p[:, n:, :n].copy()
    E = np.eye(n)
    k = len(a)
    def objective(h):
        w = np.exp(h)
        cov = np.einsum('i,ijk->jk', w[:-1], a) + w[-1]*E
        c = cho_factor(cov)
        alpha = cho_solve(c, y)
        G = (cho_solve(c, E) - np.outer(alpha, alpha))*0.5
        return y@alpha/2 + np.log(np.diag(c[0])).sum(), w*np.r_[np.einsum('ij,kij->k', G, a), np.trace(G)]
    h = minimize(objective, np.r_[np.full(k, -3.), -8.], jac=True, method='L-BFGS-B', bounds=[(-15,5)]*k+[(-11,-4)], options={'maxiter':155}).x
    w = np.exp(h)
    cov = np.einsum('i,ijk->jk', w[:-1], a) + w[-1]*E
    return np.einsum('i,ijk->jk', w[:-1], b) @ cho_solve(cho_factor(cov), y)


def _warps(study):
    train = study['train']
    n = len(train['x'])
    x = np.vstack([train['x'], study['test']['x']])
    slope, off = _calibration(study['panel_a'])
    z = (x-off)/slope
    panel = study['panel_b']
    u = panel['u']
    B = np.linalg.lstsq(np.c_[np.ones(len(u)),u],panel['v'],rcond=None)[0]
    coords = (z, B[0]+z@B[1:], (z-B[0])@np.linalg.pinv(B[1:]))
    d = cdist(z,z,'sqeuclidean')
    i,j = np.triu_indices(z.shape[1]); gram = lambda f:f@f.T
    base = [np.ones_like(d),gram(z),gram(z[:,i]*z[:,j])]+[np.exp(-d/h) for h in (1,2,4,8,16,32)]
    def parts(idx,h,kind):
        v = coords[idx].copy()
        v /= np.maximum(1e-12, np.sqrt(np.mean(v[:n]**2,axis=0)))
        if kind == 'rat': w=v/(1+h*v*v)
        elif kind == 'root': w=v/np.sqrt(1+h*v*v)
        elif kind == 'soft': w=v/(1+h*abs(v))
        else: w=np.tanh(h*v)
        w /= np.maximum(1e-12, np.sqrt(np.mean(w[:n]**2,axis=0)))
        return base+[gram(w),gram(w[:,i]*w[:,j]),gram((v[:,:,None]*w[:,None,:]).reshape(len(w),-1))]
    return n, base, parts


_archive_solve = solve_batch


def _components(study):
    train = study['train']
    y, t = map(np.asarray, (train['y'], train['traj']))
    n, base, parts = _warps(study)
    specs = [(0,.4,'rat'), (0,.8,'rat'), (0,1.5,'root'), (0,.8,'soft'), (2,1.5,'soft'), (2,2.5,'root'), (1,.4,'tanh')]
    yp = np.array([_gp(parts(*s),n,y) for s in specs])
    mode = np.linalg.svd(t,full_matrices=False)[2][0]
    tp = _gp(base,n,t@mode)[:,None]*mode
    return yp, tp, _archive_solve(study)


def solve_batch(study: dict) -> dict:
    yp, tp, archive = _components(study)
    yh = .6*yp[:4].mean(axis=0) + .4*yp[4:].mean(axis=0)
    traj = .35*tp + .65*archive['state_pred']
    return {'y_hat':yh, 'state_pred':traj,
            'diagnostics_vec':np.array([yh.mean(),yh.std(),traj[:,-1].mean(),traj[:,-1].std()])}
