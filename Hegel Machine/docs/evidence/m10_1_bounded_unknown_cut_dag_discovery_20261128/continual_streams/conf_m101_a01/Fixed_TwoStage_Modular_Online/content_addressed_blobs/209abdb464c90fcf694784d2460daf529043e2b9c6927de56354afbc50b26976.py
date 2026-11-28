import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _profile(tr, times):
    def curve(rate):
        return -np.expm1(-rate * times)

    def error(rate):
        f = curve(rate)
        return np.mean((tr - np.outer(tr @ f / (f @ f), f)) ** 2)

    rate = minimize_scalar(error, bounds=(0.1, 3.0), method='bounded').x
    f = curve(rate)
    return tr @ f / (f @ f), f


def _gp(x, xq, v, vq, target, kind):
    d, dq = _aff(x), _aff(xq)
    tdist = (v[:, None, :] - v[None, :, :]) ** 2
    qdist = (vq[:, None, :] - v[None, :, :]) ** 2
    n, width = v.shape
    p = d.shape[1]
    eye = np.eye(n)
    rhs = np.column_stack((d, target))

    def cov(dist, h):
        q = dist @ np.exp(h[:-1])
        if kind == 'gaussian':
            return np.exp(-q / 2)
        if kind == 'rational':
            return 1 / (1 + q / 2)
        r = np.sqrt(q)
        return (1 + r + q / 3) * np.exp(-r)

    def fitted(h):
        c = cho_factor(cov(tdist, h) + np.exp(h[-1]) * eye)
        f = cho_solve(c, rhs)
        a = d.T @ f[:, :p]
        beta = np.linalg.solve(a, d.T @ f[:, p:])
        alpha = f[:, p:] - f[:, :p] @ beta
        return c, a, beta, alpha

    def loss(h):
        c, a, beta, alpha = fitted(h)
        r = target - d @ beta
        return ((n - p) * np.linalg.slogdet(r.T @ alpha)[1]
                + target.shape[1] * (2 * np.log(np.diag(c[0])).sum() + np.linalg.slogdet(a)[1])
                + width * np.var(h[:-1]))

    h = minimize(loss, np.r_[np.full(width, -.2), -7.5], method='L-BFGS-B',
                 bounds=[(-5, 3)] * width + [(-14, -2)], options={'maxiter': 85}).x
    _, _, beta, alpha = fitted(h)
    return dq @ beta + cov(qdist, h) @ alpha


def _joint_prediction(study):
    train = study['train']
    x, xq = train['x'], study['test']['x']
    z, f = _profile(train['traj'], study['eval_times'])
    target = np.column_stack((train['y'], z))
    panel = study['panel_b']
    op = np.linalg.lstsq(_aff(panel['u']), panel['v'], rcond=None)[0]
    inv = np.linalg.inv(op[1:])
    v, vq = x @ inv, xq @ inv
    pm = _gp(x, xq, v, vq, target, 'matern')
    pr = _gp(x, xq, v, vq, target, 'rational')
    pg = _gp(x, xq, v, vq, target, 'gaussian')
    y, zq = (0.5 * pm + 0.25 * pr + 0.25 * pg).T
    state = np.outer(zq, f)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}


def _coordinates(study, x):
    a, b = study['panel_a'], study['panel_b']
    keep = np.linalg.norm(a['u']-a['v'],axis=1)<1
    aa = np.linalg.lstsq(_aff(a['u'][keep]),a['v'][keep],rcond=None)[0]
    bb = np.linalg.lstsq(_aff(b['u']),b['v'],rcond=None)[0]
    u = (x-aa[0]) @ np.linalg.inv(aa[1:])
    v = (x-bb[0]) @ np.linalg.inv(bb[1:])
    return (x, u, v, _aff(u) @ bb, _aff(x) @ bb)


def _geometry(v, warp, mode):
    if warp == 'sin':
        v = np.sin(.9*v)
    elif warp == 'tanh':
        v = np.tanh(.7*v)
    elif warp == 'soft':
        v = v/np.sqrt(1+.6*v*v)
    d = (v[:,None,:]-v[None,:,:])**2
    if mode == 'pair':
        return np.stack([d[:,:,i]+d[:,:,j] for i in range(4) for j in range(i+1,4)],axis=2)
    if mode == 'add':
        return d
    return d.sum(axis=2)


def _kernel(d, kind, ell):
    q = d / ell**2
    if kind == 'sq':
        k = np.exp(-q/2)
    elif kind == 'rq':
        k = 1/(1+q/2)
    else:
        r = np.sqrt(q)
        k = (1+r+q/3)*np.exp(-r)
    return k if k.ndim==2 else k.mean(axis=2)


def _readout(estimate, link, cap):
    if link == 'sqrt':
        return cap-estimate**2
    if link == 'log':
        return cap-np.exp(estimate)
    return estimate


def _fit_kernel(K, trend, y, i, j, link, cap):
    d = trend[i]
    if link == 'sqrt':
        y = np.sqrt(cap-y)
    elif link == 'log':
        y = np.log(cap-y)
    f = cho_solve(cho_factor(K[np.ix_(i,i)]),np.column_stack((d,y)))
    u, v = f[:,:-1], f[:,-1]
    beta = np.linalg.solve(d.T@u,d.T@v)
    return _readout(trend[j]@beta+K[np.ix_(j,i)]@(v-u@beta),link,cap)


_BASE_WEIGHT = 0.405
_READOUT_MODELS = [((4, 'sin', 'full', 'rq', 2.5, 3e-06, 1, 'id', 0.0), 0.405), ((3, 'id', 'full', 'sq', 1.6, 3e-06, 1, 'log', 2.3), 0.09000000000000001), ((2, 'sin', 'full', 'sq', 4.0, 0.0003, 1, 'id', 0.0), 0.1)]


def solve_batch(study):
    result = _joint_prediction(study)
    x, xq = study['train']['x'], study['test']['x']
    xx = np.vstack((x,xq))
    coordinates = _coordinates(study,xx)
    d = _aff(xx)
    q = np.column_stack([d]+[xx[:,i]*xx[:,j] for i in range(4) for j in range(i,4)])
    i, j = np.arange(len(x)), np.arange(len(x),len(xx))
    y = _BASE_WEIGHT * result['y_hat']
    for (idx,warp,mode,kind,ell,lam,order,link,cap), weight in _READOUT_MODELS:
        k = _kernel(_geometry(coordinates[idx],warp,mode),kind,ell)
        k.flat[::len(xx)+1] += lam
        y += weight*_fit_kernel(k,d if order==1 else q,study['train']['y'],i,j,link,cap)
    last = result['state_pred'][:,-1]
    result['y_hat'] = y
    result['diagnostics_vec'] = np.array([y.mean(),y.std(),last.mean(),last.std()])
    return result
