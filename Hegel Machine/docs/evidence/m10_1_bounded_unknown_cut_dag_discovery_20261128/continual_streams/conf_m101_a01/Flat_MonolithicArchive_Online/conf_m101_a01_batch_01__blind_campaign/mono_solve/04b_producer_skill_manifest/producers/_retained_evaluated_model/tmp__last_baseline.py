import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _trend(x, degree):
    terms = [np.ones(len(x)), x]
    if degree == 2:
        terms.extend(x[:, i] * x[:, j] for i in range(4) for j in range(i, 4))
    return np.column_stack(terms)


def _smooth(x, z, y, degree, family):
    n = len(x)
    scale = np.maximum(x.std(0), 1e-8)
    x, z = x / scale, z / scale
    F, G = _trend(x, degree), _trend(z, degree)
    U = np.linalg.qr(F, mode='complete')[0][:, F.shape[1]:]
    r = U.T @ y
    I = np.eye(len(r))
    diagonal = np.tril_indices(4, -1)
    pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
    D = (x[:,None,:] - x[None,:,:])**2
    Dz = (z[:,None,:] - x[None,:,:])**2
    if family == 'rot':
        start = np.r_[np.full(4, .55), np.zeros(6), -8.]
        bounds = [(-1.1, 2.3)]*4 + [(-1.6,1.6)]*6 + [(-15,-2)]
        def kernel(a,b,d,p):
            T = np.diag(np.exp(-p[:4])); T[diagonal] = p[4:10]
            return np.exp(-.5*cdist(a@T,b@T,'sqeuclidean'))
        def penalty(p):
            return .3*np.sum((p[:4] - np.mean(p[:4]))**2) + 1.3*np.sum(p[4:10]**2)
    elif family in ('axes', 'mixed'):
        start = np.r_[.5, .4, -1., np.zeros(4), -8.]
        bounds = [(-1.2,2.2),(-1.2,2.2),(-5,5)] + [(-1.3,1.3)]*4 + [(-15,-2)]
        def kernel(a,b,d,p):
            full = np.exp(-.5*d.sum(2)*np.exp(-2*p[0]))
            parts = np.exp(-.5*d*np.exp(-2*(p[1]+p[3:7])))
            low = parts.mean(2)
            if family == 'mixed':
                low = .35*low + .65*sum(parts[:,:,i]*parts[:,:,j] for i,j in pairs)/6
            weight = 1/(1+np.exp(-p[2]))
            return (1-weight)*full + weight*low
        def penalty(p):
            return .3*np.sum(p[3:7]**2) + .025*p[2]**2
    else:
        start, bounds = [.55,-8.], [(-1.2,2.4),(-15,-2)]
        def kernel(a,b,d,p):
            q=d.sum(2)*np.exp(-2*p[0]); return np.exp(-.5*q)
        def penalty(p):
            return .02*p[0]**2
    def loss(p):
        K = kernel(x,x,D,p)
        L = cholesky(U.T @ K @ U + np.exp(p[-1])*I,lower=True)
        v = solve_triangular(L,r,lower=True)
        return np.log(np.diag(L)).sum() + len(r)/2*np.log(max(v@v/len(r),1e-30)) + penalty(p)
    p = minimize(loss,start,bounds=bounds,method='L-BFGS-B',options={'maxiter':95}).x
    K = kernel(x,x,D,p)
    K.flat[::n+1] += np.exp(p[-1])
    L = cholesky(K,lower=True)
    w = solve_triangular(L,np.column_stack((F,y)),lower=True)
    coef = np.linalg.lstsq(w[:,:-1],w[:,-1],rcond=None)[0]
    return G@coef + kernel(z,x,Dz,p)@solve_triangular(L.T,w[:,-1]-w[:,:-1]@coef)
def _gp(x, z, y, length, nugget, matern=False):
    scale = x.std(axis=0)
    x, z = x / scale, z / scale
    def kernel(a, b):
        d = cdist(a, b, 'sqeuclidean') / length**2
        if not matern:
            return np.exp(-0.5 * d)
        r = np.sqrt(5 * d)
        return (1 + r + r*r/3) * np.exp(-r)
    f, g = _quadratic(x), _quadratic(z)
    K = kernel(x, x)
    K.flat[::len(x)+1] += nugget
    L = cholesky(K, lower=True)
    w = solve_triangular(L, np.column_stack([f, y]), lower=True)
    c = np.linalg.lstsq(w[:, :-1], w[:, -1], rcond=None)[0]
    return g @ c + kernel(z, x) @ solve_triangular(L.T, w[:, -1] - w[:, :-1] @ c)


def _quadratic(x):
    return np.column_stack([np.ones(len(x)), x] + [x[:, i] * x[:, j] for i in range(4) for j in range(i, 4)])



_READOUT_MODELS = ((1, 'iso'),)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, z = train['x'], study['test']['x']
    t = train['traj']
    mode = np.linalg.svd(t, full_matrices=False)[2][0]
    y = np.mean([_smooth(x, z, train['y'], d, family) if family != 'baseline'
                 else _gp(x, z, train['y'], 1.5, .0001)
                 for d, family in _READOUT_MODELS], axis=0)
    amp = _gp(x, z, t @ mode, 2.4, .001, matern=True)
    state = amp[:, None] * mode
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
