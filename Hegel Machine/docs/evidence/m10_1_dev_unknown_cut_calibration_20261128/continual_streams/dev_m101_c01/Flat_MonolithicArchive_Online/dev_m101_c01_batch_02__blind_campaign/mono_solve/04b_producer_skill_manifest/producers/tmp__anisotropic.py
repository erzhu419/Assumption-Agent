import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def extrapolate(x, xt, y, basis_fn):
    # Profile the quadratic trend before estimating residual length scales.
    f, ft = basis_fn(x), basis_fn(xt)
    q = np.linalg.qr(f, mode='complete')[0][:, f.shape[1]:]
    centered = q.T @ y
    centered = centered.reshape(len(q.T), -1)
    scale = np.maximum(centered.std(0), 1e-10)
    centered /= scale
    dx = (x[:, None, :] - x[None, :, :])**2
    dxt = (xt[:, None, :] - x[None, :, :])**2
    eye = np.eye(len(x))
    def kernel(d, p):
        a = np.exp(-0.5*d*np.exp(-2*p[:4]))
        return a.prod(2) + np.exp(p[4])*a.mean(2)
    def objective(p):
        g = q.T @ (kernel(dx, p) + np.exp(p[5])*eye) @ q
        c = cholesky(g, lower=True)
        u = cho_solve((c, True), centered)
        v = (u*centered).sum(0) / len(g)
        return np.log(v).sum()/2 + np.log(c.diagonal()).mean()*centered.shape[1]
    init = np.log([1.2]*4+[.2,.003]); bd = [(-1.6,1.35)]*4+[(-5,2),(-11,-1)]
    p = minimize(objective, init, method='L-BFGS-B', bounds=bd, options={'maxiter':65}).x
    g = kernel(dx,p)+np.exp(p[5])*eye
    c = cholesky(g, lower=True)
    fi = cho_solve((c, True),f)
    beta = np.linalg.solve(f.T@fi, fi.T@y)
    pred = ft@beta + kernel(dxt,p)@cho_solve((c,True),y-f@beta)
    return pred
