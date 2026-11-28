"""Panel-conditioned Gaussian regression for scalar and trajectory outputs."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar


def affine(x):
    return np.column_stack((np.ones(len(x)), x))


def fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def coordinates(study, x):
    a, b, c = (study['panel_'+key] for key in 'abc')
    offset, gain = [], []
    for u, v in zip(a['u'].T, a['v'].T):
        d = affine(u[:,None]); k = np.argsort(u); l, r = np.triu_indices(len(u), 1)
        slope = np.median((v[k[r]]-v[k[l]]) / (u[k[r]]-u[k[l]]))
        coeff = np.array([np.median(v-slope*u), slope])
        for _ in range(5):
            keep = np.argsort(abs(v-d@coeff))[:int(.8*len(u))]
            coeff = fit(d[keep], v[keep])
        offset.append(coeff[0]); gain.append(coeff[1])
    cx = (x-offset)/gain
    bx = affine(x)@fit(affine(b['u']), b['v'])
    u = c['u']
    vmap = lambda q: affine(np.arctan(2.12*q))
    def wmap(q, a):
        return np.column_stack((affine(q), q*q/(1+a*q*q)))
    def loss(a):
        d = wmap(u, a)
        return np.sum((d@fit(d, c['v2'])-c['v2'])**2)
    alpha = minimize_scalar(loss, bounds=(0., 1.), method='bounded').x
    v = fit(vmap(u), c['v1']); w = fit(wmap(u, alpha), c['v2'])
    return cx, bx, vmap(cx)@v, wmap(bx,alpha)@w


def gp(x, xt, y, additive=False):
    mu, sd = y.mean(), max(y.std(), 1e-12)
    yy = (y-mu)/sd
    n, d = x.shape
    sq = (x[:,None,:]-x[None,:,:])**2
    dsq = (xt[:,None,:]-x[None,:,:])**2
    def kernel(dist, th):
        w = np.exp(th[:d]); e = -.5*dist*w
        per = np.exp(e) if additive else None
        k = np.exp(e.sum(2))
        if additive:
            extra = np.exp(th[d])*per/d
            k = k + extra.sum(2)
        return k, per if not additive else extra, e
    def objective(th):
        k, extra, e = kernel(sq, th)
        noise = np.exp(th[-1]); k.flat[::n+1] += noise
        fac = cho_factor(k, check_finite=False)
        al = cho_solve(fac, yy, check_finite=False)
        g = .5*(cho_solve(fac, np.eye(n), check_finite=False)-np.outer(al,al))
        joint = k.copy(); joint.flat[::n+1] -= noise
        if additive:
            joint -= extra.sum(2)
        grad = np.einsum('ij,ijk->k', g, e*(joint[:,:,None]+(extra if additive else 0.)))
        if additive:
            grad = np.r_[grad,np.sum(g*extra.sum(2))]
        return .5*yy@al+np.log(np.diag(fac[0])).sum(), np.r_[grad, noise*np.trace(g)]
    th = minimize(objective, np.r_[np.zeros(d),[-2.] if additive else [],-9.], jac=True,
                  method='L-BFGS-B', bounds=[(-11,7)]*d+([(-10,5)] if additive else [])+[(-15,-2)]).x
    k = kernel(sq,th)[0]; k.flat[::n+1] += np.exp(th[-1])
    return mu+sd*(kernel(dsq,th)[0]@cho_solve(cho_factor(k),yy))


def solve_batch(study):
    tr = study['train']; x, xt = tr['x'], study['test']['x']; n = len(x)
    cx, bx, v, w = coordinates(study, np.concatenate((x,xt)))
    q = np.column_stack((cx,v)); y = gp(q[:n],q[n:],tr['y'],True)
    q = np.column_stack((bx,w)); z = tr['traj']; _, _, vt = np.linalg.svd(z,full_matrices=False)
    a = gp(q[:n],q[n:],z@vt[0]); state = a[:,None]*vt[0]
    return {'y_hat':y, 'state_pred':state,
            'diagnostics_vec':np.array([y.mean(), y.std(), state[:,-1].mean(), state[:,-1].std()])}
