"""Nonlinear panel calibration and additive kernel features for frozen ridge heads."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(s, x):
    a = s['panel_a']
    u, v = a['u'], a['v']
    i, j = np.triu_indices(len(u), 1)
    off, gain = [], []
    for p, q in zip(u.T, v.T):
        delta = p[j] - p[i]
        ok = np.abs(delta) > 1e-10
        slope = np.median((q[j][ok] - q[i][ok])/delta[ok]) if np.any(ok) else 1.
        coef = np.array([np.median(q - slope*p), slope])
        d = _affine(p[:,None])
        for _ in range(4):
            r = np.abs(q-d@coef)
            keep = r <= max(1e-8, 6*np.median(r))
            if keep.sum() < 2:
                break
            coef = np.linalg.lstsq(d[keep], q[keep], rcond=None)[0]
        off.append(coef[0]); gain.append(coef[1] if abs(coef[1]) > 1e-6 else 1.)
    cx = (x-off)/gain
    # Panel C v1 supports an arctangent near scale 2. Keep the
    # saturating channels separate rather than remixing them into v1.
    return np.column_stack((cx, np.arctan(2.0*cx)))


def _dist(x):
    return (x[:,None,:]-x[None,:,:])**2


def _kernel(sq, t):
    d = sq.shape[-1]
    exponent = -.5 * sq * np.exp(t[:d])
    joint = np.exp(exponent.sum(axis=-1))
    add = np.exp(exponent + t[d])/d
    return joint, add, exponent


def _learn(sq, y):
    n, _, d = sq.shape
    sd = np.std(y)
    y = (y - np.mean(y))/max(sd,1e-10)
    def objective(t):
        joint, add, exp = _kernel(sq,t)
        k = joint + add.sum(axis=-1)
        noise = np.exp(t[-1]); k.flat[::n+1] += noise
        f = cho_factor(k, check_finite=False)
        alpha = cho_solve(f,y,check_finite=False)
        g = .5*(cho_solve(f,np.eye(n),check_finite=False)-np.outer(alpha,alpha))
        grad = np.einsum('ij,ijk->k',g,exp*(joint[:,:,None]+add))
        return .5*y@alpha+np.log(np.diag(f[0])).sum(), np.r_[grad,np.sum(g*add.sum(axis=-1)),noise*np.trace(g)]
    return minimize(objective,np.r_[np.zeros(d),-2.,-9.],jac=True,method='L-BFGS-B',bounds=[(-11.,7.)]*d+[(-10.,5.),(-15.,-2.)],options={'maxiter':110}).x


def _gram(sq,t):
    j,a,_ = _kernel(sq,t)
    return (j+a.sum(axis=-1))/np.exp(t[-1])


def _features(g,n,penalty):
    # A training-landmark square root gives exactly the requested kernel ridge fit.
    ev, u = np.linalg.eigh(g[:n,:n])
    good = ev > max(ev[-1]*1e-12,1e-12)
    return g[:,:n] @ (u[:,good]*np.sqrt(penalty/ev[good]))


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']; tr = s['train']; n = len(tr['x'])
    x = np.concatenate((tr['x'],s['test']['x']))
    q = _coordinates(s,x)
    sq = _dist(q)
    a = tr['traj'] @ np.linalg.svd(tr['traj'],full_matrices=False)[2][0]
    ty = _learn(sq[:n,:n],tr['y'])
    ta = _learn(sq[:n,:n],a)
    gy = _gram(sq,ty)
    ga = _gram(sq,ta)
    return {'raw': _features(0.25*gy + 0.75*ga,n,.014),
            'aligned': _features(ga,n,.014),
            'hybrid': _features(gy,n,.007)}
