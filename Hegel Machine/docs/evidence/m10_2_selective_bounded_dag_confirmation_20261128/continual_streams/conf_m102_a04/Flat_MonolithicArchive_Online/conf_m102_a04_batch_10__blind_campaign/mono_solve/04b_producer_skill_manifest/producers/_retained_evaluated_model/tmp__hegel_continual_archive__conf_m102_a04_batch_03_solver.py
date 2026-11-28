"""Cyclic reference calibration and two-mode bilinear response prediction."""
import numpy as np
from scipy.optimize import least_squares


def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def one(z):
    return np.c_[np.ones(len(z)), z]


def _panel_maps(panel, z):
    u = np.asarray(panel['u'], dtype=float)

    # Both responses share their nonlinear shape parameters across coordinates.
    def softsign(w, q):
        return one(w / (1 + q[0]*abs(w)) + q[1]*w)

    def curvature(w, q):
        w2 = w*w
        return np.c_[one(w), w2, w2/(1+q[0]*w2)]

    def fit(features, target, initial, bounds, **options):
        def residual(q):
            f = features(u, q)
            return (f @ lin(f, target) - target).ravel()
        q = least_squares(residual, initial, bounds=bounds, **options).x
        return features(z, q) @ lin(features(u, q), target)

    c = fit(softsign, panel['v1'], [.85, -.01], ([.02, -1], [10, 1]))
    d = fit(curvature, panel['v2'], [.25], (.08, .65), gtol=1e-13)
    return np.c_[c, d]


def _time_basis(t, trajectory):
    def basis(r):
        return -np.expm1(-t[:, None]*r)

    def residual(r):
        f = basis(r)
        return (f @ lin(f, trajectory.T) - trajectory.T).ravel()

    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .8], [.8, 4])).x
    return basis(rates)


def _calibrate(panel, raw):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    def design(a,b):
        return np.stack((np.ones_like(a),a,b,a*b,a**3),axis=-1)
    f = design(u,np.roll(u,-1,axis=1))
    keep = np.linalg.norm(v,axis=1)<4
    for _ in range(4):
        w = np.stack([lin(f[keep,i],v[keep,i]) for i in range(4)])
        r = np.linalg.norm(np.einsum('nik,ik->ni',f,w)-v,axis=1)
        keep = r <= max(.002,5*np.median(r))
    def invert(x):
        def forward(z):
            return np.sum(design(z,np.roll(z,-1))*w,axis=1)-x
        return least_squares(forward,(x-w[:,0])/w[:,1],gtol=1e-11).x
    return np.stack([invert(x) for x in raw])


def _readout(f, target, n, reg):
    p=(f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-7)
    p=one(p)
    penalty=reg*np.eye(p.shape[1]); penalty[0,0]=0
    return p[n:]@np.linalg.solve(p[:n].T@p[:n]+penalty,p[:n].T@target)


def solve_batch(study: dict) -> dict:
    train=study['train']; n=len(train['x'])
    z=_calibrate(study['panel_a'],np.vstack([train['x'],study['test']['x']]))
    g,h=np.split(_panel_maps(study['panel_c'],z),2,axis=1)
    prod=g*h
    y=_readout(np.c_[g,h,prod.sum(1)],train['y'],n,.001)
    f=_time_basis(np.asarray(study['eval_times']),train['traj'])
    amp=lin(f,train['traj'].T).T
    cross=(g[:,:,None]*h[:,None,:]).reshape(len(z),16)
    state=_readout(np.c_[g,h,cross],amp,n,.0001)@f.T
    terminal=state[:,-1]
    return dict(y_hat=y,state_pred=state,diagnostics_vec=np.array([y.mean(),y.std(),terminal.mean(),terminal.std()]))
