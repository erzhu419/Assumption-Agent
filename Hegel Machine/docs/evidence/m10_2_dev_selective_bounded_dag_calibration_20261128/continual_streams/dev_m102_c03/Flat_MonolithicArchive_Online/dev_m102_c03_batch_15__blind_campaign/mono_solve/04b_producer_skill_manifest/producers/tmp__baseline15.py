"""Predict scalar responses and relaxation trajectories from calibrated inputs."""
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = _aff(u)
    r = np.linalg.norm(v - a @ np.linalg.lstsq(a, v, rcond=None)[0], axis=1)
    keep = r < max(3*np.median(r), 1e-3)
    def design(z, j):
        w = z[:, j]
        p = z[:, (j+1)%4]
        return np.column_stack((np.ones(len(z)), w, p, w*p, w**3))
    def estimate(mask):
        return np.column_stack([np.linalg.lstsq(design(u[mask], j), v[mask, j], rcond=None)[0] for j in range(4)])
    def forward(z, b):
        return np.column_stack([design(z, j) @ b[:, j] for j in range(4)])
    b = estimate(keep)
    r = np.linalg.norm(v - forward(u, b), axis=1)
    keep = r < max(5*np.median(r), .001)
    b = estimate(keep)
    lin = np.diag(b[1]); lin[(np.arange(4)+1)%4, np.arange(4)] = b[2]
    z0 = np.linalg.solve(lin.T, (x-b[0]).T).T
    return np.array([least_squares(lambda z: forward(z[None,:], b)[0]-row, init, gtol=1e-12).x for row, init in zip(x,z0)])


def _branch(panel, z, key, kind):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u',key))
    if kind == 'soft':
        fun = lambda x,p: x/(1+np.abs(x*p))
        start, bounds = .84, (.1, 4.)
    else:
        fun = lambda x,p: np.tanh(x*p)
        start, bounds = 1.2, (.25, 3.)
    def design(x,p):
        if kind == "soft":
            return np.column_stack((_aff(x), fun(x,p)))
        return np.column_stack((_aff(x), x*np.roll(x,-1,axis=1), fun(x,p)))
    def profile(p, predict=False):
        a = design(u,p)
        b = np.linalg.lstsq(a,v,rcond=1e-10)[0]
        return design(z,p) @ b if predict else (a @ b-v).ravel()
    p = least_squares(profile, np.full(4,start), bounds=bounds, gtol=1e-11).x
    return profile(p, True)


def _design(z,c,d):
    return np.column_stack((_aff(z), c, d, (c[:,:,None]*d[:,None,:]).reshape(len(z),16)))


def _time(t, p):
    return -np.expm1(-t[:,None]*np.exp(p))


def solve_batch(study: dict) -> dict:
    train = study['train']; n = len(train['x'])
    z = _calibrate(study['panel_a'], np.vstack((train['x'],study['test']['x'])))
    c = _branch(study['panel_c'],z,'v1','soft'); d = _branch(study['panel_c'],z,'v2','cyclic')
    f = _design(z,c,d)
    t = np.asarray(study['eval_times']); traj = np.asarray(train['traj'])
    def residual(p):
        q = _time(t,p)
        return (traj.T - q @ np.linalg.lstsq(q,traj.T,rcond=None)[0]).ravel()
    rates = least_squares(residual, np.log([.4,1.8]), bounds=(np.log([.12,1.0]),np.log([.9,4.]))).x
    q = _time(t,rates)
    a = np.linalg.lstsq(q,traj.T,rcond=None)[0].T
    pred = f[n:] @ np.linalg.lstsq(f[:n], np.column_stack((train['y'],a)),rcond=1e-9)[0]
    yh, state = pred[:,0], pred[:,1:] @ q.T
    last = state[:,-1]
    return dict(y_hat=yh,state_pred=state,diagnostics_vec=np.array([yh.mean(),yh.std(),last.mean(),last.std()]))
