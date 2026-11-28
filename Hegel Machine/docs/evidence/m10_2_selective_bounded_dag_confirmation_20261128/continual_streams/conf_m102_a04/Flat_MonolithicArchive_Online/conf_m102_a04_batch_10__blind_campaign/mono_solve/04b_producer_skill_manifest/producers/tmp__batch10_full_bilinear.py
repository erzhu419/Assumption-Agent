"""Cyclic calibration, panel operators, and separate response readouts."""
import numpy as np
from scipy.optimize import least_squares


def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def one(z):
    return np.c_[np.ones(len(z)), z]


def _panel_maps(panel, z):
    u = np.asarray(panel['u'])

    def odd(w, k):
        return one(np.c_[np.tanh(k*w), w, w**3, w**5])

    def even(w, k):
        return one(np.cos(k*w))

    def fit(design, key, start, bounds):
        y = np.asarray(panel[key])
        def loss(q):
            d = design(u, q[0])
            return (d @ lin(d, y) - y).ravel()
        k = least_squares(loss, [start], bounds=bounds, gtol=1e-13).x[0]
        return design(z, k) @ lin(design(u, k), y)

    return fit(odd, 'v1', 1.77, (1.3, 2.3)), fit(even, 'v2', 2.4, (1.8, 2.8))


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
    train = study['train']
    n = len(train['x'])
    z = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    g, h = _panel_maps(study['panel_c'], z)
    cross = (g[:, :, None]*h[:, None, :]).reshape(len(z),16)
    y = _readout(np.c_[g, h, (g*h).sum(1)], train['y'], n, 1e-5)
    times = _time_basis(np.asarray(study['eval_times']), train['traj'])
    amp = lin(times, train['traj'].T).T
    state = _readout(np.c_[g,h,cross], amp, n, 1e-5) @ times.T
    terminal = state[:, -1]
    return dict(y_hat=y, state_pred=state, diagnostics_vec=np.array([y.mean(), y.std(), terminal.mean(), terminal.std()]))
