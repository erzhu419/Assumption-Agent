"""Predict both readouts from a robustly calibrated panel composition."""
import numpy as np
from scipy.optimize import least_squares


def _intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-11)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    a = _intercept(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        r = np.linalg.norm(a @ _ls(a[keep], v[keep]) - v, axis=1)
        keep = r < max(.002, 3*np.median(r))
    opposite = (np.arange(4) + 2) % 4

    def nonlinear(q):
        return np.tanh(q[0]*u[keep]), np.sin(q[1]*u[keep][:, opposite])

    def fit(q):
        f, g = nonlinear(q)
        # Offsets and gains are per channel; nonlinear amplitudes are shared.
        n = len(f)
        d = np.zeros((n, 4, 10))
        j = np.arange(4)
        d[:, j, j] = 1
        d[:, j, j+4] = u[keep]
        d[:, :, 8], d[:, :, 9] = f, g
        d = d.reshape(-1, 10)
        c = _ls(d, v[keep].ravel())
        return c, d @ c - v[keep].ravel()

    q = least_squares(lambda q:fit(q)[1], [1.8,2.4], bounds=([1.,1.8],[2.6,3.]), gtol=1e-13).x
    c, _ = fit(q)
    def forward(z):
        return c[:4] + c[4:8]*z + c[8]*np.tanh(q[0]*z) + c[9]*np.sin(q[1]*z[opposite])
    return np.array([least_squares(lambda z:forward(z)-r, (r-c[:4])/c[4:8], gtol=1e-12).x for r in x]).reshape(-1,4)


def _middle(panel, z):
    u, v = panel['u'], panel['v']
    def design(x, q):
        return _intercept(np.column_stack((x,np.tanh(q[0]*x),np.arcsinh(q[1]*x))))
    def residual(q):
        d = design(u, q)
        inv = np.linalg.pinv(d)
        lev = np.sum(d*inv.T, axis=1)
        return ((d @ (inv @ v) - v)/np.maximum(1-lev[:,None],.002)).ravel()
    q = least_squares(residual,[5.5,6.],bounds=(.08,8.)).x
    return design(z,q) @ _ls(design(u,q),v)


def _features(study, x):
    z = _calibrate(study['panel_a'], x)
    w = _middle(study['panel_b'], z)
    p = study['panel_c']
    i,j=np.triu_indices(4,1)
    u=p['u']
    d=u[:,i]*u[:,j]
    active=np.linalg.norm(_ls(_intercept(d),p['v1'])[1:],axis=1) > .002
    h=_intercept(w[:,i[active]]*w[:,j[active]]) @ _ls(_intercept(d[:,active]),p['v1'])
    cross=(z[:,:,None]*h[:,None,:]).reshape(len(z),16)
    return z,w,h,cross


def _ridge(x, y, test, penalty):
    mu=x.mean(axis=0)
    sd=np.maximum(x.std(axis=0),1e-10)
    x=(x-mu)/sd
    target=y.mean(axis=0)
    c=np.linalg.solve(x.T@x+penalty*np.eye(x.shape[1]), x.T@(y-target))
    return target+(test-mu)/sd @ c


def solve_batch(study: dict) -> dict:
    tr=study['train']
    n=len(tr['x'])
    z,w,h,cross=_features(study,np.vstack((tr['x'],study['test']['x'])))
    f=np.column_stack((z,h,cross))
    y=_ridge(f[:n],tr['y'],f[n:],.01)
    traj=np.asarray(tr['traj'])
    modes=np.linalg.svd(traj,full_matrices=False)[2][:2]
    state=_ridge(f[:n],traj@modes.T,f[n:],.02)@modes
    end=state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
