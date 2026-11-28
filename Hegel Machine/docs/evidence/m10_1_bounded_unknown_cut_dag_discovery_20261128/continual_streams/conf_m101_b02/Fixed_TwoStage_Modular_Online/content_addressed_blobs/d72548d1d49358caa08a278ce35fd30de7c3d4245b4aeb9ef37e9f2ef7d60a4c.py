import numpy as np
from scipy.optimize import least_squares


def _a_basis(u, j):
    d = [np.ones(len(u)), *u.T, u[:, j]**3, u[:, j]*u[:, (j+1)%4], u[:, j]/(1+u[:, j]**2), np.cos(u[:, 0])]
    if j:
        d.append(np.cos(u[:, j]))
    return np.column_stack(d)


def _invert(p, v):
    u, y = p['u'], p['v']
    h = [_a_basis(u,j) for j in range(4)]
    w = np.ones(len(u))
    for _ in range(24):
        q = np.sqrt(w)
        b = [np.linalg.lstsq(a*q[:,None], y[:,j]*q, rcond=None)[0] for j,a in enumerate(h)]
        r = np.linalg.norm(np.column_stack([a@c for a,c in zip(h,b)])-y, axis=1)
        w = 1/(1+(r/max(.001,1.5*np.median(r)))**2)
    keep = np.argsort(r)[:-3]
    b = [np.linalg.lstsq(a[keep],y[keep,j],rcond=None)[0] for j,a in enumerate(h)]
    def forward(z):
        return np.column_stack([_a_basis(z,j)@b[j] for j in range(4)])
    return np.array([least_squares(lambda z: forward(z[None,:])[0]-row, row, max_nfev=60).x for row in v])


def _panel(p, key, z, fun, k, square=False):
    def basis(x):
        return np.column_stack([np.ones(len(x)),x,fun(k*x)]+([x*x] if square else []))
    return basis(z) @ np.linalg.lstsq(basis(p['u']),p[key],rcond=None)[0]


def _responses(s, z):
    b = _panel(s['panel_b'],'v',z,np.sin,2.8)
    c = _panel(s['panel_c'],'v1',z,np.tanh,2.28)
    d = _panel(s['panel_c'],'v2',z,np.cos,2.5,True)
    return b,c,d


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    z = _invert(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    b, c, d = _responses(study, z)
    b_sat = np.tanh(2 * b)
    # Both branches contribute to the shared fit through all channel pairs.
    coupling = (b_sat[:, :, None] * c[:, None, :]).reshape(len(z), -1)
    g = np.column_stack((np.ones(len(z)), 0.22 * np.column_stack((z, b_sat, coupling))))
    fd = 0.03 * np.column_stack((b, b * z))
    cd = 0.03 * np.column_stack((c, d, c * d))
    return dict(g=g, fd=fd, cd=cd, y=train['y'], traj=train['traj'])
