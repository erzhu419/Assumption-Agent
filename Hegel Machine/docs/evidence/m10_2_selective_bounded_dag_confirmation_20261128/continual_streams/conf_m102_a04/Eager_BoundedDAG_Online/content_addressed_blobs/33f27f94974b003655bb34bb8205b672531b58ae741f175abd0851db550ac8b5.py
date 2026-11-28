import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _ls(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-10)[0]


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    center = np.median(v, axis=0)
    r = np.linalg.norm(v-center, axis=1)
    good = r <= 3.5*np.median(r)
    u, v = u[good], v[good]
    partner = (np.arange(u.shape[1])+2) % u.shape[1]
    def design(z, q, j):
        a, b = z[:,j], z[:,partner[j]]
        return np.column_stack((np.ones(len(z)), a, np.arctan(q[0]*a), np.sin(q[1]*b), b))
    def fit(q):
        f = [design(u, q, j) for j in range(u.shape[1])]
        return f, [_ls(fj, v[:,j]) for j, fj in enumerate(f)]
    def residual(q):
        f, coef = fit(q)
        return np.column_stack([fj@cj-v[:,j] for j,(fj,cj) in enumerate(zip(f,coef))]).ravel()
    q = least_squares(residual, [1.4, 2.15], bounds=([.3,1.2],[4.,3.4]), gtol=1e-12).x
    _, coef = fit(q)
    def forward(z):
        return np.column_stack([design(z, q, j)@cj for j,cj in enumerate(coef)])
    z = np.array(x, dtype=float, copy=True)
    c = np.asarray(coef)
    for _ in range(25):
        r = forward(z)-x
        if np.max(np.abs(r)) < 1e-10:
            break
        jac = np.zeros((len(z),z.shape[1],z.shape[1]))
        i = np.arange(z.shape[1])
        jac[:,i,i] = c[:,1]+c[:,2]*q[0]/(1+(q[0]*z)**2)
        jac[:,i,partner] = c[:,3]*q[1]*np.cos(q[1]*z[:,partner])+c[:,4]
        dz = np.linalg.solve(jac, r[...,None])[...,0]
        z -= np.clip(dz, -.35, .35)
    return z


def _map(panel, z):
    u = np.asarray(panel['u'], dtype=float)
    i,j = np.triu_indices(u.shape[1],1)
    def design(x, q, first):
        extra = x**3 if first else x[:,i]*x[:,j]
        return np.column_stack((np.ones(len(x)), x, np.arctan(q*x), extra))
    result = []
    for first,key in ((True,'v1'),(False,'v2')):
        y = np.asarray(panel[key], dtype=float)
        def residual(q):
            f = design(u,q,first)
            return np.mean((f@_ls(f,y)-y)**2)
        q = minimize_scalar(residual, bounds=(.3,3.5), method='bounded').x
        result.append(design(z,q,first)@_ls(design(u,q,first),y))
    return np.column_stack(result)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    n = len(train)
    z = _calibrate(study['panel_a'], np.vstack((train, test)))
    w = _map(study['panel_c'], z)
    w = (w-w[:n].mean(axis=0))/np.maximum(w[:n].std(axis=0), .1)
    return {'features': w, 'n_train': np.array(n, dtype=int)}
