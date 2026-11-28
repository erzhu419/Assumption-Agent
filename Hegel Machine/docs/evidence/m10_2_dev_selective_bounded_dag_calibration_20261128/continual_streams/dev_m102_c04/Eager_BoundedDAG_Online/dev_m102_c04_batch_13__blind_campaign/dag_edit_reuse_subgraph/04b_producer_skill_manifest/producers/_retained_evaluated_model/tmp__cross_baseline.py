import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    a = _affine(u)
    mask = np.ones(len(u), dtype=bool)
    for _ in range(8):
        w = _ls(a[mask], v[mask])
        e = np.linalg.norm(a @ w - v, axis=1)
        new = e <= max(1e-10, 4 * np.median(e))
        if new.sum() < 8 or np.array_equal(mask, new):
            break
        mask = new
    d = np.array([_ls(a[mask][:, [0, j+1]], v[mask, j]) for j in range(4)]).T
    if np.median(np.linalg.norm((u*d[1]+d[0]-v)[mask], axis=1)) <= 1.35*np.median(e[mask])+1e-10:
        return d[0], np.diag(d[1])
    return w[0], w[1:]


def _panel_operators(panel):
    u = panel['u']
    def design(x, k, j):
        if j == 1:
            h = x / (1 + k * np.abs(x))
        else:
            h = x*x / (1 + k*x*x)
        return np.column_stack((_affine(x), h))
    def fit(j, initial):
        v = panel[f'v{j}']
        def residual(k):
            f = design(u, k[0], j)
            return (f @ _ls(f, v) - v).ravel()
        k = least_squares(residual, [initial], bounds=(0.01, 5.)).x[0]
        w = _ls(design(u, k, j), v)
        return lambda x: design(x, k, j) @ w
    return fit(1, 0.8), fit(2, 0.16)


def _features(p, r):
    # A common diagonal coefficient is represented by the dot product.
    common = np.column_stack((p, r, np.einsum('ij,ij->i', p, r)))
    offdiag = np.column_stack([p[:, i]*r[:, j] for i in range(4)
                               for j in range(4) if i != j])
    return common, offdiag


def _ridge(f, target, query, penalty):
    fm, ym = f.mean(axis=0), target.mean(axis=0)
    a = f-fm
    w = np.linalg.solve(a.T @ a + penalty*np.eye(f.shape[1]), a.T @ (target-ym))
    return (query-fm) @ w + ym


def _fit_r(panel, mode):
    from scipy.optimize import minimize_scalar
    u, z = panel['u'], panel['v2']
    def design(x,k):
        a=_affine(x)
        i,j=np.triu_indices(x.shape[1])
        p=x[:,i]*x[:,j]
        n=np.sum(x*x,axis=1,keepdims=True)
        cross=p[:,i!=j]
        if mode=='poly': return np.c_[a,p]
        if mode=='radial': return np.c_[a,p]/(1+k*n)
        if mode=='cross': return np.c_[a,cross]
        if mode=='hybrid': return np.c_[a,x*x/(1+k*x*x),cross]
        if mode=='bounded_cross': return np.c_[a,x*x/(1+k*x*x),cross/(1+k*cross*cross)]
        if mode=='scale_mix': return np.c_[a,x*x/(1+.22*x*x),cross/(1+k*cross*cross)]
        if mode=='cross_scale': return np.c_[a,x*x/(1+k*x*x),cross/(1+.22*cross*cross)]
        if mode=='cycle_mix': return np.c_[a,x*x/(1+k*x*x),x*np.roll(x,-1,axis=1)]
    def loss(k):
        f=design(u,k); e=f@_ls(f,z)-z; return np.mean(e*e)
    k=minimize_scalar(loss,bounds=(0.,5.),method='bounded').x if mode not in ('poly','cross') else 0.
    w=_ls(design(u,k),z)
    return lambda x: design(x,k) @ w

def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x, xt = train['x'], study['test']['x']
    offset, matrix = _calibration(study['panel_a'])
    q = np.linalg.solve(matrix.T, (np.concatenate((x, xt))-offset).T).T
    p_op, r_op = _panel_operators(study['panel_c'])
    r_op = _fit_r(study["panel_c"], 'cross')
    common, offdiag = _features(p_op(q), r_op(q))
    n = len(x)
    y = _ridge(common[:n], train['y'], common[n:], 1e-6)
    return {'common':common, 'offdiag':offdiag, 'y_hat':y}
