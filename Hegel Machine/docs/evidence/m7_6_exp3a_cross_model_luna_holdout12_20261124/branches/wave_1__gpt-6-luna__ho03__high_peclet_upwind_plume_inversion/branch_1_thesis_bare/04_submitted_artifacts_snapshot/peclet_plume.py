"""Scharfetter-Gummel finite-volume model and robust point-source inversion."""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares


def _bernoulli(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1.e-5
    out[small] = 1. - z[small]/2. + z[small]**2/12. - z[small]**4/720.
    pos = z > 50.
    out[pos] = z[pos]*np.exp(-z[pos])
    neg = z < -50.
    out[neg] = -z[neg]
    mid = ~(small | pos | neg)
    out[mid] = z[mid]/np.expm1(z[mid])
    return out


def _operator(ux_face, uy_face, diffusivity, kappa, h):
    ux, uy = np.asarray(ux_face, float), np.asarray(uy_face, float)
    n = ux.shape[1]
    if ux.shape != (n+1,n) or uy.shape != (n,n+1):
        raise ValueError('face velocities must have shapes (N+1,N) and (N,N+1)')
    d, h = float(diffusivity), float(h)
    if d <= 0 or h <= 0:
        raise ValueError('diffusivity and h must be positive')
    px, py = ux*h/d, uy*h/d
    bx, bmx = _bernoulli(px), _bernoulli(-px)
    by, bmy = _bernoulli(py), _bernoulli(-py)
    s = d/(h*h)
    a = lil_matrix((n*n,n*n), dtype=float)
    for i in range(n):
        for j in range(n):
            r = i*n+j
            diag = float(kappa)
            diag += s*(bmx[i+1,j] + bx[i,j])
            if i+1 < n: a[r,(i+1)*n+j] -= s*bx[i+1,j]
            if i > 0: a[r,(i-1)*n+j] -= s*bmx[i,j]
            diag += s*(bmy[i,j+1] + by[i,j])
            if j+1 < n: a[r,i*n+j+1] -= s*by[i,j+1]
            if j > 0: a[r,i*n+j-1] -= s*bmy[i,j]
            a[r,r] += diag
    return a.tocsc()


def forward_peclet_field(ux_face, uy_face, diffusivity, kappa, source_map, h):
    """Solve div(u c - D grad c)+kappa*c=source with zero ghost values."""
    src = np.asarray(source_map, dtype=float)
    n = src.shape[0]
    if src.shape != (n,n):
        raise ValueError('source_map must be square')
    mat = _operator(ux_face, uy_face, diffusivity, kappa, h)
    return splu(mat).solve(src.ravel()).reshape((n,n))


def invert_point_source(problem):
    n, h = int(problem['N']), float(problem['h'])
    ux, uy = problem['ux_face'], problem['uy_face']
    d, kappa, sigma = float(problem['D']), float(problem['kappa']), float(problem['sigma_s'])
    ij = np.asarray(problem['sensor_ij'], dtype=int)
    obs = np.asarray(problem['c_obs'], dtype=float)
    lu = splu(_operator(ux, uy, d, kappa, h))
    x = (np.arange(n)+0.5)*h
    X, Y = np.meshgrid(x, x, indexing='ij')
    def predict(par):
        x0, y0, q = par
        src = q*np.exp(-0.5*((X-x0)**2+(Y-y0)**2)/(sigma*sigma))
        field = lu.solve(src.ravel()).reshape(n,n)
        return field[ij[:,0], ij[:,1]]
    scale = max(float(np.median(np.abs(obs))), 1.e-12)
    best = None
    for start in ([.25,.25,1.], [.5,.5,1.], [.75,.75,1.], [.25,.75,1.], [.75,.25,1.]):
        result = least_squares(lambda p: (predict(p)-obs)/scale, start,
            bounds=([0.,0.,0.],[n*h,n*h,np.inf]), loss='soft_l1', f_scale=0.08,
            x_scale=[0.3,0.3,1.], max_nfev=90)
        if best is None or result.cost < best.cost:
            best = result
    return {'source_xy': [float(best.x[0]), float(best.x[1])], 'strength_Q': float(best.x[2])}
