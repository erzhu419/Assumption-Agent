"""Conservative Scharfetter-Gummel plume forward model and source inversion."""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares


def _bernoulli(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = np.abs(z) < 1e-6
    out[small] = 1.0 - z[small] / 2.0 + z[small] ** 2 / 12.0
    pos = ~small
    out[pos] = z[pos] / np.expm1(np.clip(z[pos], -700.0, 700.0))
    return out


def _matrix(ux_face, uy_face, diffusivity, kappa, h):
    ux, uy = np.asarray(ux_face), np.asarray(uy_face)
    n = ux.shape[1]
    d = float(diffusivity) / h
    # Cell-centered conservative divergence: each oriented face contributes
    # equal/opposite fluxes; boundary ghosts have zero concentration.
    diag = np.full((n, n), float(kappa))
    rows, cols, vals = [], [], []
    def add(r, c, v):
        rows.append(np.asarray(r).ravel()); cols.append(np.asarray(c).ravel()); vals.append(np.asarray(v).ravel())
    ii, jj = np.indices((n,n)); ids = ii*n + jj
    # West/east boundary fluxes.
    diag += d * _bernoulli(ux[0,:] * h / diffusivity)[None,:] / h
    diag += d * _bernoulli(-ux[n,:] * h / diffusivity)[None,:] / h
    diag += d * _bernoulli(uy[:,0] * h / diffusivity)[:,None] / h
    diag += d * _bernoulli(-uy[:,n] * h / diffusivity)[:,None] / h
    # Interior x faces, directed i -> i+1.
    ax = d * _bernoulli(-ux[1:n,:] * h / diffusivity)
    bx = -d * _bernoulli(ux[1:n,:] * h / diffusivity)
    diag[:-1,:] += ax/h; diag[1:,:] += -bx/h
    add(ids[:-1,:], ids[1:,:], bx/h); add(ids[1:,:], ids[:-1,:], -ax/h)
    # Interior y faces, directed j -> j+1.
    ay = d * _bernoulli(-uy[:,1:n] * h / diffusivity)
    by = -d * _bernoulli(uy[:,1:n] * h / diffusivity)
    diag[:,:-1] += ay/h; diag[:,1:] += -by/h
    add(ids[:,:-1], ids[:,1:], by/h); add(ids[:,1:], ids[:,:-1], -ay/h)
    add(ids, ids, diag)
    from scipy.sparse import coo_matrix
    return coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))), shape=(n*n,n*n)).tocsc()


def forward_peclet_field(ux_face, uy_face, diffusivity, kappa, source_map, h):
    ux, uy = np.asarray(ux_face), np.asarray(uy_face)
    A = _matrix(ux, uy, diffusivity, kappa, h)
    rhs = np.asarray(source_map, dtype=float)
    return splu(A).solve(rhs.ravel()).reshape(ux.shape[1], uy.shape[0])


def invert_point_source(problem):
    n, h = int(problem['N']), float(problem['h'])
    D, kap, sig = float(problem['D']), float(problem['kappa']), float(problem['sigma_s'])
    A = _matrix(problem['ux_face'], problem['uy_face'], D, kap, h)
    lu = splu(A)
    ij = np.asarray(problem['sensor_ij'], dtype=int)
    obs = np.asarray(problem['c_obs'], dtype=float)
    rows, cols = ij[:, 0], ij[:, 1]
    # Robust bounded residual fit. Bounds stay within the physical domain.
    def predict(p):
        x, y, q = p
        xx = (np.arange(n)[:, None] + .5) * h
        yy = (np.arange(n)[None, :] + .5) * h
        src = q * np.exp(-.5*((xx-x)**2+(yy-y)**2)/sig**2)
        field = lu.solve(src.ravel()).reshape(n,n)
        return field[rows, cols]
    X = (np.arange(n)[:, None] + .5) * h
    Y = (np.arange(n)[None, :] + .5) * h
    def unit_predict(x, y):
        src = np.exp(-.5*((X-x)**2+(Y-y)**2)/sig**2)
        return lu.solve(src.ravel()).reshape(n,n)[rows, cols]
    def score(x, y):
        a = unit_predict(x, y)
        q = max(0.0, float(np.dot(a, obs) / max(np.dot(a, a), 1e-30)))
        r = (q*a-obs)**2
        return float(np.mean(np.partition(r, 13)[:14])), q
    L = float(problem.get('L', n*h))
    best = (np.inf, None, None, None)
    for x in np.linspace(.04*L, .96*L, 37):
        for y in np.linspace(.04*L, .96*L, 37):
            v, q = score(x, y)
            if v < best[0]: best = (v, x, y, q)
    def residual(p):
        a = unit_predict(p[0], p[1])
        r = p[2] * a - obs
        return r[np.argsort(np.abs(r))[:14]]
    fit = least_squares(residual, [best[1], best[2], best[3]],
                        bounds=([0.,0.,0.],[L,L,100.]),
                        x_scale=[.1*L,.1*L,max(best[3],1.)], max_nfev=80)
    return {'source_xy':[float(fit.x[0]),float(fit.x[1])], 'strength_Q':float(fit.x[2])}

