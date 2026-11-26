"""Finite-volume Robin continuation and ridge-profiled sensor inversion."""
import numpy as np
from scipy import linalg, sparse


class _Grid:
    def __init__(self, shape, lengths, gamma):
        self.shape = shape
        self.n = int(np.prod(shape))
        self.h = np.asarray(lengths) / shape
        self.vol = float(np.prod(self.h))
        self.gamma = gamma
        cells = np.arange(self.n).reshape(shape)
        self.faces = []
        for axis, h in enumerate(self.h):
            views = np.moveaxis(cells, axis, 0)
            self.faces.append((views[:-1].ravel(), views[1:].ravel(),
                               views[0].ravel(), views[-1].ravel(), h))
        self.ends = np.concatenate([face[1] for face in self.faces])
        self.starts = np.concatenate([face[0] for face in self.faces])
        ix = cells.ravel()
        self.rows = np.concatenate((ix, self.starts, self.ends))
        self.cols = np.concatenate((ix, self.ends, self.starts))

    def matrix(self, kappa, sigma):
        kappa = kappa.ravel()
        diagonal = self.vol * sigma.ravel()
        weights = []
        for c, d, low, high, h in self.faces:
            area = self.vol / h
            kc, kd = kappa[c], kappa[d]
            w = (area / h) * (2.0 * kc * kd / (kc + kd))
            diagonal[c] += w
            diagonal[d] += w
            for wall in (low, high):
                kw = kappa[wall]
                diagonal[wall] += area * (self.gamma * kw) / (0.5 * h * self.gamma + kw)
            weights.append(w)
        off = -np.concatenate(weights)
        return sparse.csr_matrix((np.concatenate((diagonal, off, off)),
                                  (self.rows, self.cols)), shape=(self.n, self.n))


class _Multigrid:
    def __init__(self, a, shape, transfers=None):
        self.levels = []
        self.transfers = []
        while a.shape[0] > 90:
            # A row-sum bound on the diagonally scaled spectral radius keeps
            # weighted Jacobi stable on all Galerkin levels.
            d = a.diagonal()
            bound = np.max(np.asarray(abs(a).sum(axis=1)).ravel() / d)
            jac = (1.35 / bound) / d
            if transfers is None:
                coarse = tuple((n + 1) // 2 for n in shape)
                xyz = np.indices(shape).reshape(3, -1) // 2
                group = np.ravel_multi_index(xyz, coarse)
                n = len(group)
                t = sparse.csr_matrix((np.ones(n), (np.arange(n), group)),
                                      shape=(n, int(np.prod(coarse))))
                p = t - (a @ t).multiply(jac[:, None]).tocsr()
                r = p.T.tocsr()
                shape = coarse
            else:
                p, r = transfers[len(self.levels)]
            self.levels.append((a, jac, p, r))
            self.transfers.append((p, r))
            a = (r @ (a @ p)).tocsr()
        dense = a.toarray()
        self.coarse = linalg.cho_factor((dense + dense.T) * 0.5, check_finite=False)

    def apply(self, b, level=0):
        if level == len(self.levels):
            return linalg.cho_solve(self.coarse, b, check_finite=False)
        a, jac, p, r = self.levels[level]
        x = jac * b
        residual = b - a @ x
        x += jac * residual
        residual = b - a @ x
        x += p @ self.apply(r @ residual, level + 1)
        residual = b - a @ x
        x += jac * residual
        residual = b - a @ x
        x += jac * residual
        return x


def _dot(x, y):
    return np.einsum('i,i->', x, y)


def _pcg(a, b, x, mg, maxiter, rtol=3e-10):
    target = _dot(b, b) * rtol**2
    if not target:
        return np.zeros_like(b), True
    x = x.copy()
    r = b - a @ x
    fresh = True
    for _ in range(maxiter):
        if _dot(r, r) <= target:
            r = b - a @ x
            if _dot(r, r) <= target:
                return x, True
            fresh = True
        z = mg.apply(r)
        rho = _dot(r, z)
        p = z if fresh else z + (rho / old_rho) * p
        fresh = False
        ap = a @ p
        step = rho / _dot(p, ap)
        x += step * p
        r -= step * ap
        old_rho = rho
    r = b - a @ x
    return x, _dot(r, r) <= target


def _profile(z, y, s, lam_z, lam_s):
    # Square-root action of the ridge projector, in orthogonal coordinates.
    # Complement and shrunk range are kept separate to avoid subtracting
    # nearly equal energies when a sensor mode aligns with nuisance columns.
    if z.shape[1] == 0:
        sw, yw = s, y
    else:
        u, sv, _ = linalg.svd(z, full_matrices=False, check_finite=False)
        s_range = u.T @ s
        y_range = u.T @ y
        scale = np.sqrt(lam_z) / np.hypot(sv, np.sqrt(lam_z))
        sw = np.concatenate((s - u @ s_range, scale[:, None] * s_range))
        yw = np.concatenate((y - u @ y_range, scale[:, None] * y_range))
    gram = sw.T @ sw
    cross = sw.T @ yw
    amplitudes = cross / gram.diagonal()[:, None]
    gram.flat[::gram.shape[0] + 1] += lam_s
    weights = linalg.solve(gram, cross, assume_a='pos', check_finite=False)
    return amplitudes, weights


def solve_robin_continuation_tomography(study: dict) -> dict:
    k0, k1, k2, s0, s1 = (np.asarray(study[key], dtype=float) for key in
                         ('kappa_base', 'kappa_mod1', 'kappa_mod2', 'sigma_base', 'sigma_mod'))
    grid = _Grid(k0.shape, [study['Lx'], study['Ly'], study['Lz']], study['gamma_robin'])
    theta = np.asarray(study['thetas'], dtype=float)
    rhs = np.asarray(study['source_modes'], dtype=float).reshape(-1, grid.n) * grid.vol
    sensors = np.ravel_multi_index(np.asarray(study['sensor_ijk']).T, grid.shape)
    stack = np.empty((len(theta), len(sensors), len(rhs)))

    def operator(t):
        return grid.matrix(k0 + t * k1 + (t * t) * k2, s0 + t * s1)

    ref = operator(np.sqrt(theta.min() * theta.max()))
    mg = _Multigrid(ref, grid.shape)
    x = np.zeros_like(rhs)
    terminal = None
    for k in np.argsort(theta):
        a = operator(theta[k])
        local = None
        for m, b in enumerate(rhs):
            x[m], ok = _pcg(a, b, x[m], mg if local is None else local, 80)
            if not ok:
                if local is None:
                    local = _Multigrid(a, grid.shape, mg.transfers)
                x[m], ok = _pcg(a, b, x[m], local, 300)
                if not ok:
                    # Preserve the residual contract in especially poorly
                    # conditioned coefficient arrangements.
                    x[m] = sparse.linalg.spsolve(a.tocsc(), b)
        stack[k] = x[:, sensors].T
        if k == len(theta) - 1:
            terminal = x.copy().reshape((-1,) + grid.shape)
    stack = stack.reshape(-1, len(rhs))
    amplitudes, weights = _profile(np.asarray(study['Z_nuisance'], dtype=float),
                                    np.asarray(study['Y_obs'], dtype=float), stack,
                                    study['nuisance_ridge'], study['source_ridge'])
    return dict(terminal_fields=terminal, continuation_sensor_predictions=stack,
                profiled_amplitudes=amplitudes, joint_source_weights=weights)
