"""Cell-centered Robin diffusion solves and nuisance-profiled source fitting."""

def __hegel_load_skill_autogenous_prim_conservative_spd_solve():
    import types as __hegel_types
    import numpy as np
    import numpy as np
    from scipy import linalg, sparse
    from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg

    def _aggregate(n, row, col, weights, rng):
        priority = rng.permutation(n)
        labels = np.full(n, -1, dtype=np.intp)
        count = 0
        while np.any(labels < 0):
            high = priority.copy()
            np.maximum.at(high, row, priority[col])
            roots = np.flatnonzero((labels < 0) & (high == priority))
            labels[roots] = count + np.arange(roots.size)
            count += roots.size
            attach = (labels[row] < 0) & (labels[col] >= 0)
            (i, j, w) = (row[attach], col[attach], weights[attach])
            best = np.zeros(n)
            np.maximum.at(best, i, w)
            tied = w == best[i]
            chosen = np.full(n, count, dtype=np.intp)
            np.minimum.at(chosen, i[tied], labels[j[tied]])
            found = chosen < count
            labels[found] = chosen[found]
            keep = (labels[row] < 0) & (labels[col] < 0)
            (row, col, weights) = (row[keep], col[keep], weights[keep])
        return (labels, count)

    def galerkin_preconditioner(operator: 'sparse.spmatrix | sparse.sparray', *, strength_cutoff: 'float'=0.08, max_coarse_fraction: 'float'=0.76, coarse_size: 'int'=64, seed: 'int'=12) -> 'LinearOperator':
        """Map a real sparse SPD matrix of shape (n, n) to a symmetric (n, n) V-cycle approximate inverse using smoothed aggregates and Galerkin coarse operators."""
        a = sparse.csr_matrix(operator, dtype=float)
        n0 = a.shape[0]
        if not n0 or a.shape[1] != n0 or coarse_size < 1:
            raise ValueError('expected a nonempty square operator and positive coarse_size')
        if not 0 <= strength_cutoff <= 1 or not 0 < max_coarse_fraction < 1:
            raise ValueError('invalid aggregation thresholds')
        (levels, rng) = ([], np.random.default_rng(seed))
        while a.shape[0] > coarse_size:
            (n, d) = (a.shape[0], a.diagonal())
            coo = a.tocoo()
            off = (coo.row != coo.col) & (coo.data != 0)
            (i, j, v) = (coo.row[off], coo.col[off], coo.data[off])
            strength = np.abs(v) / np.sqrt(d[i]) / np.sqrt(d[j])
            strong = strength >= strength_cutoff
            (labels, nc) = _aggregate(n, i[strong], j[strong], strength[strong], rng)
            if nc > max_coarse_fraction * n:
                (labels, nc) = _aggregate(n, i, j, strength, rng)
            if nc == n:
                bottom = lambda r: r / d
                break
            bound = np.max(np.asarray(abs(a).sum(axis=1)).ravel() / d)
            inv = 2.0 / 3.0 / bound / d
            t = sparse.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, nc))
            p = (t - (a @ t).multiply(inv[:, None])).tocsr()
            pt = p.T.tocsr()
            levels.append((a, p, pt, inv))
            ac = pt @ (a @ p)
            a = (0.5 * ac + 0.5 * ac.T).tocsr()
        else:
            factor = linalg.cho_factor(a.toarray())
            bottom = lambda r: linalg.cho_solve(factor, r)

        def cycle(r, level=0):
            if level == len(levels):
                return bottom(r)
            (a_level, p, pt, inv) = levels[level]
            z = inv * r
            z += p @ cycle(pt @ (r - a_level @ z), level + 1)
            return z + inv * (r - a_level @ z)

        def apply(r):
            return cycle(np.asarray(r).reshape(n0))
        return LinearOperator((n0, n0), matvec=apply, rmatvec=apply, dtype=float)

    def solve_residual_checked_cg(operator: 'np.ndarray | sparse.spmatrix | sparse.sparray | LinearOperator', rhs: 'np.ndarray', *, preconditioner: 'LinearOperator | None'=None, tol: 'float'=1e-08, maxiter: 'int'=450, refinements: 'int'=4) -> 'tuple[np.ndarray, float]':
        """Solve a real SPD operator of shape (n, n) against rhs of shape (n,), returning the float64 solution (n,) and its directly recomputed relative 2-norm residual."""
        (a, b) = (aslinearoperator(operator), np.asarray(rhs, dtype=float))
        if b.ndim != 1 or not b.size or a.shape != (b.size, b.size):
            raise ValueError('expected a square operator and matching nonempty vector')
        if not np.all(np.isfinite(b)) or not 0 < tol < 1 or maxiter < 1 or (refinements < 0):
            raise ValueError('invalid right-hand side or solver controls')
        if preconditioner is not None and preconditioner.shape != a.shape:
            raise ValueError('preconditioner shape must match the operator')
        x = np.zeros_like(b)
        bscale = np.max(np.abs(b))
        if bscale == 0:
            return (x, 0.0)
        bnorm = linalg.norm(b / bscale)
        for attempt in range(refinements + 1):
            r = b - a @ x
            rscale = np.max(np.abs(r))
            if not np.isfinite(rscale):
                break
            if rscale == 0:
                return (x, 0.0)
            unit_r = r / rscale
            rel = float(rscale / bscale * (linalg.norm(unit_r) / bnorm))
            if rel <= tol:
                return (x, rel)
            if attempt == refinements:
                break
            inner_tol = min(0.01, 0.85 * tol / rel)
            (step, _) = cg(a, unit_r, M=preconditioner, rtol=inner_tol, atol=0.0, maxiter=maxiter)
            if not np.all(np.isfinite(step)):
                break
            x += rscale * step
        r = b - a @ x
        if np.all(np.isfinite(r)):
            rel = float(linalg.norm(r / bscale, check_finite=False) / bnorm)
            if rel <= tol:
                return (x, rel)
        raise RuntimeError('CG did not meet the directly recomputed residual tolerance')
    return __hegel_types.SimpleNamespace(galerkin_preconditioner=galerkin_preconditioner, solve_residual_checked_cg=solve_residual_checked_cg)
__hegel_skill_ns_autogenous_prim_conservative_spd_solve = __hegel_load_skill_autogenous_prim_conservative_spd_solve()
import numpy as np
from scipy import sparse
galerkin_preconditioner = __hegel_skill_ns_autogenous_prim_conservative_spd_solve.galerkin_preconditioner
solve_residual_checked_cg = __hegel_skill_ns_autogenous_prim_conservative_spd_solve.solve_residual_checked_cg

def _robin_operator(kappa, sigma, gamma, hx, hy):
    """Assemble each interior face as one symmetric flux contribution."""
    cells = np.arange(kappa.size).reshape(kappa.shape)
    fx = 2 * hy / hx * kappa[:-1, :] * kappa[1:, :] / (kappa[:-1, :] + kappa[1:, :])
    fy = 2 * hx / hy * kappa[:, :-1] * kappa[:, 1:] / (kappa[:, :-1] + kappa[:, 1:])
    diag = hx * hy * np.broadcast_to(sigma, kappa.shape).copy()
    diag[:-1, :] += fx
    diag[1:, :] += fx
    diag[:, :-1] += fy
    diag[:, 1:] += fy
    for i in (0, -1):
        kb = kappa[i, :]
        diag[i, :] += hy * gamma * kb / (0.5 * hx * gamma + kb)
    for j in (0, -1):
        kb = kappa[:, j]
        diag[:, j] += hx * gamma * kb / (0.5 * hy * gamma + kb)
    left = np.concatenate((cells[:-1, :].ravel(), cells[:, :-1].ravel()))
    right = np.concatenate((cells[1:, :].ravel(), cells[:, 1:].ravel()))
    face = np.concatenate((fx.ravel(), fy.ravel()))
    return sparse.csr_matrix((np.concatenate((diag.ravel(), -face, -face)), (np.concatenate((cells.ravel(), left, right)), np.concatenate((cells.ravel(), right, left)))), shape=(kappa.size, kappa.size))

def _profile(S, Z, Y, lam_z, lam_s):
    p = Z.shape[1]
    za = np.vstack((Z, np.sqrt(lam_z) * np.eye(p)))
    data = np.pad(np.column_stack((S, Y)), ((0, p), (0, 0)))
    coefficients = np.linalg.lstsq(za, data, rcond=None)[0]
    residual = data - za @ coefficients
    m = S.shape[1]
    (sr, yr) = (residual[:, :m], residual[:, m:])
    energy = np.einsum('ij,ij->j', sr, sr)
    dots = sr.T @ yr
    single = np.divide(dots, energy[:, None], out=np.zeros_like(dots), where=energy[:, None] > 0)
    joint = np.linalg.lstsq(np.vstack((sr, np.sqrt(lam_s) * np.eye(m))), np.pad(yr, ((0, m), (0, 0))), rcond=None)[0]
    return (single, joint)

def solve_robin_spd_and_profile_sources(problem: dict) -> dict:
    """Return the converged source fields and both sets of fitted amplitudes."""
    kappa = np.asarray(problem['kappa'], dtype=float)
    (nx, ny) = kappa.shape
    (hx, hy) = (problem['Lx'] / nx, problem['Ly'] / ny)
    A = _robin_operator(kappa, np.asarray(problem['sigma'], dtype=float), problem['gamma_robin'], hx, hy)
    preconditioner = galerkin_preconditioner(A)
    sources = np.asarray(problem['candidate_sources'], dtype=float)
    fields = np.empty_like(sources)
    for (index, q) in enumerate(sources):
        (field, _) = solve_residual_checked_cg(A, hx * hy * q.ravel(), preconditioner=preconditioner, tol=2e-10, maxiter=600)
        fields[index] = field.reshape(kappa.shape)
    ij = np.asarray(problem['sensor_ij'])
    S = fields[:, ij[:, 0], ij[:, 1]].T
    (single, joint) = _profile(S, np.asarray(problem['nuisance_design'], dtype=float), np.asarray(problem['Y_obs'], dtype=float), problem['nuisance_ridge'], problem['source_ridge'])
    return {'basis_fields': fields, 'single_source_profiled_amplitudes': single, 'joint_source_weights': joint}
