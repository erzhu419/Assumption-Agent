"""Dual-channel transport inversion with a box-simplex KKT certificate."""

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu


def _transport_axis(length, kappa, velocity, robin):
    """Cell-centered diffusion, upwind advection, and Robin boundary flux."""
    h = length / kappa.size
    left, right = kappa[:-1], kappa[1:]
    face = (2 * left * right / (left + right)) / h**2
    pos = np.maximum(velocity, 0.0) / h
    neg = np.maximum(-velocity, 0.0) / h
    lower = -face - pos[1:]
    upper = -face - neg[:-1]
    diag = pos + neg
    diag[:-1] += face
    diag[1:] += face
    edge = kappa[[0, -1]]
    diag[[0, -1]] += (robin / h) / (1 + robin * h / (2 * edge))
    return lower, diag, upper


def _spectral_axis(axis):
    lower, diag, upper = axis
    # A = D Q diag(evals) Q.T D^-1 for the nonsymmetric upwind stencil.
    log_d = np.r_[0., np.cumsum(0.5 * (np.log(-lower) - np.log(-upper)))]
    d = np.exp(log_d - np.mean(log_d))
    evals, q = eigh_tridiagonal(diag, -np.sqrt(lower * upper))
    return evals, d[:, None] * q, q.T / d


def _responses(s):
    src = np.asarray(s['source_modes'], dtype=float)
    m, nx, ny = src.shape
    axes = [_transport_axis(s['L' + a], np.asarray(s['kappa_' + a], dtype=float),
                            np.asarray(s['vel_' + a], dtype=float), s['robin_' + a])
            for a in 'xy']
    lx, rx, ix = _spectral_axis(axes[0])
    ly, ry, iy = _spectral_axis(axes[1])
    modal = ix @ src @ iy.T
    eigsum = lx[:, None] + ly
    ax, by = [diags(a, [-1, 0, 1], format='csc') for a in axes]
    base = kron(ax, eye(ny), format='csc') + kron(eye(nx), by, format='csc')
    potential = np.asarray(s['V_pocket_channel_b'], dtype=float).ravel()
    rhs = src.reshape(m, -1).T
    i, j = np.asarray(s['sensor_ij']).T
    reads_a, reads_b = [], []
    for mu in s['mu_shifts']:
        ua = rx @ (modal / (eigsum + mu)) @ ry.T
        op = base + diags(potential + mu, format='csc')
        ub = splu(op).solve(rhs).T.reshape(m, nx, ny)
        reads_a.append(ua[:, i, j].T)
        reads_b.append(ub[:, i, j].T)
    return np.concatenate(reads_a + reads_b), ua, ub


def _profile(z, xy, ridge):
    """Apply the benchmark's ridge-residual map to both design and data."""
    u, sigma, _ = np.linalg.svd(z, full_matrices=False)
    if ridge == 0:
        u = u[:, sigma > sigma.max(initial=0.) * max(z.shape) * np.finfo(float).eps]
        return xy - u @ (u.T @ xy)
    scale = sigma**2 / (sigma**2 + ridge)
    return xy - u @ (scale[:, None] * (u.T @ xy))


def _box_simplex(S, y, ridge, cap, budget):
    n = S.shape[1]
    H = S.T @ S + ridge * np.eye(n)
    c = S.T @ y
    if not 0 <= budget <= cap * n:
        raise ValueError('Infeasible box-simplex budget')
    w = np.full(n, budget / n)
    if budget == 0 or budget == cap * n:
        w.fill(0.0 if budget == 0 else cap)
        g = S.T @ (S @ w - y) + ridge * w
        return w, float(g.min() if budget == 0 else g.max())

    # 0 denotes a free coordinate; -1 and +1 denote fixed bounds.
    active = np.zeros(n, dtype=int)
    for _ in range(20 * n * n):
        f = np.flatnonzero(active == 0)
        if not f.size:
            g = S.T @ (S @ w - y) + ridge * w
            lo, hi = np.flatnonzero(active < 0), np.flatnonzero(active > 0)
            i, j = lo[np.argmin(g[lo])], hi[np.argmax(g[hi])]
            if g[j] <= g[i] + 1e-12:
                return w, float((g[i] + g[j]) / 2)
            active[[i, j]] = 0
            continue

        fixed = np.flatnonzero(active)
        # Eliminate the equality by writing the last free coordinate as
        # remaining budget minus the sum of the other free coordinates.
        target = w.copy()
        rem = budget - cap * np.count_nonzero(active > 0)
        p = f[-1]
        a = f[:-1]
        target[p] = rem
        if a.size:
            target[a] = 0.0
            grad = H @ target - c
            reduced = H[np.ix_(a, a)] - H[a, p, None] - H[p, a] + H[p, p]
            rhs = grad[p] - grad[a]
            try:
                target[a] = np.linalg.solve(reduced, rhs)
            except np.linalg.LinAlgError:
                target[a] = np.linalg.lstsq(reduced, rhs, rcond=None)[0]
            target[p] = rem - target[a].sum()

        # Move along the feasible face only as far as its first blocking bound.
        direction = target - w
        bad_low = f[target[f] < -1e-13]
        bad_high = f[target[f] > cap + 1e-13]
        if bad_low.size or bad_high.size:
            steps = np.r_[-w[bad_low] / direction[bad_low],
                          (cap - w[bad_high]) / direction[bad_high]]
            hit = int(np.argmin(steps))
            j = np.r_[bad_low, bad_high][hit]
            w = np.clip(w + max(0.0, steps[hit]) * direction, 0.0, cap)
            active[j] = -1 if hit < bad_low.size else 1
            w[j] = 0.0 if active[j] < 0 else cap
            continue

        w = np.clip(target, 0.0, cap)
        g = S.T @ (S @ w - y) + ridge * w
        nu = float(g[f].mean())
        violations = (g[fixed] - nu) * active[fixed]
        if not fixed.size or violations.max() <= 1e-12:
            return w, nu
        active[fixed[np.argmax(violations)]] = 0
    raise RuntimeError('Box-simplex active set did not converge')


def solve_coupled_channel_vortex_inversion(study: dict) -> dict:
    """Fit shared source weights and reconstruct both final-shift fields."""
    s = study
    raw, ua, ub = _responses(s)
    prof = _profile(np.asarray(s['Z_nuisance'], dtype=float),
                    np.column_stack((raw, np.asarray(s['y_obs'], dtype=float).ravel())),
                    float(s['nuisance_ridge']))
    S, y = prof[:, :-1], prof[:, -1]
    ridge, cap, budget = map(float, (s['source_ridge'], s['u_box'], s['total_budget']))
    w, nu = _box_simplex(S, y, ridge, cap, budget)
    res = S @ w - y
    j = float(0.5 * (res @ res + ridge * (w @ w)))
    g = S.T @ res + ridge * w
    ll, lu = np.maximum(g - nu, 0.0), np.maximum(nu - g, 0.0)
    # Complementarity terms give the tangent lower bound without subtracting
    # large, nearly equal terms at a certified optimum.
    gap = float(ll @ w + lu @ (cap - w) + nu * (w.sum() - budget))
    cert = dict(weights=w.tolist(), nu_sum=nu, lam_lower=ll.tolist(), lam_upper=lu.tolist(),
                primal_objective=j, dual_lower_bound=min(j, j - gap))
    return dict(weights=w, terminal_field_channel_a=np.einsum('m,mij->ij', w, ua),
                terminal_field_channel_b=np.einsum('m,mij->ij', w, ub), kkt_certificate=cert)
