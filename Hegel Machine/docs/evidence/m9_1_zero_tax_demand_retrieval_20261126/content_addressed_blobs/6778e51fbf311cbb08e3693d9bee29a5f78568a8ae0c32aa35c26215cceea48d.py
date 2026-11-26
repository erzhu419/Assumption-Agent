"""Reference-panel alignment, robust potency fitting, and certified box-L0 ridge."""
import numpy as np


def _cohort_fit(cohort):
    centers = np.asarray(cohort['center_id'])
    strata = np.asarray(cohort['stratum'])
    raw = np.asarray(cohort['X_raw'], dtype=float)
    controls = np.asarray(cohort['controls'], dtype=float)
    y = np.asarray(cohort['y_potency'], dtype=float)
    z = np.empty_like(raw)
    w = np.empty(len(y))
    for c in np.unique(centers):
        rows = centers == c
        panel = controls[rows]
        cols = [col[np.isfinite(col)] for col in panel.T]
        cols = [col for col in cols if len(col) >= 2]
        offset = np.median([col.mean() for col in cols])
        gain = np.median([col.std() for col in cols])
        z[rows] = (raw[rows] - offset) / gain
        for s in np.unique(strata[rows]):
            cell = rows & (strata == s)
            w[cell] = 1.0 / cell.sum()
    w /= w.mean()
    finite = np.isfinite(z)
    means = (w @ np.where(finite, z, 0.0)) / (w @ finite)
    z = np.where(finite, z, means)
    design = np.column_stack((np.ones(len(y)), strata == 1, strata == 2, z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for _ in range(100):
        r = y - design @ beta
        scale = max(1e-8, 1.4826 * np.median(np.abs(r - np.median(r))))
        # Cauchy weights; pass square roots to weighted least squares.
        root = 1.0 / np.hypot(1.0, r / (2.0 * scale))
        nxt = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        if np.max(np.abs(nxt - beta)) < 1e-9:
            beta = nxt
            break
        beta = nxt
    return z, beta[3:]


def _sparse_fit(spec):
    X = np.asarray(spec['X'], dtype=float)
    y = np.asarray(spec['y'], dtype=float)
    ridge, cost, M = (float(spec[k]) for k in ('lam2', 'lam0', 'M'))
    p = X.shape[1]
    G = X.T @ X
    b = X.T @ y
    H = G + ridge * np.eye(p)
    d = np.full(p, max(0.0, np.linalg.eigvalsh(G)[0] - 1e-11 * max(1.0, np.linalg.norm(G, np.inf))))
    R = G - np.diag(d)
    a = d + ridge
    t = np.minimum(M, np.sqrt(2 * cost / a))
    k = cost / t + 0.5 * a * t
    rd = np.diag(R)
    hd = np.diag(H)
    slack = 2e-12 * (1 + y @ y + 2 * M * np.sum(abs(b)) + M*M*np.sum(abs(H)) + p * cost)

    def objective(w):
        r = X @ w - y
        return float(0.5 * (r @ r + ridge * (w @ w)) + cost * np.count_nonzero(abs(w) > 1e-8))

    def box_fit(mask, start):
        ids = np.flatnonzero(mask)
        w = np.zeros(p)
        if len(ids):
            u = np.linalg.solve(H[np.ix_(ids, ids)], b[ids])
            w[ids] = np.clip(u, -M, M)
            if np.any(abs(u) > M):
                for _ in range(250):
                    change = 0.
                    for j in ids:
                        v = np.clip(w[j] + (b[j] - H[j] @ w) / hd[j], -M, M)
                        change = max(change, abs(v - w[j]))
                        w[j] = v
                    if change < 1e-12:
                        break
        w[abs(w) <= 1e-8] = 0.
        return w

    def lower(z, status):
        # Expand the loss around z and discard the PSD quadratic in R.
        r = X @ z - y
        g = X.T @ r - d * z
        c = 0.5 * (r @ r + d @ (z*z)) - z @ (X.T @ r)
        u = np.clip(-g / a, -M, M)
        active = 0.5 * a * u*u + g*u + cost
        # The objective counts |w| > 1e-8, so an inactive coordinate can
        # still move inside this interval without paying the activation cost.
        v = np.clip(-g / a, -min(M, 1e-8), min(M, 1e-8))
        inactive = 0.5 * a * v*v + g*v
        parts = np.where(status == 1, active,
                         np.where(status == -1, inactive, np.minimum(inactive, active)))
        return float(c + parts.sum() - slack)

    def relax(status, start):
        # The undecided coordinate's convex envelope is linear from 0 to t,
        # then quadratic. Fixed coordinates retain their exact box quadratic.
        z = start.copy()
        z[status == -1] = 0.
        ids = np.flatnonzero(status != -1)
        for _ in range(120):
            change = 0.
            for j in ids:
                h = b[j] - R[j] @ z + rd[j]*z[j]
                if status[j] == 1:
                    v = min(M, max(-M, h/hd[j]))
                else:
                    h0 = abs(h)
                    if h0 <= k[j]:
                        v = 0.
                    elif h0 <= k[j] + rd[j]*t[j] or t[j] == M:
                        v = min(t[j], (h0-k[j])/rd[j]) if rd[j] > 0 else t[j]
                    else:
                        v = min(M, h0/hd[j])
                    if h < 0:
                        v = -v
                change = max(change, abs(v-z[j]))
                z[j] = v
            if change < 2e-10:
                break
        return z, lower(z, status)

    best = np.zeros(p)
    upper = objective(best)
    leaves = []
    stop = 2e-7
    stack = [(np.zeros(p, dtype=np.int8), best)]
    while stack:
        status, start = stack.pop()
        z, lb = relax(status, start)
        free = status == 0
        if lb < upper - stop:
            mask = (status == 1) | (free & (abs(z) >= t * 0.5))
            w = box_fit(mask, z)
            value = objective(w)
            if value < upper:
                upper, best = value, w
        if lb >= upper - stop or not np.any(free):
            leaves.append({'fixed_zero': status == -1, 'fixed_active': status == 1,
                           'lb': lb, 'minorant_diagonal': d.copy(), 'minorant_center': z})
            continue
        frac = abs(z) / t
        score = np.minimum(frac, 1-frac)
        j = np.argmax(np.where(free, score, -np.inf))
        first = 1 if frac[j] >= 0.5 else -1
        for v in (-first, first):
            child = status.copy()
            child[j] = v
            stack.append((child, z))
    return {'weights': best, 'objective': upper,
            'certified_lower_bound': min(upper, min(e['lb'] for e in leaves)),
            'subcube_leaves': leaves}


def solve_multicenter_certified_l0(study: dict) -> dict:
    aligned, potency = _cohort_fit(study['cohort'])
    return {'aligned_biomarkers': aligned, 'robust_potency_weights': potency,
            'certified_selection': _sparse_fit(study['sparse_selection'])}
