"""Reference-panel alignment, robust potency regression, and box L0 certificates."""

import numpy as np


def _align(cohort):
    _, center = np.unique(cohort['center_id'], return_inverse=True)
    stratum = np.asarray(cohort['stratum'])
    controls = np.asarray(cohort['controls'], dtype=float)
    z = np.array(cohort['X_raw'], dtype=float, copy=True)
    for i in range(center.max() + 1):
        rows = center == i
        panel = controls[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        panel = np.where(np.isfinite(panel), panel, np.nan)
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        z[rows] = (z[rows] - offset) / gain
    cell = 3 * center + stratum
    weight = 1.0 / np.bincount(cell)[cell]
    weight /= weight.mean()
    finite = np.isfinite(z)
    means = (weight @ np.where(finite, z, 0.0)) / (weight @ finite)
    return np.where(np.isnan(z), means, z)


def _potency(z, stratum, y):
    stratum = np.asarray(stratum)
    y = np.asarray(y, dtype=float)
    design = np.column_stack((np.ones(len(y)), stratum == 1, stratum == 2, z))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    mad = np.median(np.abs(residual - np.median(residual)))
    scale = max(mad, np.finfo(float).eps * max(1.0, np.max(np.abs(y))))
    for _ in range(200):
        root_weight = 1 / np.hypot(1, (y - design @ beta) / scale)
        new_beta = np.linalg.lstsq(root_weight[:, None] * design, root_weight * y,
                                  rcond=None)[0]
        change = np.linalg.norm(new_beta - beta)
        beta = new_beta
        if change <= 1e-10 * (1 + np.linalg.norm(beta)):
            break
    return beta[3:]


def _sparse(data):
    x, y = (np.asarray(data[k], dtype=float) for k in ('X', 'y'))
    ridge, charge, box = (float(data[k]) for k in ('lam2', 'lam0', 'M'))
    p = x.shape[1]
    gram = x.T @ x
    xy = x.T @ y
    h = gram + ridge * np.eye(p)
    # Keep the PSD check well away from cancellation at the smallest eigenvalue.
    eig = np.linalg.eigvalsh(gram)
    d = np.full(p, max(0.0, eig[0] - 1e-10 * max(1.0, eig[-1])))
    r = gram - np.diag(d)
    q = d + ridge
    knee = np.minimum(box, np.sqrt(2 * charge / q))
    slope = charge / knee + q * knee / 2
    lipschitz = max(eig[-1] - d[0], 1e-12)
    zero_eps = 1e-8
    # The open active interval has the same infimum as its closure.
    active_floor = min(box, zero_eps)

    def objective(w):
        residual = x @ w - y
        return float((residual @ residual + ridge * (w @ w)) / 2
                     + charge * np.count_nonzero(np.abs(w) > zero_eps))

    def fit_support(mask, start):
        """Polish the ridge solution on a fixed support under the box."""
        out = np.zeros(p)
        ind = np.flatnonzero(mask)
        if not len(ind):
            return out
        hh = h[np.ix_(ind, ind)]
        b = xy[ind]
        w = np.linalg.solve(hh, b)
        if np.max(np.abs(w)) > box:
            w = np.clip(start[ind], -box, box)
            for _ in range(500):
                prev = w.copy()
                for j in range(len(w)):
                    w[j] = np.clip(w[j] + (b[j] - hh[j] @ w) / hh[j, j], -box, box)
                if np.max(np.abs(w - prev)) <= 2e-13 * (1 + box):
                    break
        out[ind] = w
        out[np.abs(out) <= zero_eps] = 0.0
        return out

    best_w = np.zeros(p)
    best = objective(best_w)

    def improve(w):
        nonlocal best_w, best
        val = objective(w)
        if val < best:
            best_w, best = w.copy(), val

    def witness(v, state):
        """Tangent to the PSD remainder plus exact separable infima."""
        residual = x @ v - y
        g = x.T @ residual
        linear = g - d * v
        a = np.clip(np.abs(linear) / q, active_floor, box)
        on = q * a**2 / 2 - abs(linear) * a + charge
        # Allow the tiny zero-cost band specified by the objective. This also
        # yields a valid (possibly smaller) bound for exactly-zero supports.
        z = np.minimum(np.abs(linear) / q, active_floor)
        off = q * z**2 / 2 - abs(linear) * z
        terms = np.where(state == 0, off, np.where(state == 1, on, np.minimum(off, on)))
        const = residual @ residual / 2 - g @ v + (d * v) @ v / 2
        margin = 5e-13 * (1 + abs(const) + np.sum(np.abs(terms)))
        return float(const + terms.sum() - margin), on - off

    def relax(state, start, steps=450):
        """Optimize the separable convex envelope to choose a tangent point."""
        free = state < 0
        v = np.where(state == 0, 0, start)
        extrap = v.copy()
        t = 1.0
        for _ in range(steps):
            u = extrap - (r @ extrap - xy) / lipschitz
            mag = np.abs(u)
            # Prox of the envelope: linear, kink, then quadratic.
            low = np.clip(mag - slope / lipschitz, 0.0, knee)
            high = mag / (1 + q / lipschitz)
            val = np.where(free, np.where(high > knee, high, low), high)
            new = np.where(state == 0, 0.0, np.sign(u) * np.minimum(val, box))
            if np.max(np.abs(new - v)) < 5e-11 * (1 + box):
                return new
            nt = (1 + np.sqrt(1 + 4*t*t)) / 2
            if (extrap - new) @ (new - v) > 0:
                extrap, t = new, 1.0
            else:
                extrap, t = new + (t - 1) / nt * (new - v), nt
            v = new
        return v

    def local_search(w, allowed):
        """Find a good incumbent by exact one-coordinate box L0 updates."""
        w = w.copy()
        for _ in range(120):
            prev = w.copy()
            for j in allowed:
                b = xy[j] - h[j] @ w + h[j, j] * w[j]
                val = np.clip(b / h[j, j], -box, box)
                w[j] = (val if abs(val) > zero_eps and
                        b * val - h[j, j] * val**2 / 2 > charge else 0.0)
            if np.max(np.abs(w - prev)) <= 1e-11 * (1 + box):
                break
        return fit_support(w != 0, w)

    root = np.full(p, -1, dtype=np.int8)
    initial = fit_support(np.ones(p, dtype=bool), np.zeros(p))
    for start in (initial, best_w):
        improve(local_search(start, np.argsort(-np.abs(xy))))

    leaves = []
    stack = [(root, best_w.copy())]
    tolerance = 5e-6
    while stack:
        state, start = stack.pop()
        free = np.flatnonzero(state < 0)
        fixed = state == 1
        # Try the incumbent tangent first; many sibling subcubes prune here.
        v = best_w
        lb, delta = witness(v, state)
        if lb < best - tolerance:
            relaxed = relax(state, start) if len(free) else fit_support(fixed, start)
            alt, alt_delta = witness(relaxed, state)
            if alt > lb:
                v, lb, delta = relaxed, alt, alt_delta
            # Rounded envelope solutions and then polished support fits.
            mask = fixed | ((state < 0) & (np.abs(relaxed) >= knee / 2))
            candidate = fit_support(mask, relaxed)
            improve(candidate)
            alt, alt_delta = witness(candidate, state)
            if alt > lb:
                v, lb, delta = candidate, alt, alt_delta
        if lb >= best - tolerance or not len(free):
            leaves.append({'fixed_zero': np.flatnonzero(state == 0).tolist(),
                           'fixed_active': np.flatnonzero(fixed).tolist(),
                           'lb': lb, 'minorant_diagonal': d.copy(),
                           'minorant_center': v.copy()})
            continue
        fraction = np.abs(relaxed[free]) / knee[free]
        fractional = np.minimum(fraction, 1 - fraction)
        # Strong branching: estimate the weaker child's bound after adjusting
        # the tangent rather than judging only by the current fractional value.
        ranked = sorted(range(len(free)), key=lambda k: (-max(0., fractional[k]), abs(delta[free[k]])))
        options = ranked[:6] if len(free) > 5 and p >= 19 else ranked[:1]
        chosen = None
        best_score = (-np.inf, -np.inf)
        for k in options:
            j = free[k]
            children = []
            estimates = []
            for bit in (0, 1):
                child = state.copy()
                child[j] = bit
                lb0, _ = witness(v, child)
                trial = relaxed
                if lb0 < best - tolerance and len(options) > 1:
                    trial = relax(child, relaxed, steps=25)
                    lb0 = max(lb0, witness(trial, child)[0])
                estimates.append(lb0)
                children.append((child, trial))
            clipped = np.minimum(estimates, best)
            score = (min(clipped) + .25 * max(clipped), min(clipped))
            if score > best_score:
                chosen, best_score = children, score
                preferred = int(estimates[1] < estimates[0])
            if min(estimates) >= best - tolerance:
                break
        stack.extend((chosen[1-preferred], chosen[preferred]))

    return {'weights': best_w, 'objective': best,
            'certified_lower_bound': min(best, min(leaf['lb'] for leaf in leaves)),
            'subcube_leaves': leaves}


def solve_multicenter_certified_l0(study: dict) -> dict:
    """Solve the three stages of a multicenter study."""
    cohort = study['cohort']
    z = _align(cohort)
    return {'aligned_biomarkers': z,
            'robust_potency_weights': _potency(z, cohort['stratum'], cohort['y_potency']),
            'certified_selection': _sparse(study['sparse_selection'])}
