"""Reference-panel alignment, robust potency regression and boxed L0 proof."""

import numpy as np


def _align(cohort):
    raw = np.asarray(cohort['X_raw'], dtype=float)
    controls = np.asarray(cohort['controls'], dtype=float)
    group = np.asarray(cohort['center_id'])
    stratum = np.asarray(cohort['stratum'])
    aligned = np.empty_like(raw)
    for center in np.unique(group):
        rows = group == center
        panel = controls[rows]
        panel = np.where(np.isfinite(panel), panel, np.nan)
        valid = (np.isfinite(panel).sum(axis=0) >= 2)
        panel = panel[:, valid]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        if not gain > 0:
            raise ValueError('Each reference panel must have positive estimated gain')
        aligned[rows] = (raw[rows] - offset) / gain
    _, cell, counts = np.unique(np.column_stack((group, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = 1.0 / counts[cell.ravel()]
    weights /= weights.mean()
    good = np.isfinite(aligned)
    total = (weights[:, None] * good).sum(axis=0)
    numerator = weights @ np.where(good, aligned, 0.0)
    fill = np.divide(numerator, total, out=np.zeros_like(numerator), where=total > 0)
    return np.where(good, aligned, fill)


def _potency(aligned, stratum, y):
    stratum = np.asarray(stratum)
    design = np.column_stack((np.ones(len(y)), stratum == 1, stratum == 2, aligned))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    # Continuation starts with a broad Cauchy loss, then reduces glitch leverage.
    for scale in (1.0, 0.35, 0.2):
        for _ in range(60):
            weight = 1 / np.hypot(1, (y - design @ beta) / scale)
            update = np.linalg.lstsq(design * weight[:, None], y * weight, rcond=None)[0]
            change = np.linalg.norm(update - beta)
            beta = update
            if change <= 1e-9 * (1 + np.linalg.norm(beta)):
                break
    return beta[3:]


def _certify(problem):
    x = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    l0, l2, box = (float(problem[k]) for k in ('lam0', 'lam2', 'M'))
    p = x.shape[1]
    G = x.T @ x
    b = x.T @ y
    # Certify a uniform diagonal by a reconstruction of a Cholesky factor,
    # then use diagonal dominance of the reconstruction error as a guard.
    scale = max(1., np.linalg.norm(G, ord=np.inf))
    margin = 1e-11 * scale
    floor = max(0., np.linalg.eigvalsh(G)[0] - margin)
    if floor:
        L = np.linalg.cholesky(G - np.eye(p) * (floor + 0.5 * margin))
        if np.linalg.norm(G - np.eye(p) * (floor + 0.5 * margin) - L @ L.T, ord=np.inf) > 0.25 * margin:
            floor = 0.
    d = np.full(p, floor)
    a = d + l2
    H = G.copy()
    H.flat[::p+1] += l2
    diag = H.diagonal().copy()
    q = np.maximum(diag - a, 0.)
    knee = np.minimum(np.sqrt(2 * l0 / a), box)
    slope = l0 / knee + 0.5 * a * knee
    sign_guard = 5e-13 * (1 + y @ y + (np.sum(abs(G)) + l2 * p) * box**2 + np.sum(abs(b)) * box + p * l0)
    near_zero = min(1e-8, box)
    best = np.zeros(p)
    upper = float(0.5 * y @ y)
    leaves = []

    def score(w):
        res = y - x @ w
        return float(0.5 * (res @ res + l2 * (w @ w)) + l0 * np.count_nonzero(abs(w) > 1e-8))

    def improve(w, state):
        nonlocal upper, best
        w = w.copy()
        g = b - H @ w
        for _ in range(125):
            delta = 0.
            for j in np.flatnonzero(state != -1):
                t = g[j] + diag[j] * w[j]
                v = np.clip(t / diag[j], -box, box)
                if state[j] == 0 and v * (t - 0.5 * diag[j] * v) < l0:
                    v = 0.
                step = v - w[j]
                if step:
                    g -= step * H[:, j]
                    w[j] = v
                    delta = max(delta, abs(step))
            if delta <= 1e-10:
                break
        val = score(w)
        if val < upper:
            upper, best = val, w

    def relax(w, state):
        w = w.copy()
        w[state == -1] = 0.
        g = b - H @ w
        for _ in range(160):
            delta = 0.
            for j in np.flatnonzero(state != -1):
                t = g[j] + diag[j] * w[j]
                s = abs(t)
                if state[j] == 1:
                    v = min(s / diag[j], box)
                elif s <= slope[j]:
                    v = 0.
                elif s < slope[j] + q[j] * knee[j]:
                    v = (s - slope[j]) / q[j]
                else:
                    v = min(max(knee[j], s / diag[j]), box)
                v = np.copysign(v, t)
                step = v - w[j]
                if step:
                    g -= step * H[:, j]
                    w[j] = v
                    delta = max(delta, abs(step))
            if delta <= 1e-10:
                break
        return w

    def bound(w, state):
        res = y - x @ w
        gradient = x.T @ res
        z = abs(gradient + d * w)
        v = np.minimum(z / a, box)
        on = l0 - v * (z - 0.5 * a * v)
        val = 0.5 * (res @ res + (d * w) @ w) + gradient @ w
        # Inactive coefficients are uncharged throughout the threshold interval.
        # Minimize the same tangent quadratic on that interval rather than
        # subtracting a global error estimate independent of the witness.
        u = np.minimum(v, near_zero)
        off = -u * (z - 0.5 * a * u)
        free = state == 0
        val += (on[state == 1].sum() + off[state == -1].sum()
                + np.minimum(on[free], off[free]).sum())
        return float(val - sign_guard), on - off

    root = np.zeros(p, dtype=np.int8)
    improve(best, root)
    stack = [(root, best)]
    while stack:
        state, warm = stack.pop()
        w = relax(warm, state)
        low, on = bound(w, state)
        if upper - low > 9.9e-6:
            improve(w, state)
        if upper - low <= 9.9e-6:
            leaves.append({'fixed_zero': np.flatnonzero(state == -1).tolist(),
                           'fixed_active': np.flatnonzero(state == 1).tolist(),
                           'lb': low, 'minorant_diagonal': d.copy(), 'minorant_center': w})
            continue
        free = np.flatnonzero(state == 0)
        if free.size == 0:
            raise RuntimeError('Fully fixed box solve did not close the certificate gap')
        ratio = abs(w[free]) / knee[free]
        gap = np.where((ratio > 1e-8) & (ratio < 1 - 1e-8),
                       (1 - ratio) * (l0 - 0.5 * a[free] * knee[free]**2 * ratio), 0.)
        j = free[np.argmax(gap)] if np.any(gap > 0) else free[np.argmin(abs(on[free]))]
        first = 1 if abs(w[j]) >= knee[j] / 2 else -1
        for mode in (-first, first):
            child = state.copy()
            child[j] = mode
            stack.append((child, w))
        if len(stack) + len(leaves) > 2048:
            raise RuntimeError('Certificate exceeded the subcube budget')
    return {'weights': best, 'objective': upper,
            'certified_lower_bound': min(leaf['lb'] for leaf in leaves),
            'subcube_leaves': leaves}


def solve_multicenter_certified_l0(study: dict) -> dict:
    """Solve the three stages; sparse proof leaves include their tangents."""
    cohort = study['cohort']
    aligned = _align(cohort)
    beta = _potency(aligned, cohort['stratum'], np.asarray(cohort['y_potency'], dtype=float))
    return {'aligned_biomarkers': aligned, 'robust_potency_weights': beta,
            'certified_selection': _certify(study['sparse_selection'])}
