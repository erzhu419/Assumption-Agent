"""Reference-panel alignment, robust potency, and a sparse box certificate."""

import heapq
from itertools import count

import numpy as np


def _cohort_fit(cohort):
    x = np.asarray(cohort['X_raw'], dtype=float)
    q = np.asarray(cohort['controls'], dtype=float)
    centers = np.asarray(cohort['center_id'])
    strata = np.asarray(cohort['stratum'])
    y = np.asarray(cohort['y_potency'], dtype=float)
    z = np.empty_like(x)
    for center in np.unique(centers):
        rows = centers == center
        panel = q[rows]
        valid = np.isfinite(panel)
        keep = valid.sum(axis=0) >= 2
        panel = np.where(valid[:, keep], panel[:, keep], np.nan)
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        if not gain > 0:
            raise ValueError('A center has no usable scanner gain')
        z[rows] = (x[rows] - offset) / gain
    _, cells, sizes = np.unique(np.column_stack((centers, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    cell_weights = (len(x) / len(sizes)) / sizes[cells]
    finite = np.isfinite(z)
    observed = np.where(finite, z, 0.0)
    fill = (cell_weights @ observed) / (cell_weights @ finite)
    z = np.where(finite, z, fill)
    design = np.column_stack((np.ones(len(x)), strata == 1, strata == 2, z))

    def weighted_fit(weights):
        root = np.sqrt(weights)
        return np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]

    beta = weighted_fit(cell_weights)
    residual = y - design @ beta
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    scale = max(scale, np.finfo(float).eps * max(1., np.max(np.abs(y))))
    for _ in range(100):
        residual = y - design @ beta
        update = weighted_fit(cell_weights / (1. + (residual / scale)**2))
        delta = np.max(np.abs(update - beta))
        beta = update
        if delta <= 1e-10 * (1. + np.max(np.abs(beta))):
            break
    return z, beta[3:]


def _sparse_fit(problem):
    x = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    l0, l2, box = (float(problem[key]) for key in ('lam0', 'lam2', 'M'))
    p = x.shape[1]
    gram = x.T @ x
    h = x.T @ y
    diag = np.diag(gram)
    norm = max(1., np.max(np.sum(np.abs(gram), axis=1)))
    margin = 64 * np.finfo(float).eps * max(x.shape) * norm
    gersh = 2 * diag - np.abs(gram).sum(axis=1)
    if np.min(gersh) >= margin:
        d = gersh - margin
    else:
        d = np.full(p, max(0., np.linalg.eigvalsh(gram)[0] - margin))
    hess = gram - np.diag(d)
    ridge_hess = gram + l2 * np.eye(p)
    curvature = d + l2
    cutoff = min(box, 1e-8)
    span = np.minimum(box - cutoff, np.sqrt(2 * l0 / curvature))
    transition = cutoff + span
    slope = curvature * (cutoff + 0.5 * span)
    slope += np.divide(l0, span, out=np.zeros_like(span), where=span > 0)
    misc = max(1., y @ y + np.abs(h).sum() * box + (norm + l2) * p * box**2 + p*l0)
    arithmetic_pad = 64 * np.finfo(float).eps * max(x.shape) * misc
    def minimize(zero, active, start, envelope):
        """Minimize the convex envelope, or the smooth incumbent subproblem."""
        smooth = hess if envelope else ridge_hess
        limits = np.where(zero, cutoff, box)
        w = np.clip(start, -limits, limits)
        gradient = smooth @ w - h
        for _ in range(600):
            movement = 0.
            for j in range(p):
                old = w[j]
                a = smooth[j, j]
                linear = a * old - gradient[j]
                if envelope:
                    denom = a + curvature[j]
                    magnitude = abs(linear)
                    value = magnitude / denom
                    if not active[j] and not zero[j] and span[j] > 0:
                        if magnitude > denom * cutoff:
                            if magnitude <= a * cutoff + slope[j]:
                                value = cutoff
                            elif magnitude <= a * transition[j] + slope[j]:
                                value = min(transition[j], (magnitude - slope[j]) / a) if a > 0 else cutoff
                else:
                    value = abs(linear) / a
                new = np.copysign(min(limits[j], value), linear)
                step = new - old
                if step:
                    gradient += step * smooth[:, j]
                    w[j] = new
                    movement = max(movement, abs(step))
            if movement <= 2e-12 * box:
                break
        return w

    def witness(center, zero, active):
        residual = y - x @ center
        gradient = x.T @ residual
        z = np.abs(gradient + d * center)
        v = np.minimum(box, z / curvature)
        on = l0 - v * (z - 0.5 * curvature * v)
        # The numerical support cutoff permits an inactive coefficient in
        # [-1e-8, 1e-8]. Minimize its separable term on that interval.
        v_off = np.minimum(v, cutoff)
        off = -v_off * (z - 0.5 * curvature * v_off)
        free = ~(zero | active)
        lb = (0.5 * (residual @ residual + (d * center) @ center)
              + gradient @ center + off[zero].sum() + on[active].sum()
              + np.minimum(off[free], on[free]).sum())
        lb = float(lb - arithmetic_pad)
        return lb, on - off

    def objective(w):
        residual = x @ w - y
        return float(0.5 * (residual @ residual + l2 * (w @ w))
                     + l0 * np.count_nonzero(np.abs(w) > 1e-8))

    zeros = np.zeros(p, dtype=bool)
    serial = count()
    pending, leaves = [], []
    best_w = np.zeros(p)
    upper = objective(best_w)
    tol = 9.8e-6

    def add(zero, active, start):
        nonlocal upper, best_w
        center = minimize(zero, active, start, True)
        lb, on = witness(center, zero, active)
        if lb < upper - tol:
            free = ~(zero | active)
            for support in (active | (free & (on < 0.)),
                            active | (free & (np.abs(center) > cutoff + 0.5 * span))):
                candidate = minimize(~support, support, center, False)
                val = objective(candidate)
                if val < upper:
                    best_w, upper = candidate, val
        heapq.heappush(pending, (lb, next(serial), zero, active, center, on))

    add(zeros, zeros, np.zeros(p))
    while pending:
        lb, _, zero, active, center, on = heapq.heappop(pending)
        if lb >= upper - tol:
            leaves.append({'fixed_zero': zero, 'fixed_active': active, 'lb': lb,
                           'minorant_diagonal': d.copy(), 'minorant_center': center})
            continue
        free = ~(zero | active)
        if not np.any(free):
            raise RuntimeError('Box subproblem did not reach certificate tolerance')
        if len(pending) + len(leaves) + 2 > 2048:
            raise RuntimeError('Subcube budget exceeded')
        # Disjunction violation: distance from the two minimizers of the
        # free separable term, zero and its optimal nonzero coefficient.
        ratio = np.divide(np.abs(center) - cutoff, span, out=np.zeros_like(span), where=span > 0)
        ratio = np.clip(ratio, 0., 1.)
        priority = np.minimum(ratio, np.maximum(0., 1. - ratio))
        priority += 0.02 * l0 / (l0 + np.abs(on))
        j = np.argmax(np.where(free, priority, -np.inf))
        zchild, achild = zero.copy(), active.copy()
        zchild[j], achild[j] = True, True
        add(zchild, active, center)
        add(zero, achild, center)
    return {'weights': best_w, 'objective': upper,
            'certified_lower_bound': min(leaf['lb'] for leaf in leaves),
            'subcube_leaves': leaves}


def solve_multicenter_certified_l0(study: dict) -> dict:
    aligned, potency = _cohort_fit(study['cohort'])
    return {'aligned_biomarkers': aligned, 'robust_potency_weights': potency,
            'certified_selection': _sparse_fit(study['sparse_selection'])}
