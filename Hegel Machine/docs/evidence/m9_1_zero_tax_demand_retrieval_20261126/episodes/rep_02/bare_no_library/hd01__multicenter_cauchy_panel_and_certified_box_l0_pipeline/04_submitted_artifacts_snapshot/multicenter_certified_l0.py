"""Reference-panel alignment, robust potency fitting, and certified subset selection."""

import heapq

import numpy as np


def _align(cohort):
    raw = np.asarray(cohort['X_raw'], dtype=float)
    controls = np.asarray(cohort['controls'], dtype=float)
    centers = np.asarray(cohort['center_id'])
    strata = np.asarray(cohort['stratum'])
    aligned = np.empty_like(raw)
    for center in np.unique(centers):
        rows = centers == center
        panel = controls[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        panel = np.where(np.isfinite(panel), panel, np.nan)
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        aligned[rows] = (raw[rows] - offset) / gain
    _, cell, counts = np.unique(np.column_stack((centers, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = (len(cell) / len(counts)) / counts[cell]
    observed = np.isfinite(aligned)
    col_means = np.where(observed, aligned, 0).T @ weights / (observed.T @ weights)
    return np.where(observed, aligned, col_means), strata


def _potency(aligned, strata, y, scale_factor=1.0):
    design = np.column_stack((np.ones(len(strata)), strata == 1, strata == 2, aligned))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    for iteration in range(90):
        residual = y - design @ coef
        if iteration < 12:
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if iteration == 11:
                scale *= scale_factor
        root_weight = 1 / np.hypot(1, residual / max(scale, 1e-8))
        updated = np.linalg.lstsq(design * root_weight[:, None], y * root_weight, rcond=None)[0]
        if iteration >= 12 and np.max(np.abs(updated - coef)) < 1e-9:
            return updated[3:]
        coef = updated
    return coef[3:]


def _select(problem):
    X = np.asarray(problem['X'], dtype=float)
    y = np.asarray(problem['y'], dtype=float)
    l2, l0, box = (float(problem[k]) for k in ('lam2', 'lam0', 'M'))
    gram = X.T @ X
    h = X.T @ y
    p = len(h)
    ridge = gram + l2 * np.eye(p)
    c0 = 0.5 * (y @ y)
    eps = 1e-8
    zero_box = min(eps, box)
    # The PSD remainder supplies a global tangent, valid even on a subcube
    # where some coordinates are forced to zero or charged a fixed fee.
    eigmin = np.linalg.eigvalsh(gram)[0]
    d = np.full(p, max(0.0, eigmin - 1e-11 * np.max(np.diag(gram))))
    B = gram - np.diag(d)
    # Extract additional diagonal curvature in correlation coordinates.
    # A small-norm column need not limit the curvature of every column.
    diagonal = B.diagonal().copy()
    positive = diagonal > 0
    if positive.any():
        scales = np.sqrt(diagonal[positive])
        correlation = (B[np.ix_(positive, positive)] / scales[:, None]) / scales
        extra = max(0.0, np.linalg.eigvalsh(correlation)[0] - 1e-11)
        d[positive] += extra * diagonal[positive]
        B = gram - np.diag(d)
    a = d + l2
    knee = np.minimum(box, zero_box + np.sqrt(2 * l0 / a))
    slope = np.full(p, np.inf)
    if box > zero_box:
        slope = l0 / (knee - zero_box) + 0.5 * a * (knee + zero_box)
    movable = B.diagonal().clip(min=0)
    best_w = np.zeros(p)
    best_obj = c0
    seen_supports = set()

    def objective(w):
        r = X @ w - y
        return float(0.5 * (r @ r + l2 * (w @ w)) + l0 * np.count_nonzero(np.abs(w) > eps))

    def incumbent(w):
        nonlocal best_w, best_obj
        value = objective(w)
        if value < best_obj:
            best_w, best_obj = w.copy(), value

    def fit_support(mask, start):
        key = mask.tobytes()
        if key in seen_supports:
            return
        seen_supports.add(key)
        # An off coordinate still has a small, unpenalized interval. An
        # active coordinate is charged even when its quadratic minimizer
        # falls in that interval; evaluating the actual objective can only
        # improve on that charged value.
        limits = np.where(mask, box, zero_box)
        w = np.clip(start, -limits, limits)
        grad = ridge @ w - h
        for _ in range(140):
            max_step = 0.0
            for j in range(p):
                t = min(limits[j], max(-limits[j], w[j] - grad[j] / ridge[j, j]))
                delta = t - w[j]
                if delta:
                    w[j] = t
                    grad += delta * ridge[:, j]
                    max_step = max(max_step, abs(delta) * np.sqrt(ridge[j, j]))
            if max_step < 1e-10:
                break
        incumbent(w)

    def polish(w):
        w = w.copy()
        grad = ridge @ w - h
        for _ in range(35):
            previous = w.copy()
            for j in range(p):
                diag = ridge[j, j]
                linear = grad[j] - diag * w[j]
                t = np.clip(-linear / diag, -box, box)
                off = np.clip(-linear / diag, -zero_box, zero_box)
                if (0.5 * diag * t + linear) * t + l0 >= (0.5 * diag * off + linear) * off:
                    t = off
                delta = t - w[j]
                if delta:
                    w[j] = t
                    grad += delta * ridge[:, j]
            if np.max(np.abs(w - previous)) < 1e-8:
                break
        incumbent(w)
        fit_support(np.abs(w) > eps, w)

    def relaxation(state, start):
        fixed_zero = state == 0
        active = state == 1
        free = state < 0
        c = start.copy()
        c[fixed_zero] = np.clip(c[fixed_zero], -zero_box, zero_box)
        grad = ridge @ c - h
        # Exact cyclic coordinate updates for the convex envelope of each
        # undecided coordinate's diagonal quadratic plus L0 charge.
        for _ in range(100):
            max_delta = 0.0
            for j in range(p):
                diagonal = ridge[j, j]
                linear = grad[j] - diagonal * c[j]
                if fixed_zero[j] or box == zero_box:
                    t = min(zero_box, max(-zero_box, -linear / diagonal))
                elif free[j] and diagonal * zero_box < abs(linear) <= slope[j] + movable[j] * knee[j]:
                    mag = zero_box
                    if movable[j] > 0:
                        mag = min(knee[j], max(zero_box, (abs(linear) - slope[j]) / movable[j]))
                    t = np.copysign(mag, -linear)
                else:
                    t = min(box, max(-box, -linear / diagonal))
                delta = t - c[j]
                if delta:
                    c[j] = t
                    grad += delta * ridge[:, j]
                    max_delta = max(max_delta, abs(delta) * np.sqrt(diagonal))
            if max_delta < 2e-9:
                break
        bc = B @ c
        v = bc - h
        minimizer = np.clip(-v / a, -box, box)
        nonzero = (0.5 * a * minimizer + v) * minimizer + l0
        # Account for the objective's numerical zero band as well as zero itself.
        near_zero = np.clip(-v / a, -min(eps, box), min(eps, box))
        zero_cost = (0.5 * a * near_zero + v) * near_zero
        bound = c0 - 0.5 * (c @ bc) + nonzero[active].sum()
        bound += zero_cost[fixed_zero].sum() + np.minimum(zero_cost, nonzero)[free].sum()
        bound -= 1e-12 * (1 + c0 + abs(bound))
        return float(bound), c

    root = np.full(p, -1, dtype=np.int8)
    ridge_w = np.clip(np.linalg.solve(ridge, h), -box, box)
    polish(ridge_w)
    polish(best_w)
    pending = []
    leaves = []
    serial = 0
    tolerance = 8e-6

    def add(state, start):
        nonlocal serial
        lb, center = relaxation(state, start)
        serial += 1
        heapq.heappush(pending, (lb, serial, state, center))

    add(root, ridge_w)
    while pending:
        lb, _, state, center = heapq.heappop(pending)
        undecided = np.flatnonzero(state < 0)
        incumbent(center)
        if lb < best_obj - tolerance:
            fit_support((state == 1) | ((state < 0) & (np.abs(center) > 0.5 * (zero_box + knee))), center)
            if lb < best_obj - tolerance and len(undecided):
                mag = np.abs(center[undecided])
                gap = np.zeros(len(undecided))
                fractional = (mag > zero_box) & (mag < knee[undecided])
                idx = undecided[fractional]
                mf = mag[fractional]
                gap[fractional] = l0 + 0.5 * a[idx] * (mf**2 - zero_box**2) - slope[idx] * (mf - zero_box)
                j = undecided[np.argmax(gap)]
                for decision in (0, 1):
                    child = state.copy()
                    child[j] = decision
                    add(child, center)
                continue
        leaves.append({'fixed_zero': np.flatnonzero(state == 0).tolist(),
                       'fixed_active': np.flatnonzero(state == 1).tolist(),
                       'lb': lb, 'minorant_diagonal': d.copy(), 'minorant_center': center})
    return {'weights': best_w, 'objective': best_obj,
            'certified_lower_bound': min(best_obj, min(leaf['lb'] for leaf in leaves)),
            'subcube_leaves': leaves}


def solve_multicenter_certified_l0(study: dict) -> dict:
    cohort = study['cohort']
    aligned, strata = _align(cohort)
    return {'aligned_biomarkers': aligned,
            'robust_potency_weights': _potency(aligned, strata, np.asarray(cohort['y_potency'])),
            'certified_selection': _select(study['sparse_selection'])}
