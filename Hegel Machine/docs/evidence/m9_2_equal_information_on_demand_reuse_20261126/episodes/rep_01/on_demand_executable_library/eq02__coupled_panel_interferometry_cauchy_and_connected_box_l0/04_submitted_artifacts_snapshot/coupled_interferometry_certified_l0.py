"""Reference-panel calibration and certified box-constrained sparse regression."""
import heapq
import numpy as np


def _panel(panel):
    labels = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    raw = np.asarray(panel['X_panel_raw'], dtype=np.float64)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    z = np.empty_like(raw)
    balance = np.empty(len(raw))
    for label in np.unique(labels):
        rows = labels == label
        c = controls[rows]
        c = c[:, np.sum(np.isfinite(c), axis=0) >= 2]
        offset = np.median(np.nanmean(c, axis=0))
        gain = np.median(np.nanstd(c, axis=0))
        z[rows] = (raw[rows] - offset) / gain
        for s in (0, 1, 2):
            cell = rows & (strata == s)
            balance[cell] = len(raw) / (15 * np.count_nonzero(cell))
    finite = np.isfinite(z)
    denom = balance @ finite
    fill = np.divide(balance @ np.where(finite, z, 0.0), denom,
                     out=np.zeros(z.shape[1]), where=denom > 0)
    return np.where(finite, z, fill)


def _calibrate(h, y):
    def fit(weight):
        sqrt = np.sqrt(weight)
        return np.linalg.lstsq(h * sqrt[:, None], y * sqrt, rcond=None)[0]

    theta = fit(np.ones(len(y)))
    scale = 0.0
    for step in range(100):
        residual = y - np.einsum('ij,j->i', h, theta)
        if step < 16:
            mad = np.median(np.abs(residual - np.median(residual)))
            scale = max(1.8 * mad, 1e-10)
        update = fit(1.0 / (1.0 + (residual / scale)**2))
        converged = np.linalg.norm(update - theta) <= 1e-10 * (1 + np.linalg.norm(theta))
        theta = update
        if converged and step >= 16:
            break
    return theta


def _selection(x, y, ridge, l0, box):
    p = x.shape[1]
    gram = x.T @ x
    xy = x.T @ y
    q = gram.copy()
    ix = np.diag_indices(p)
    q[ix] += ridge
    norms = np.sqrt(np.maximum(np.diag(gram), np.finfo(float).tiny))
    ev = np.linalg.eigvalsh(gram / norms[:, None] / norms[None, :])[0]
    d = max(0.0, ev * 0.995 - 1e-12) * norms**2
    h = gram.copy()
    h[ix] -= d
    a = d + ridge
    hd = np.diag(h)
    qd = np.diag(q)
    if l0 == 0:
        t = np.zeros(p)
        slope = np.zeros(p)
    elif box > 0:
        t = np.minimum(box, np.sqrt(np.divide(2*l0, a, out=np.full(p, np.inf), where=a > 0)))
        slope = l0/t + 0.5*a*t
    else:
        t = np.zeros(p)
        slope = np.zeros(p)
    free_order = np.argsort(-np.abs(xy) / np.sqrt(np.maximum(qd, 1e-300)))
    pad = 1e-12 * (1 + y @ y + box**2 * np.sum(np.abs(gram)))
    best_w = np.zeros(p)
    best = float(0.5 * (y @ y))

    def record(w):
        nonlocal best, best_w
        w = np.where(np.abs(w) > 1e-8, w, 0.0)
        r = x @ w - y
        f = float(0.5 * (r @ r + ridge * (w @ w)) + l0 * np.count_nonzero(w))
        if f < best:
            best, best_w = f, w.copy()

    def polish(w, support):
        j = np.flatnonzero(support)
        result = np.zeros(p)
        if j.size:
            sub = q[np.ix_(j, j)]
            try:
                v = np.linalg.solve(sub, xy[j])
            except np.linalg.LinAlgError:
                v = np.linalg.lstsq(sub, xy[j], rcond=None)[0]
            if np.max(np.abs(v)) > box:
                v = np.clip(v, -box, box)
                r = sub @ v - xy[j]
                for _ in range(350):
                    change = 0.
                    for k in range(len(j)):
                        if sub[k, k] <= 0:
                            continue
                        z = min(box, max(-box, v[k] - r[k]/sub[k, k]))
                        delta = z - v[k]
                        if delta:
                            v[k] = z
                            r += sub[:, k] * delta
                            change = max(change, abs(delta))
                    if change < 1e-10:
                        break
            result[j] = v
        record(result)
        return result

    def sparse_descent(w):
        w = w.copy()
        g = q @ w - xy
        for _ in range(100):
            change = 0.0
            for j in free_order:
                if qd[j] <= 0:
                    continue
                v = g[j] - qd[j]*w[j]
                z = min(box, max(-box, -v/qd[j]))
                if 0.5*qd[j]*z*z + v*z + l0 >= 0 or abs(z) <= 1e-8:
                    z = 0.0
                delta = z - w[j]
                if delta:
                    w[j] = z
                    g += q[:, j]*delta
                    change = max(change, abs(delta))
            if change < 1e-9:
                break
        record(w)
        return w

    def bound(state, center):
        # Tangent to the PSD remainder of the least squares Hessian;
        # minimize the resulting separate box quadratics exactly.
        residual = x @ center - y
        v = x.T @ residual - d * center
        u = np.minimum(box, np.divide(np.abs(v), a, out=np.full(p, box), where=a > 0))
        benefit = u * (np.abs(v) - 0.5*a*u)
        deduction = np.where(state == 1, benefit-l0,
                             np.where(state == 0, np.maximum(benefit-l0, 0.), 0.))
        return float(0.5*(residual @ residual + d @ (center*center)) -
                     (x.T @ residual) @ center - deduction.sum() - pad)

    def relax(state, warm):
        w = warm.copy()
        w[state == -1] = 0.0
        if not box:
            return w, bound(state, w)
        order = [int(j) for j in free_order if state[j] != -1 and qd[j] > 0]
        g = h @ w - xy
        free = state == 0
        for step in range(220):
            change = 0.0
            for j in order:
                v = g[j] - hd[j]*w[j]
                mag = abs(v)
                if free[j] and l0 > 0 and mag <= slope[j] + hd[j]*t[j]:
                    z = 0.0 if mag <= slope[j] else min(t[j], (mag - slope[j])/hd[j])
                else:
                    z = min(box, mag / qd[j])
                z = -z if v > 0 else z
                delta = z - w[j]
                if delta:
                    w[j] = z
                    g += h[:, j]*delta
                    change = max(change, abs(delta))
            if change < 1e-9:
                break
            if step in (3, 11, 31, 79) and bound(state, w) >= best - 1e-7:
                break
        return w, bound(state, w)

    state = np.zeros(p, dtype=np.int8)
    sparse_descent(best_w)
    center, lower = relax(state, best_w)
    sparse_descent(center)
    polish(center, np.abs(center) >= 0.5*t if l0 > 0 and box > 0 else np.ones(p, dtype=bool))
    pending = [(lower, 0, state, center)]
    popped = 0
    leaves = []
    dlist = d.tolist()
    while pending:
        lb, _, state, center = heapq.heappop(pending)
        free = np.flatnonzero(state == 0)
        if lb < best - 1e-7:
            support = (state == 1) | ((state == 0) & (np.abs(center) >= 0.5*t) & (np.abs(center) > 1e-8))
            polish(center, support)
        if lb >= best - 1e-7 or not free.size:
            leaves.append({'fixed_zero': (state == -1).tolist(),
                           'fixed_active': (state == 1).tolist(), 'lb': lb,
                           'minorant_diagonal': dlist, 'minorant_center': center.tolist()})
            continue
        if box and l0:
            ratio = np.minimum(1.0, np.abs(center[free])/t[free])
            score = np.minimum(ratio, 1.0-ratio)
            j = int(free[np.argmax(score)])
        else:
            j = int(free[0])
        for sign in (-1, 1):
            child = state.copy()
            child[j] = sign
            z, low = relax(child, center)
            popped += 1
            heapq.heappush(pending, (low, popped, child, z))
    return {'weights': best_w.tolist(), 'objective': best,
            'certified_lower_bound': min(best, min(leaf['lb'] for leaf in leaves)),
            'subcube_leaves': leaves}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    aligned = _panel(study['panel'])
    strata = np.asarray(study['panel']['stratum'])
    optics = study['interferometry']
    signal = np.asarray(optics['signal_phasor'], dtype=np.complex128)
    reference = np.asarray(optics['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(reference))
    design = np.column_stack((np.ones(len(strata)), strata == 1, strata == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    s3 = np.asarray(study['stage3_indices'])
    theta = _calibrate(design[cal], y[cal])
    target = y[s3] - np.einsum('ij,j->i', design[s3], theta)
    cert = _selection(np.asarray(study['X_cand'], dtype=np.float64), target,
                      float(study['lam2']), float(study['lam0']), float(study['M']))
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': target, 'certified_selection': cert}
