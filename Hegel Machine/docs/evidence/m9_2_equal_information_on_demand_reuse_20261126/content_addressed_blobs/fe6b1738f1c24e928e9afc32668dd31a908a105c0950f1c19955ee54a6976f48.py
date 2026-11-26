"""Panel calibration and globally bounded connected box-L0 selection."""
import heapq
import numpy as np
from scipy.optimize import minimize


def _align(panel):
    group = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    raw = np.asarray(panel['X_panel_raw'], dtype=np.float64)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    aligned = np.empty_like(raw)
    weight = np.empty(len(group))
    for center in np.unique(group):
        rows = group == center
        ref = controls[rows]
        ref = ref[:, (np.isfinite(ref).sum(axis=0) >= 2)]
        offset = np.median(np.nanmean(ref, axis=0))
        gain = np.median(np.nanstd(ref, axis=0))
        aligned[rows] = (raw[rows] - offset) / gain
        for s in (0, 1, 2):
            cell = rows & (strata == s)
            weight[cell] = 1.0 / cell.sum()
    weight /= weight.mean()
    known = np.isfinite(aligned)
    fill = np.nansum(aligned * weight[:, None], axis=0) / (weight @ known)
    return np.where(known, aligned, fill)


def _calibrate(h, y):
    theta = np.linalg.lstsq(h, y, rcond=None)[0]
    for multiplier in (2.5, 1.5):
        residual = y - np.einsum('ij,j->i', h, theta)
        scale = max(1e-8, multiplier * 1.4826 * np.median(np.abs(residual - np.median(residual))))
        for _ in range(75):
            residual = y - np.einsum('ij,j->i', h, theta)
            root = 1 / np.hypot(1, residual / scale)
            nxt = np.linalg.lstsq(h * root[:, None], y * root, rcond=None)[0]
            change = np.linalg.norm(nxt - theta)
            theta = nxt
            if change <= 1e-10 * (1 + np.linalg.norm(theta)):
                break
    return theta


def _select(x, y, l2, l0, m):
    p = x.shape[1]
    g = x.T @ x
    b = x.T @ y
    yy = 0.5 * (y @ y)
    jj = np.arange(p)
    qmat = g + np.diag(np.full(p, l2))
    norms = np.sqrt(np.maximum(np.diag(g), np.finfo(float).tiny))
    eig = np.linalg.eigvalsh(g / norms[:, None] / norms[None, :])[0]
    d = max(0., eig - 2e-12) * np.diag(g)
    r = g - np.diag(d)
    k = d + l2
    rd = np.diag(r)
    qdiag = rd + k
    t = np.full(p, m)
    np.sqrt(np.divide(2*l0, k, out=np.full(p, np.inf), where=k > 0), out=t)
    t = np.minimum(m, t)
    slope = np.divide(l0, t, out=np.zeros(p), where=t > 0) + 0.5*k*t
    best = yy
    weights = np.zeros(p)
    cache = set()

    def offer(support, initial):
        nonlocal best, weights
        key = support.tobytes()
        if key in cache or l0 * support.sum() >= best:
            return
        cache.add(key)
        idx = jj[support]
        if not len(idx):
            return
        a = qmat[np.ix_(idx, idx)]
        v = b[idx]
        try:
            w = np.linalg.solve(a, v)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(a, v, rcond=None)[0]
        if np.any(np.abs(w) > m):
            def fun(z):
                az = a @ z
                return (0.5*az-v) @ z, az-v
            w = minimize(fun, np.clip(w, -m, m), jac=True, method='L-BFGS-B',
                         bounds=[(-m,m)]*len(idx), options={'ftol': 1e-15, 'gtol': 1e-10}).x
        w[np.abs(w) <= 1e-8] = 0.
        full = np.zeros(p)
        full[idx] = w
        err = x @ full - y
        obj = float(0.5*(err @ err + l2*(w @ w)) + l0*np.count_nonzero(w))
        if obj < best:
            best, weights = obj, full

    def relax(zero, active, start):
        free = ~(zero | active)
        ids = jj[~zero]
        c = np.where(zero, 0., start).copy()
        rc = r @ c
        for sweep in range(110):
            change = 0.
            for j in ids:
                z = b[j] - rc[j] + rd[j]*c[j]
                az = abs(z)
                if free[j] and az <= slope[j]:
                    u = 0.
                elif free[j] and az <= slope[j] + rd[j]*t[j] and rd[j] > 0:
                    u = (az-slope[j])/rd[j]
                else:
                    u = min(m, az/qdiag[j]) if qdiag[j] > 0 else m
                u = np.copysign(u, z)
                delta = u-c[j]
                if delta:
                    rc += delta*r[j]
                    c[j] = u
                    change = max(change, abs(delta))
            if change < 2e-9:
                break
        rc = r @ c
        z = np.abs(b-rc)
        u = np.minimum(m, np.divide(z, k, out=np.full(p,m), where=k > 0))
        on = l0 - u*(z - 0.5*k*u)
        const = yy - 0.5*(c @ rc)
        lb = const + on[active].sum() + np.minimum(on[free],0.).sum()
        # A downward rounding margin for the float64 Gram representation.
        lb -= 4e-13 * (1 + abs(yy) + np.abs(c) @ (np.abs(g) @ np.abs(c)) + p*l0 + m*np.abs(b).sum())
        frac = np.divide(np.abs(c), t, out=np.zeros(p), where=t > 0)
        return float(lb), c, frac, on

    leaves = []
    pending = []
    serial = 0
    tol = 8e-6

    def add(zero, active, start):
        nonlocal serial
        lb, c, frac, on = relax(zero, active, start)
        free = ~(zero | active)
        if lb < best-tol or not pending and not leaves:
            offer(active | (free & (frac >= 0.5)), c)
            offer(active | (free & (on < 0)), c)
        if lb >= best-tol or not free.any():
            leaves.append((zero, active, lb, c))
        else:
            score = np.minimum(frac, 1-frac)
            j = jj[free][np.argmax(score[free] - 1e-12*np.abs(on[free]))]
            serial += 1
            heapq.heappush(pending, (lb, serial, zero, active, c, j))

    mask = np.zeros(p, dtype=bool)
    add(mask, mask, np.zeros(p))
    while pending:
        lb, _, zero, active, c, j = heapq.heappop(pending)
        if lb >= best-tol:
            leaves.append((zero, active, lb, c))
            continue
        nz, na = zero.copy(), active.copy()
        nz[j] = True
        na[j] = True
        add(zero, na, c)
        add(nz, active, c)
    diagonal = d.tolist()
    return {'weights': weights.tolist(), 'objective': best,
            'certified_lower_bound': min(best, min(leaf[2] for leaf in leaves)),
            'subcube_leaves': [{'fixed_zero': z.tolist(), 'fixed_active': a.tolist(),
                                'lb': lb, 'minorant_diagonal': diagonal, 'minorant_center': c.tolist()}
                               for z, a, lb, c in leaves]}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    aligned = _align(study['panel'])
    stratum = np.asarray(study['panel']['stratum'])
    optical = study['interferometry']
    signal = np.asarray(optical['signal_phasor'], dtype=np.complex128)
    ref = np.asarray(optical['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(ref))
    h = np.column_stack((np.ones(len(stratum)), stratum == 1, stratum == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    theta = _calibrate(h[cal], y[cal])
    response = y[target] - np.einsum('ij,j->i', h[target], theta)
    selection = _select(np.asarray(study['X_cand'], dtype=np.float64), response,
                        float(study['lam2']), float(study['lam0']), float(study['M']))
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': response, 'certified_selection': selection}
