"""Reference-panel calibration and certified connected box-L0 selection."""

import heapq
import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize


def _aligned(panel):
    groups, group = np.unique(panel['center_id'], return_inverse=True)
    stratum = np.asarray(panel['stratum'], dtype=int)
    raw = np.asarray(panel['X_panel_raw'], dtype=np.float64)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    aligned = np.empty_like(raw)
    for i in range(len(groups)):
        rows = group == i
        ref = controls[rows]
        finite = np.isfinite(ref)
        keep = finite.sum(axis=0) >= 2
        ref = np.where(finite[:, keep], ref[:, keep], np.nan)
        offset = np.median(np.nanmean(ref, axis=0))
        gain = np.median(np.nanstd(ref, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - offset) / gain
    cell = 3 * group + stratum
    weight = len(cell) / (3 * len(groups) * np.bincount(cell)[cell])
    ok = np.isfinite(aligned)
    fill = (weight @ np.where(ok, aligned, 0.0)) / (weight @ ok)
    return np.where(ok, aligned, fill)


def _cauchy(H, y):
    theta = np.linalg.lstsq(H, y, rcond=None)[0]
    floor = 1e-10 * max(1.0, np.median(np.abs(y)))
    # Start with broader downweighting, then adapt the scale to inlier scatter.
    for scale_factor, limit in ((3.0, 14), (1.7, 24), (1.35, 42)):
        for _ in range(limit):
            r = y - H @ theta
            mad = 1.4826 * np.median(np.abs(r - np.median(r)))
            root_weight = 1.0 / np.hypot(1.0, r / max(floor, scale_factor * mad))
            new = np.linalg.lstsq(root_weight[:, None] * H, root_weight * y, rcond=None)[0]
            if np.linalg.norm(new - theta) <= 1e-10 * (1 + np.linalg.norm(theta)):
                theta = new
                break
            theta = new
    return theta


def _box_l0(X, y, lam0, lam2, M):
    p = X.shape[1]
    bits = 1 << np.arange(p)
    full = (1 << p) - 1
    ld = np.longdouble
    xl, yl = X.astype(ld), y.astype(ld)
    gl, hl, cl = xl.T @ xl, xl.T @ yl, (yl @ yl) / 2
    G, h = gl.astype(float), hl.astype(float)
    A = G + lam2 * np.eye(p)
    scale = np.sqrt(np.maximum(0, np.diag(G)))
    safe_scale = np.where(scale > 0, scale, 1.)
    corr = G / safe_scale[:, None] / safe_scale[None, :]
    # Scaling the columns before taking the smallest eigenvalue gives a
    # nonnegative, scale-aware PSD minorant of G (without the ridge term).
    eig = np.linalg.eigvalsh(corr)[0]
    d = max(0., eig - 1e-10) * scale ** 2
    Q = G - np.diag(d)
    q = d + lam2
    lipschitz = max(1e-12, np.linalg.eigvalsh(Q)[-1])
    a = np.full(p, M)
    nz = q > 0
    a[nz] = np.minimum(M, np.sqrt(2 * lam0 / q[nz]))
    slope = np.divide(lam0, a, out=np.zeros(p), where=a > 0) + .5 * q * a
    # Worst-case magnitude of the expanded quadratic on the box, used for
    # outward rounding of the bounds obtained in extended precision.
    guard = float(64 * (max(X.shape) + 4) * np.finfo(ld).eps *
                  (1 + cl + M * np.sum(np.abs(xl).T @ np.abs(yl)) +
                   M * M * np.sum(np.abs(xl).T @ np.abs(xl)) + p * (lam0 + M*M*lam2)))
    guard += 2e-10

    def bound(center, zero, active):
        c = center.astype(ld)
        dd = d.astype(ld)
        g = hl - gl @ c
        const = cl - (c @ gl @ c) / 2 + (dd * c) @ c / 2
        linear = np.abs(g + dd * c)
        b = np.divide(linear, q.astype(ld), out=np.full(p, M, dtype=ld), where=(q > 0) & (linear < M * q))
        on = lam0 - b * (linear - q * b / 2)
        # The specified count treats coefficients up to 1e-8 as inactive.
        # Allow that interval in the bound; it also bounds the exact-zero cube.
        small = np.minimum(b, 1e-8)
        off = -small * (linear - q * small / 2)
        val = const + on[active].sum() + off[zero].sum()
        val += np.minimum(on[~(zero | active)], off[~(zero | active)]).sum()
        return float(np.nextafter(float(val) - guard, -np.inf))

    def relaxation(start, zero, active):
        # Eliminate relaxed indicators via the box perspective of q*w^2/2.
        # The resulting coordinate penalty is linear up to a and quadratic above it.
        free = ~(zero | active)
        threshold = np.where(free, slope / lipschitz, 0.)
        anchor = np.where(free, a, 0.)
        shrink = 1.0 / (1.0 + q / lipschitz)
        x = start.copy()
        x[zero] = 0.
        z = x.copy()
        momentum = 1.
        for _ in range(320):
            u = z + (h - Q @ z) / lipschitz
            mag = np.abs(u)
            linear = np.maximum(0., mag - threshold)
            new = np.sign(u) * np.minimum(M, np.where(linear <= anchor, linear, mag * shrink))
            new[zero] = 0.
            step = new - x
            if np.max(np.abs(step)) <= 2e-10 * (1 + M):
                return new
            if (z - new) @ step > 0:
                z, momentum = new, 1.
            else:
                nxt = .5 * (1. + np.sqrt(1. + 4. * momentum * momentum))
                z, momentum = new + ((momentum - 1.) / nxt) * step, nxt
            x = new
        return x

    best = np.zeros(p)
    upper = float(cl)
    tried = set()

    def primal(mask, start):
        nonlocal best, upper
        if mask in tried or not mask:
            return
        tried.add(mask)
        idx = np.flatnonzero(bits & mask)
        B, f = A[np.ix_(idx, idx)], h[idx]
        try:
            z = solve(B, f, assume_a='pos', check_finite=False)
        except np.linalg.LinAlgError:
            z = start[idx]
        if np.any(np.abs(z) > M) or np.max(np.abs(B @ z - f)) > 1e-7:
            def fun(v):
                g = B @ v
                return .5 * (v @ g) - f @ v, g - f
            z = minimize(fun, np.clip(z, -M, M), jac=True, method='L-BFGS-B',
                         bounds=[(-M, M)] * len(idx), options={'ftol': 1e-15, 'gtol': 1e-10}).x
            interior = np.abs(z) < M
            if interior.any():
                rhs = f[interior] - B[np.ix_(interior, ~interior)] @ z[~interior]
                try:
                    polish = solve(B[np.ix_(interior, interior)], rhs, assume_a='pos', check_finite=False)
                    if np.all(np.abs(polish) <= M):
                        z[interior] = polish
                except np.linalg.LinAlgError:
                    pass
        w = np.zeros(p)
        w[idx] = np.where(np.abs(z) > 1e-8, z, 0.)
        r = X @ w - y
        obj = .5 * (r @ r + lam2 * (w @ w)) + lam0 * np.count_nonzero(w)
        if obj < upper:
            upper, best = float(obj), w

    heap = []
    leaves = []
    serial = 0
    tol = 3e-6

    def visit(zmask, amask, start, parent_lb=-np.inf, parent_center=None):
        nonlocal serial
        zero, active = (bits & zmask) != 0, (bits & amask) != 0
        free = ~(zero | active)
        if parent_center is not None:
            pb = bound(parent_center, zero, active)
            if pb >= upper - tol:
                leaves.append((zero, active, pb, parent_center))
                return
        if not free.any():
            # Solve the fixed-support box QP even when the relaxation is slow.
            idx = np.flatnonzero(active)
            c = relaxation(start, zero, active)
            if idx.size:
                primal(amask, c)
                if np.array_equal(best != 0, active):
                    c = best
            lb = bound(c, zero, active)
            if lb < upper - tol:
                # Tighten this leaf using its own box fit rather than the incumbent.
                if idx.size:
                    B, f = A[np.ix_(idx, idx)], h[idx]
                    def fun(v):
                        g = B @ v
                        return .5 * v @ g - f @ v, g - f
                    c = c.copy()
                    c[idx] = minimize(fun, c[idx], jac=True, method='L-BFGS-B',
                                      bounds=[(-M, M)] * len(idx),
                                      options={'ftol': 1e-15, 'gtol': 1e-11}).x
                    lb = bound(c, zero, active)
            leaves.append((zero, active, lb, c))
            return
        c = relaxation(start, zero, active)
        lb = bound(c, zero, active)
        if parent_center is not None and pb > lb:
            lb, witness = pb, parent_center
        else:
            witness = c
        if lb < upper - tol:
            frac = np.divide(np.abs(c), a, out=np.ones(p), where=a > 0)
            for cutoff in ((.3, .6, .9) if not (zmask | amask) else (.55,)):
                primal(amask | int(bits[free & (frac >= cutoff)].sum()), c)
        if lb >= upper - tol:
            leaves.append((zero, active, lb, witness))
        else:
            serial += 1
            heapq.heappush(heap, (lb, serial, zmask, amask, c, witness))

    visit(0, 0, best)
    while heap:
        lb, _, zmask, amask, c, witness = heapq.heappop(heap)
        if lb >= upper - tol:
            leaves.append(((bits & zmask) != 0, (bits & amask) != 0, lb, witness))
            continue
        if len(heap) + len(leaves) + 2 > 2048:
            raise ArithmeticError('Box-L0 certificate exceeded 2048 leaves')
        free = (bits & (zmask | amask)) == 0
        frac = np.clip(np.divide(np.abs(c), a, out=np.ones(p), where=a > 0), 0., 1.)
        j = int(np.argmax(np.where(free, frac * (1. - frac), -1.)))
        bit = int(bits[j])
        visit(zmask | bit, amask, c, lb, witness)
        visit(zmask, amask | bit, c, lb, witness)

    lower = min(l[2] for l in leaves)
    if not 0 <= upper - lower <= 1e-5:
        raise ArithmeticError(f'Box-L0 certificate gap is {upper - lower:g}')
    diagonal = d.tolist()
    return {'schema': 'hegel_box_l0_cert_schema_v2',
            'weights': best.tolist(), 'objective': upper, 'certified_lower_bound': lower,
            'subcube_leaves': [dict(fixed_zero=z.tolist(), fixed_active=b.tolist(), lb=lb,
                                    minorant_diagonal=diagonal, minorant_center=c.tolist())
                               for z, b, lb, c in leaves]}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    panel = study['panel']
    aligned = _aligned(panel)
    optical = study['interferometry']
    signal = np.asarray(optical['signal_phasor'], dtype=np.complex128)
    ref = np.asarray(optical['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(ref))
    s = np.asarray(panel['stratum'])
    H = np.column_stack((np.ones(len(s)), s == 1, s == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    theta = _cauchy(H[cal], y[cal])
    residual = y[target] - H[target] @ theta
    selection = _box_l0(np.asarray(study['X_cand'], dtype=np.float64), residual,
                        float(study['lam0']), float(study['lam2']), float(study['M']))
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': residual, 'certified_selection': selection}
