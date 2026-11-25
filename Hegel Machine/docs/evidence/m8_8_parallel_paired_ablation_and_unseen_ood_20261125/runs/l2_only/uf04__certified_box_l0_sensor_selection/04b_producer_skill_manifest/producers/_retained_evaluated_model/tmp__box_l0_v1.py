"""Best-bound branch and bound for thresholded box-L0 regression."""
import heapq
import numpy as np
from scipy.optimize import lsq_linear


_TAU = 1e-9
_GAP = 9.8e-6


def _diagonal(H, Hx, He):
    """Propose a diagonal split, then certify the remainder by a factor witness."""
    s = np.sqrt(H.diagonal())
    A = H / s[:, None] / s[None, :]
    p = len(s)
    j = np.arange(p)
    I = np.eye(p)
    d = np.full(p, max(0., np.linalg.eigvalsh(A)[0]) * .5)
    if np.all(d > 1e-13):
        for mu in (1., .2, .04, .008):
            for _ in range(20):
                R = A - np.diag(d)
                inv = np.linalg.solve(R, I)
                g = mu * inv[j, j] - 1 / d
                step = np.linalg.solve(mu * inv**2 + np.diag(d**-2), -g)
                dec = -g @ step
                if dec < 1e-10:
                    break
                old = -np.log(d).sum() - mu * np.linalg.slogdet(R)[1]
                t = 1.
                while t > 1e-8:
                    z = d + t * step
                    if np.all(z > 0):
                        try:
                            L = np.linalg.cholesky(A - np.diag(z))
                        except np.linalg.LinAlgError:
                            pass
                        else:
                            val = -np.log(z).sum() - 2 * mu * np.log(L[j, j]).sum()
                            if val <= old - .01 * t * dec:
                                d = z
                                break
                    t *= .5
                else:
                    break
    else:
        return None
    d *= .999999 * s**2
    try:
        L = np.linalg.cholesky(H - np.diag(d)).astype(np.longdouble)
    except np.linalg.LinAlgError:
        return None
    r = Hx - np.diag(d.astype(np.longdouble)) - L @ L.T
    u = np.finfo(np.longdouble).eps
    err = He + (8 * p + 16) * u * (np.abs(Hx) + np.diag(d) + np.abs(L) @ np.abs(L).T)
    # H-diag(d) = L L.T + remainder. After repairing its diagonal,
    # the remainder is symmetric diagonally dominant and nonnegative.
    correction = np.maximum(0, (np.abs(r) + err).sum(axis=1) - np.abs(r.diagonal()) - r.diagonal())
    d = np.asarray(d - correction * (1 + (8 * p + 16) * u), dtype=float)
    d = np.nextafter(d, 0.)
    return d if np.all(d > 0) else None


def solve_certified_box_l0(instance: dict) -> dict:
    X = np.asarray(instance['X'], dtype=float)
    y = np.asarray(instance['y'], dtype=float)
    ridge, B, lam = (float(instance[k]) for k in ('ridge', 'box', 'l0_penalty'))
    n, p = X.shape
    t = min(_TAU, B)
    H = X.T @ X + ridge * np.eye(p)
    c = X.T @ y
    ld = np.longdouble
    A, h = X.astype(ld), y.astype(ld)
    absA, absh = np.abs(A), np.abs(h)
    u = np.finfo(ld).eps
    eg = (8 * n + 8 * p + 32) * u
    Hx = A.T @ A + ld(ridge) * np.eye(p, dtype=ld)
    He = eg * (absA.T @ absA + ld(ridge) * np.eye(p, dtype=ld))

    # The split can always retain the ridge curvature, since X.T@X is PSD.
    d = _diagonal(H, Hx, He) if p else None
    if d is None:
        d = np.full(p, ridge)
    q = np.maximum(0., H.diagonal() - d)
    a = np.minimum(B, t + np.sqrt(2 * lam / d))
    if B > t:
        slope = .5 * d * (a + t) + lam / (a - t)
    else:
        slope = np.zeros(p)
    dl = d.astype(ld)

    best = np.zeros(p)
    upper = np.inf

    def offer(w):
        nonlocal best, upper
        z = w.astype(ld)
        r = A @ z - h
        re = eg * (absA @ np.abs(z) + absh)
        value = (.5 * ((np.abs(r) + re) @ (np.abs(r) + re) + ld(ridge) * (z @ z)) + ld(lam) * np.count_nonzero(np.abs(w) > _TAU))
        value = np.nextafter(float(value * (1 + eg)), np.inf)
        if value < upper:
            best, upper = w.copy(), value

    offer(best)

    def certificate(w, state):
        # Tangent the PSD remainder and minimize each scalar quadratic over
        # its exact indicator domain. Paid indicators permit the whole box:
        # overlap with the free interval changes no minimum over indicators.
        z = w.astype(ld)
        r = A @ z - h
        rscale = absA @ np.abs(z) + absh
        re = eg * rscale
        g = A.T @ r + ld(ridge) * z
        ge = eg * (absA.T @ (np.abs(r) + rscale) + ld(ridge) * np.abs(z))
        f = .5 * (r @ r + ld(ridge) * (z @ z))
        fe = np.abs(r) @ re + .5 * (re @ re) + eg * (f + np.abs(r) @ re)

        def scalar(radius):
            lo, hi = -ld(radius) - z, ld(radius) - z
            v = np.clip(-g / dl, lo, hi)
            gv = g + dl * v
            # A second tangent bounds the scalar minimum even if v is rounded.
            val = v * (g + .5 * dl * v) + np.minimum(gv * (lo - v), gv * (hi - v))
            size = np.maximum(np.abs(lo), np.abs(hi))
            return val - (ge + eg * (np.abs(g) + dl * size)) * size

        off = scalar(t)
        on = scalar(B) + ld(lam) * (1 - eg)
        vals = np.where(state == 0, off, np.where(state == 1, on, np.minimum(off, on)))
        bound = f + vals.sum() - fe - eg * (np.abs(vals).sum() + f)
        return np.nextafter(float(bound), -np.inf)

    def relax(w, state):
        w = np.clip(w, -np.where(state == 0, t, B), np.where(state == 0, t, B))
        g = H @ w - c
        free = (state == -1) & (B > t)
        for k in range(350):
            change = 0.
            for j in range(p):
                hj = H[j, j]
                score = hj * w[j] - g[j]
                m = abs(score)
                if free[j]:
                    if m <= hj * t:
                        v = m / hj
                    elif m <= q[j] * t + slope[j]:
                        v = t
                    elif m < q[j] * a[j] + slope[j]:
                        v = min(a[j], (m - slope[j]) / q[j])
                    elif a[j] == B:
                        v = B
                    else:
                        v = min(B, m / hj)
                else:
                    v = min(t if state[j] == 0 else B, m / hj)
                v = np.copysign(v, score)
                diff = v - w[j]
                if diff:
                    w[j] = v
                    g += diff * H[:, j]
                    change = max(change, abs(diff))
            if change <= 3e-13 * (1 + np.max(np.abs(w), initial=0.)):
                break
            if k % 24 == 23:
                g = H @ w - c
        return w

    aug = np.vstack((X, np.sqrt(ridge) * np.eye(p)))
    rhs = np.r_[y, np.zeros(p)]
    cache = set()

    def polish(mask):
        key = mask.tobytes()
        if key in cache:
            return
        cache.add(key)
        if not p:
            return
        cap = np.where(mask, B, t)
        w = np.linalg.lstsq(aug, rhs, rcond=None)[0] if mask.all() else None
        if w is None or np.any(np.abs(w) > cap):
            w = lsq_linear(aug, rhs, bounds=(-cap, cap), method='bvls', tol=1e-13).x
        offer(np.clip(w, -cap, cap))

    nodes = 0
    closed = np.inf
    serial = 0
    heap = []

    def visit(state, warm, parent_lb=-np.inf):
        nonlocal nodes, closed, serial
        nodes += 1
        w = relax(warm, state)
        free = state == -1
        size = np.abs(w)
        frac = np.divide(size - t, a - t, out=np.zeros(p), where=a > t).clip(0, 1)
        paid = state == 1
        polish(paid | (free & (frac >= .5)))
        if nodes == 1:
            for cutoff in (.05, .25, .75, .95):
                polish(paid | (free & (frac >= cutoff)))
            polish(np.zeros(p, dtype=bool))
        lb = max(parent_lb, certificate(w, state))
        if lb >= upper - _GAP:
            closed = min(closed, lb)
            return
        if not free.any():
            raise ArithmeticError('continuous leaf did not reach certification tolerance')
        serial += 1
        heapq.heappush(heap, (lb, serial, state, w, frac))

    visit(np.full(p, 0 if B <= t else -1, dtype=np.int8), best)
    while heap and heap[0][0] < upper - _GAP:
        if nodes + 2 > 600:
            raise ArithmeticError('600-node budget insufficient for the requested certificate')
        lb, _, state, w, frac = heapq.heappop(heap)
        free = np.flatnonzero(state == -1)
        # Branch at the largest convex-envelope penalty interpolation gap.
        j = free[np.argmax(np.minimum(frac[free], 1-frac[free]))]
        for side in (int(frac[j] >= .5), int(frac[j] < .5)):
            child = state.copy()
            child[j] = side
            visit(child, w, lb)
    lower = min(closed, heap[0][0] if heap else np.inf)
    r = X @ best - y
    objective = float(.5 * (r @ r + ridge * (best @ best)) + lam * np.count_nonzero(np.abs(best) > _TAU))
    if not 0 <= upper - min(lower, upper) <= 1e-5:
        raise ArithmeticError('could not certify the requested gap')
    return {'weights': best.tolist(), 'objective': objective,
            'certified_lower_bound': min(lower, np.nextafter(objective, -np.inf)),
            'nodes_explored': nodes}
