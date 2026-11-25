"""Best-bound branch and bound for thresholded box-L0 regression."""
import heapq
from itertools import product
import numpy as np
from scipy.optimize import lsq_linear, minimize


_TAU = 1e-9
_GAP = 9.8e-6


def _diagonal(H, Hx, He):
    """Propose a diagonal split, then certify the remainder by a factor witness."""
    if not np.any(Hx - np.diag(Hx.diagonal())):
        # The accumulation-error bound accounts for cancellation in dot products.
        d = np.nextafter(np.asarray(Hx.diagonal() - 2 * He.sum(axis=1), dtype=float), 0.)
        return d if np.all(d > 0) else None
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
            lo = np.nextafter(-ld(radius) - z, -np.inf)
            hi = np.nextafter(ld(radius) - z, np.inf)
            v = np.clip(-g / dl, lo, hi)
            # Interval bound on the gradient of the scalar tangent at v.
            # Use |v|, not the box radius, for rounding the value at v.
            gv = g + dl * v
            gve = ge + eg * (np.abs(g) + dl * np.abs(v))
            ve = (ge + eg * (np.abs(g) + dl * np.abs(v))) * np.abs(v)
            left = (gv + gve) * (lo - v)
            right = (gv - gve) * (hi - v)
            tail = np.minimum(left, right)
            return v * (g + .5 * dl * v) - ve + tail - eg * np.abs(tail)

        off = scalar(t)
        on = scalar(B) + ld(lam) * (1 - eg)
        vals = np.where(state == 0, off, np.where(state == 1, on, np.minimum(off, on)))
        bound = f + vals.sum() - fe - eg * (np.abs(vals).sum() + f)
        return np.nextafter(float(bound), -np.inf)

    # Pair highly correlated columns. Keep the full loss on rows belonging
    # to a single block; only linearize rows that couple different blocks.
    groups = []
    if p > 1 and B > t:
        scales = np.sqrt(H.diagonal() - ridge)
        normalizer = scales[:, None] * scales[None, :]
        corr = np.divide(np.abs(H), normalizer, out=np.zeros_like(H), where=normalizer > 0)
        np.fill_diagonal(corr, 0.)
        remaining = set(range(p))
        for k in np.argsort(-corr.ravel(), kind='stable'):
            j, l = divmod(int(k), p)
            if corr[j, l] < .8:
                break
            if j in remaining and l in remaining:
                cols = [j, l]
                remaining.difference_update(cols)
                if remaining:
                    third = max(sorted(remaining), key=lambda k: min(corr[j,k], corr[l,k]))
                    if min(corr[j,third], corr[l,third]) >= .8:
                        cols.append(third)
                        remaining.remove(third)
                groups.append(np.array(cols))
        if groups:
            groups.extend(np.array([j]) for j in sorted(remaining))
    def faces(Q):
        # Precompute the affine minimizer on each face of a three-column box.
        # Clip candidates for feasibility; the tangent, not the face search,
        # certifies the resulting minimum.
        if len(Q) != 3:
            return None
        signs = np.array(list(product((-1, 0, 1), repeat=3)), dtype=ld)
        P = np.zeros((len(signs),3,3), dtype=ld)
        for k, sign in enumerate(signs):
            free_idx = np.flatnonzero(sign == 0)
            if len(free_idx):
                ix = np.ix_(free_idx, free_idx)
                try:
                    P[k][ix] = np.linalg.inv(np.asarray(Q[ix], dtype=float))
                except np.linalg.LinAlgError:
                    pass
        return signs, P, np.eye(3, dtype=ld) - P @ Q

    blocks = []
    if groups:
        present = np.column_stack([np.any(A[:, cols] != 0, axis=1) for cols in groups])
        counts = present.sum(axis=1)
        if np.any(counts == 1):
            shared = counts != 1
            S, sy = A[shared], h[shared]
            for k, cols in enumerate(groups):
                local = (counts == 1) & present[:, k]
                E, ey = A[local][:, cols], h[local]
                Q = E.T @ E + ld(ridge) * np.eye(len(cols), dtype=ld)
                linear = E.T @ ey
                base = .5 * (ey @ ey)
                # Magnitudes before cancellation for bounding normal-equation errors.
                scale = (abs(ey) + ld(B) * np.abs(E).sum(axis=1))
                budget = scale @ scale + len(cols) * ld(ridge) * ld(B)**2
                patterns = np.arange(1 << len(cols))[:, None] >> np.arange(len(cols)) & 1
                blocks.append((cols, Q, linear, base, budget, patterns, faces(Q)))

    # In a dense observation basis, split the Gram matrix directly. Normalize
    # every block first so off-block coupling is measured relative to curvature.
    gram_blocks = []
    if groups:
        U = np.zeros_like(H)
        K = np.zeros_like(H)
        for cols in groups:
            ix = np.ix_(cols, cols)
            Q = H[ix]
            e, V = np.linalg.eigh(Q)
            U[ix] = (V / np.sqrt(np.maximum(e, ridge * .999))) @ V.T
        W = U @ H @ U
        roots = [np.linalg.inv(U[np.ix_(g, g)]) for g in groups]
        off = W.copy()
        for g in groups:
            off[np.ix_(g, g)] = 0.

        def propose(factor):
            # The common mode remains in the joint PSD remainder. Charge the
            # residual off-block entries separately by their singular directions.
            coupling = off - np.outer(factor, factor)
            result = H.copy()
            for i, cols in enumerate(groups):
                ix = np.ix_(cols, cols)
                v = roots[i] @ factor[cols]
                result[ix] -= np.outer(v, v)
                for j, other in enumerate(groups[:i]):
                    jx = np.ix_(other, other)
                    V, sig, T = np.linalg.svd(coupling[np.ix_(cols, other)], full_matrices=False)
                    F, G = roots[i] @ V, roots[j] @ T.T
                    result[ix] -= (F * sig) @ F.T
                    result[jx] -= (G * sig) @ G.T
            out = np.zeros_like(H)
            score = 0.
            for g in groups:
                ix = np.ix_(g, g)
                Q = (result[ix] + result[ix].T) * .5
                try:
                    L = np.linalg.cholesky(Q)
                except np.linalg.LinAlgError:
                    return -np.inf, out
                score += 2 * np.log(L.diagonal()).sum()
                out[ix] = Q
            return score, out * .999999

        e, V = np.linalg.eigh(off)
        v = V[:, -1] * np.sqrt(max(e[-1], 0.))
        all_cols = np.arange(p)
        complements = [np.setdiff1d(all_cols, g) for g in groups]
        for _ in range(18):
            for g, other in zip(groups, complements):
                mass = v[other] @ v[other]
                if mass > 1e-14:
                    v[g] = off[np.ix_(g, other)] @ v[other] / mass
        score, K = max((propose(np.sqrt(alpha)*v) for alpha in (0., .5, .8, 1.)), key=lambda x: x[0])
        if not np.isfinite(score):
            # Always available before rounding: a fraction of the block diagonal.
            base = np.zeros_like(H)
            for g in groups:
                ix = np.ix_(g, g)
                base[ix] = H[ix]
            fraction = max(0., .999999 * np.linalg.eigvalsh(W)[0])
            K = fraction * base
        Kl = K.astype(ld)
        try:
            L = np.linalg.cholesky(H - K).astype(ld)
        except np.linalg.LinAlgError:
            pass
        else:
            r0 = Hx - Kl - L @ L.T
            err = He + (8*p+16)*u*(np.abs(Hx) + np.abs(Kl) + abs(L) @ abs(L).T)
            fix = np.maximum(0., (np.abs(r0)+err).sum(axis=1) - abs(r0.diagonal()) - r0.diagonal())
            Kl[np.diag_indices(p)] = np.nextafter(Kl.diagonal() - 2*fix, -np.inf)
            for cols in groups:
                Q = Kl[np.ix_(cols, cols)]
                # Reject the split unless every repaired block is PSD.
                # Verify PSD in every block by a factor witness and a
                # diagonally dominant residual (including rounding error).
                k = len(cols)
                cushion = 64 * np.finfo(float).eps * np.max(np.abs(Q))
                try:
                    F = np.linalg.cholesky(np.asarray(Q - cushion*np.eye(k), dtype=float)).astype(ld)
                    E = Q - F @ F.T
                    err = (8*k+16)*u*(np.abs(Q) + np.abs(F) @ np.abs(F).T)
                    psd = np.all(2*E.diagonal() >= (np.abs(E)+err).sum(axis=1)) and np.all(E.diagonal() >= 0)
                except np.linalg.LinAlgError:
                    psd = False
                if not psd:
                    gram_blocks = []
                    break
                patterns = np.arange(1 << len(cols))[:,None] >> np.arange(len(cols)) & 1
                budget = ld(B)**2 * np.abs(Q).sum()
                gram_blocks.append((cols, Q, np.zeros(len(cols),dtype=ld), ld(0), budget, patterns, faces(Q)))

    def block_certificate(w, state, gram=False, smooth=0.):
        if gram:
            z0 = w.astype(ld)
            r = A @ z0 - h
            g = A.T @ r + ld(ridge)*z0
            cross = Kl @ z0 - g
            constant = .5*(r@r + ld(ridge)*(z0@z0)) - g@z0 + .5*(z0@(Kl@z0))
            domain = np.maximum(ld(B), np.abs(z0))
            residual_scale = absA @ domain + absh
            cross_scale = np.abs(Kl) @ domain + absA.T @ residual_scale + ld(ridge)*domain
            scale = residual_scale@residual_scale + ld(ridge)*(domain@domain) + domain@cross_scale
            active_blocks = gram_blocks
        else:
            u0 = sy - S @ w.astype(ld)
            # For any u0, shared loss is at least u0.T @ (sy-Sw)-||u0||²/2.
            cross = S.T @ u0
            cross_scale = np.abs(S).T @ np.abs(u0)
            constant = sy @ u0 - .5 * (u0 @ u0)
            scale = np.abs(sy) @ np.abs(u0) + .5 * (u0 @ u0)
            active_blocks = blocks
        candidate = np.empty(p)
        lower = constant
        for cols, Q, linear, base, budget, patterns, face in active_blocks:
            st = state[cols]
            patterns = patterns[np.all((st == -1) | (st == patterns), axis=1)]
            cap = np.where(patterns, ld(B), ld(t))
            c0 = linear + cross[cols]
            if len(cols) == 1:
                z = np.clip(c0 / Q[0, 0], -cap, cap)
            elif face is not None:
                signs, P, F = face
                corners = cap[:, None, :] * signs
                choices = np.clip((P @ c0)[None,:,:] + (F[None,:,:,:] @ corners[:,:,:,None])[...,0], -cap[:,None,:], cap[:,None,:])
                vals = (choices * (.5*(choices @ Q) - c0)).sum(axis=2)
                z = choices[np.arange(len(cap)), vals.argmin(axis=1)]
            else:
                q0, q1, q01 = Q[0, 0], Q[1, 1], Q[0, 1]
                det = q0*q1 - q01*q01
                # A two-dimensional box minimum is interior or on an edge.
                a0, a1 = cap[:, 0], cap[:, 1]
                left = np.stack((np.stack((-a0, np.clip((c0[1]+q01*a0)/q1, -a1, a1)), axis=1),
                                 np.stack(( a0, np.clip((c0[1]-q01*a0)/q1, -a1, a1)), axis=1),
                                 np.stack((np.clip((c0[0]+q01*a1)/q0, -a0, a0), -a1), axis=1),
                                 np.stack((np.clip((c0[0]-q01*a1)/q0, -a0, a0),  a1), axis=1)), axis=1)
                if det > 0:
                    v = np.array([(c0[0]*q1-c0[1]*q01)/det, (c0[1]*q0-c0[0]*q01)/det])
                    left = np.concatenate((left, np.clip(v, -cap, cap)[:, None, :]), axis=1)
                val = (left * (.5 * (left @ Q) - c0)).sum(axis=2)
                z = left[np.arange(len(cap)), val.argmin(axis=1)]
            g0 = z @ Q - c0
            # A tangent is independently valid even if the box minimizer is inaccurate.
            val = base + (z * (.5 * (z @ Q) - c0)).sum(axis=1) + ld(lam) * patterns.sum(axis=1)
            tangent = val + np.minimum(g0 * (-cap - z), g0 * (cap - z)).sum(axis=1)
            j = val.argmin()
            if smooth:
                probs = np.exp(-(val - val[j]) / smooth)
                mass = probs.sum()
                candidate[cols] = (probs @ z) / mass
                lower += val[j] - smooth * np.log(mass)
            else:
                candidate[cols] = z[j]
                lower += tangent.min()
            scale += budget + ld(B) * cross_scale[cols].sum() + len(cols) * ld(lam)
        return np.nextafter(float(lower - 16 * eg * scale), -np.inf), candidate

    if gram_blocks:
        remainder = H - np.asarray(Kl, dtype=float)
        eigenvalues, vectors = np.linalg.eigh(remainder)
        transform = vectors / np.sqrt(np.maximum(eigenvalues, np.max(eigenvalues)*1e-10))
        inverse = np.linalg.pinv(transform)

    def improve_gram(point, state, effort, upper):
        # Smooth the finite minimum for the search only. Certification always
        # recomputes the unsmoothed tangent at the final point.
        x = inverse @ point
        lb, witness = block_certificate(point, state, True)
        best = (lb, witness, point)
        temps = (.015, .001, .00003) if effort > 20 else (.001, .00003)
        for temp in temps:
            if best[0] >= upper - _GAP:
                break
            def fun(x):
                z = transform @ x
                val, mean = block_certificate(z, state, True, lam*temp)
                return -val, transform.T @ (remainder @ (z-mean))
            x = minimize(fun, x, jac=True, method='L-BFGS-B', options={'maxiter': effort, 'ftol': 1e-13, 'gtol': 1e-9}).x
            z = transform @ x
            lb, witness = block_certificate(z, state, True)
            if lb > best[0]:
                best = (lb, witness, z)
        return best

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

    def visit(state, warm, parent_lb=-np.inf, gram_warm=None):
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
        point = gram_warm
        if blocks and lb < upper - _GAP:
            for anchor in (best, w):
                block_lb, witness = block_certificate(anchor, state)
                lb = max(lb, block_lb)
                offer(witness)
                if lb >= upper - _GAP:
                    break
        if gram_blocks and lb < upper - _GAP:
            dual = -np.inf
            for anchor in (best, w, gram_warm):
                if anchor is None:
                    continue
                block_lb, witness = block_certificate(anchor, state, True)
                if block_lb > dual:
                    dual, point = block_lb, anchor
                lb = max(lb, block_lb)
                offer(witness)
                if lb >= upper - _GAP:
                    break
            if lb < upper - _GAP and free.any():
                dual, witness, point = improve_gram(point, state, 32 if nodes == 1 else 8, upper)
                lb = max(lb, dual)
                offer(witness)
                polish(np.abs(witness) > _TAU)
        if lb >= upper - _GAP:
            closed = min(closed, lb)
            return
        if not free.any():
            raise ArithmeticError('continuous leaf did not reach certification tolerance')
        serial += 1
        heapq.heappush(heap, (lb, serial, state, w, frac, point))

    visit(np.full(p, 0 if B <= t else -1, dtype=np.int8), best)
    while heap and heap[0][0] < upper - _GAP:
        if nodes + 2 > 600:
            raise ArithmeticError('600-node budget insufficient for the requested certificate')
        lb, _, state, w, frac, point = heapq.heappop(heap)
        free = np.flatnonzero(state == -1)
        # Branch at the largest convex-envelope penalty interpolation gap.
        j = free[np.argmax(np.minimum(frac[free], 1-frac[free]))]
        for side in (int(frac[j] >= .5), int(frac[j] < .5)):
            child = state.copy()
            child[j] = side
            visit(child, w, lb, point)
    lower = min(closed, heap[0][0] if heap else np.inf)
    r = X @ best - y
    objective = float(.5 * (r @ r + ridge * (best @ best)) + lam * np.count_nonzero(np.abs(best) > _TAU))
    if not 0 <= upper - min(lower, upper) <= 1e-5:
        raise ArithmeticError('could not certify the requested gap')
    return {'weights': best.tolist(), 'objective': objective,
            'certified_lower_bound': min(lower, np.nextafter(objective, -np.inf)),
            'nodes_explored': nodes}
