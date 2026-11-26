"""Box-constrained sparse ridge with a checkable minorant at each B&B leaf."""
import heapq

import numpy as np
from scipy.optimize import minimize


def _gram_minorant(X):
    # Gershgorin with outward allowance for Gram dot products and row sums.
    XL = X.astype(np.longdouble)
    G = XL.T @ XL
    AX = abs(XL)
    guard = 8 * (max(X.shape) + 4) * np.finfo(XL.dtype).eps
    err = guard * (AX.T @ AX)
    off = abs(G) + err
    np.fill_diagonal(off, 0)
    d = G.diagonal() - err.diagonal() - (1 + guard) * off.sum(axis=1)
    if np.all(d >= 0):
        return np.maximum(0., np.nextafter(d.astype(float), -np.inf))
    # A guarded Cholesky test permits a nonzero minorant even without
    # diagonal dominance. Verify its factorization by a residual bound.
    Gf = X.T @ X
    floor = np.linalg.eigvalsh(Gf)[0]
    if floor <= 0:
        return np.zeros(X.shape[1])
    d = np.full(X.shape[1], 0.98 * floor)
    R = G - np.diag(d.astype(XL.dtype))
    try:
        C = np.linalg.cholesky(R.astype(float)).astype(XL.dtype)
        V = np.linalg.solve(C.astype(float), np.eye(len(d))).astype(XL.dtype)
    except np.linalg.LinAlgError:
        return np.zeros_like(d)
    # ||CV-I||_inf bounds invertibility; bound ||C^-1||_2 by
    # its Frobenius norm via C^-1 = V(CV)^-1.
    E = C @ V - np.eye(len(d))
    e = np.linalg.norm(abs(E) + guard * (abs(C) @ abs(V) + np.eye(len(d))), 'fro')
    delta = np.max(np.sum(abs(R - C @ C.T) + err + guard * (abs(G) + np.diag(d) + abs(C) @ abs(C.T)), axis=1))
    if e < 1 and delta < (1-e)**2 / ((1+guard) * np.sum(V*V)):
        return d
    return np.zeros_like(d)


def _row_components(X):
    """Find exactly row-disjoint column groups for branch ordering."""
    present = X != 0
    remaining = np.ones(X.shape[1], dtype=bool)
    groups = []
    while remaining.any():
        cols = np.zeros_like(remaining)
        cols[np.flatnonzero(remaining)[0]] = True
        while True:
            expanded = cols | present[present[:, cols].any(axis=1)].any(axis=0)
            if np.array_equal(expanded, cols):
                break
            cols = expanded
        groups.append(np.flatnonzero(cols))
        remaining[cols] = False
    return groups


class _Search:
    def __init__(self, X, y, lam0, lam2, box):
        self.X, self.y, self.l0, self.l2, self.box = X, y, lam0, lam2, box
        self.G = X.T @ X
        self.g = X.T @ y
        self.d = _gram_minorant(X)
        self.H = self.G - np.diag(self.d)
        self.k = self.d + lam2
        self.tip = np.minimum(box, np.sqrt(2*lam0 / self.k))
        self.slope = lam0/self.tip + .5*self.k*self.tip
        self.lip = max(1e-12, np.linalg.eigvalsh(self.H)[-1])
        self.idx = np.arange(X.shape[1])
        self.groups = _row_components(X)
        self.XL, self.yL, self.dL = (z.astype(np.longdouble) for z in (X, y, self.d))
        self.Xa = abs(self.XL)
        self.guard = 8 * (max(X.shape) + 4) * np.finfo(self.XL.dtype).eps
        self.best = .5 * float(y @ y)
        self.w = np.zeros(X.shape[1])
        self.tried = set()

    def witness(self, c, zero, active):
        c = c.astype(np.longdouble)
        r = self.yL - self.XL @ c
        delta = self.guard * (abs(self.yL) + self.Xa @ abs(c))
        g = self.XL.T @ r
        eg = (1+self.guard)*(self.Xa.T @ delta) + self.guard*(self.Xa.T @ abs(r))
        dc = self.dL*c
        q = np.maximum(abs(r)-delta, 0)
        const = .5*(q @ q + dc @ c) + g @ c
        z = abs(g + dc) + eg + self.guard*(abs(g) + abs(dc))
        k = self.dL + np.longdouble(self.l2)
        b = np.minimum(self.box, z/k)
        gain = b*(z-.5*k*b)
        on = self.l0 - gain
        v = const + on[active].sum() + np.minimum(0, on[~(zero | active)]).sum()
        err = eg @ abs(c) + self.guard*(q @ q + abs(dc) @ abs(c) + abs(g) @ abs(c) + len(c)*self.l0 + gain.sum())
        return float(np.nextafter(float(v-err), -np.inf))

    def fit(self, use):
        key = sum(1 << int(i) for i in self.idx[use])
        if key in self.tried:
            return
        self.tried.add(key)
        if not key:
            return
        ids = self.idx[use]
        Q = self.G[np.ix_(ids, ids)] + self.l2*np.eye(len(ids))
        g = self.g[ids]
        v = np.linalg.solve(Q, g)
        if np.max(abs(v)) > self.box:
            def fun(z):
                qz = Q @ z
                return (.5*qz-g) @ z, qz-g
            v = minimize(fun, np.clip(v, -self.box, self.box), jac=True,
                         bounds=[(-self.box, self.box)]*len(ids), method='L-BFGS-B',
                         options={'gtol': 1e-11, 'ftol': 1e-15}).x
            # Finish the box fit by exact coordinate minimization.
            for _ in range(80):
                before = v.copy()
                for j in range(len(v)):
                    v[j] = np.clip(v[j] + (g[j]-Q[j] @ v)/Q[j,j], -self.box, self.box)
                if np.max(abs(v-before)) <= 1e-12:
                    break
        w = np.zeros_like(self.w)
        w[ids] = np.where(abs(v) > 1e-8, v, 0.)
        r = self.X @ w-self.y
        value = .5*float(r @ r) + .5*self.l2*float(w @ w) + self.l0*np.count_nonzero(w)
        if value < self.best:
            self.best, self.w = value, w

    def relax(self, zero, active, start):
        ids = self.idx[~zero]
        a = active[ids]
        H, g = self.H[np.ix_(ids, ids)], self.g[ids]
        c = start[ids].copy()
        u = c.copy()
        k, h, tip = self.k[ids], self.slope[ids], self.tip[ids]
        inv = 1/self.lip
        den = 1 + inv*k
        shrink = inv*np.where(a, 0., h)
        # If the envelope is linear all the way to the box wall, its
        # slope can exceed k*tip. Switch only after that linear segment.
        switch = tip + shrink
        t = 1.
        for it in range(320):
            v = u + inv*(g - H @ u)
            av = abs(v)
            mag = np.where(a | (av >= switch), av/den, np.maximum(0., av-shrink))
            new = np.sign(v)*np.minimum(self.box, mag)
            dif = new-c
            if not len(ids) or np.max(abs(dif)) <= 2e-11:
                c = new
                break
            if (u-new) @ dif > 0:
                u, t = new, 1.
            else:
                tn = .5*(1+np.sqrt(1+4*t*t))
                u, t = new + ((t-1)/tn)*dif, tn
            c = new
        center = np.zeros_like(start)
        center[ids] = c
        frac = np.minimum(1, abs(center)/self.tip)
        free = ~(zero | active)
        if free.any():
            self.fit(active | (free & (frac > .5)))
            self.fit(active | (free & (frac > .99)))
        else:
            self.fit(active)
        lb = self.witness(center, zero, active)
        return lb, center, frac

    def solve(self):
        p = len(self.w)
        f = np.zeros(p, dtype=bool)
        pending = []
        serial = 0
        done = []
        tol = 8e-6

        def push(zero, active, c):
            nonlocal serial
            lb, c, frac = self.relax(zero, active, c)
            heapq.heappush(pending, (lb, serial, zero, active, c, frac))
            serial += 1

        push(f, f, self.w)
        while pending:
            lb, _, zero, active, c, frac = heapq.heappop(pending)
            if lb >= self.best - tol:
                done.append({'fixed_zero': zero, 'fixed_active': active, 'lb': lb,
                             'minorant_diagonal': self.d.copy(), 'minorant_center': c})
                continue
            free = self.idx[~(zero | active)]
            if not free.size:
                raise RuntimeError('Fixed-support minorant did not close the proof gap')
            if len(done)+len(pending)+2 > 2048:
                raise RuntimeError('Proof exceeds the 2048-leaf limit')
            # Finish a partially assigned independent block when it still
            # has an envelope gap. Interleaving blocks multiplies unresolved
            # alternatives before any block can contribute a pruning margin.
            merit = np.minimum(frac, 1-frac)
            fixed = zero | active
            for group in self.groups:
                if fixed[group].any():
                    local = group[~fixed[group]]
                    if local.size and np.max(merit[local]) > 1e-8:
                        free = local
                        break
            j = free[np.argmax(merit[free])]
            z = zero.copy()
            z[j] = True
            a = active.copy()
            a[j] = True
            push(z, active, c)
            push(zero, a, c)
        return {'weights': self.w, 'objective': self.best,
                'certified_lower_bound': min(self.best, min(leaf['lb'] for leaf in done)),
                'subcube_leaves': done}


def solve_proof_carrying_box_l0(problem: dict) -> dict:
    """Minimize the sparse ridge objective and return a partition certificate."""
    return _Search(np.asarray(problem['X'], dtype=float), np.asarray(problem['y'], dtype=float),
                   float(problem['lam0']), float(problem['lam2']), float(problem['M'])).solve()
