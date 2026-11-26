"""Branch-and-bound for box-constrained sparse ridge, with explicit leaf witnesses."""
import heapq
import itertools

import numpy as np


def _minorant(X, G):
    """Use one diagonal-dominance certificate, or a checked uniform shift."""
    xl = X.astype(np.longdouble)
    u = np.finfo(xl.dtype).eps
    gram = xl.T @ xl
    err = 8 * (X.shape[0] + 2) * u * (abs(xl).T @ abs(xl))
    off = abs(gram) + err
    np.fill_diagonal(off, 0)
    margin = gram.diagonal() - err.diagonal() - off.sum(axis=1)
    margin -= 8 * (G.shape[0] + 2) * u * (abs(gram.diagonal()) + off.sum(axis=1))
    eig = np.linalg.eigvalsh(G)[0]
    uniform = max(0., eig - 1e-10 * np.max(np.abs(G)))
    if uniform > 0:
        try:
            shifted = G - uniform * np.eye(G.shape[0])
            L = np.linalg.cholesky(shifted)
            LL = L.astype(np.longdouble)
            residual = gram - LL @ LL.T
            residual[np.diag_indices_from(G)] -= uniform
            inv = np.linalg.solve(L, np.eye(G.shape[0])).astype(np.longdouble)
            inverse_err = np.linalg.norm(np.asarray(np.eye(G.shape[0]) - LL @ inv, dtype=float))
            if inverse_err >= .01 or (np.max(np.sum(abs(residual) + err, axis=1)) + 64 * G.shape[0] * u * np.max(abs(gram))) * np.sum(inv * inv) >= .9:
                uniform = 0.
        except np.linalg.LinAlgError:
            uniform = 0.
    if np.all(margin >= 0) and np.mean(margin) >= uniform:
        return np.maximum(0, np.nextafter(margin.astype(float), -np.inf))
    return np.full(G.shape[0], uniform)


class _Search:
    def __init__(self, problem):
        self.X = X = np.asarray(problem['X'], dtype=float)
        self.y = y = np.asarray(problem['y'], dtype=float)
        self.lam = float(problem['lam0'])
        self.ridge = float(problem['lam2'])
        self.box = float(problem['M'])
        self.p = X.shape[1]
        self.G = X.T @ X
        self.h = X.T @ y
        self.base = .5 * (y @ y)
        self.d = _minorant(X, self.G)
        self.R = self.G - np.diag(self.d)
        self.q = self.d + self.ridge
        self.tau = min(self.box, 1e-8)
        self.breakpoint = np.minimum(self.box, self.tau + np.sqrt(2 * self.lam / self.q))
        width = self.breakpoint - self.tau
        self.slope = np.divide(self.lam, width, out=np.full_like(width, np.inf), where=width > 0)
        self.slope += .5 * self.q * (self.breakpoint + self.tau)
        self.step = 1 / max(np.linalg.eigvalsh(self.R)[-1] * 1.000001, 1e-15)
        self.curv = self.G.diagonal() + self.ridge
        # Finish a strongly correlated pair before opening another one.
        # Correlation affects branch order only, never the proof bound.
        norm = np.sqrt(self.G.diagonal())
        scale = norm[:, None] * norm[None, :]
        corr = np.divide(abs(self.G), scale, out=np.zeros_like(self.G), where=scale > 0)
        np.fill_diagonal(corr, 0.)
        peer = np.argmax(corr, axis=1)
        idx = np.arange(self.p)
        self.pair = np.where((peer[peer] == idx) & (corr[idx, peer] >= .98), peer, -1)
        self.best = np.zeros(self.p)
        self.ub = self.base
        self.clean_best, self.clean_ub = self.best.copy(), self.ub
        self.polished = set()
        self.tol = 9e-6
        self.xl = X.astype(np.longdouble)
        self.yl = y.astype(np.longdouble)

    def _candidate(self, w):
        def objective(v):
            r = self.X @ v - self.y
            return .5 * (r @ r + self.ridge * (v @ v)) + self.lam * np.count_nonzero(abs(v) > self.tau)

        obj = objective(w)
        if obj < self.ub:
            self.best, self.ub = w.copy(), obj
        clean = w.copy()
        clean[abs(clean) <= self.tau] = 0
        obj = objective(clean)
        if obj < self.clean_ub:
            self.clean_best, self.clean_ub = clean, obj
        if obj < self.ub:
            self.best, self.ub = clean, obj

    def _polish(self, support, include_small=False):
        key = support.tobytes(), include_small
        if key in self.polished:
            return
        self.polished.add(key)
        s = np.arange(self.p) if include_small else np.flatnonzero(support)
        if not s.size:
            return
        H = self.G[np.ix_(s, s)] + self.ridge * np.eye(s.size)
        h = self.h[s]
        cap = np.where(support[s], self.box, self.tau)
        z = np.linalg.solve(H, h)
        if np.any(abs(z) > cap):
            from scipy.optimize import lsq_linear
            design = np.vstack((self.X[:, s], np.sqrt(self.ridge) * np.eye(s.size)))
            target = np.r_[self.y, np.zeros(s.size)]
            # Scale the variables so tiny and full boxes have the same bounds.
            z = cap * lsq_linear(design * cap, target, bounds=(-1., 1.), tol=1e-13).x
        w = np.zeros(self.p)
        w[s] = np.clip(z, -cap, cap)
        self._candidate(w)

    def _coordinate_incumbent(self, start, forced_zero=None):
        w = start.copy()
        if forced_zero is not None:
            w[forced_zero] = 0
        res = self.h - self.G @ w
        for _ in range(40):
            change = 0.
            for j in range(self.p):
                if forced_zero is not None and forced_zero[j]:
                    continue
                z = res[j] + self.G[j, j] * w[j]
                v = np.clip(z / self.curv[j], -self.box, self.box)
                if z*v - .5*self.curv[j]*v*v <= self.lam:
                    v = 0.
                delta = v - w[j]
                if delta:
                    w[j] = v
                    res -= delta * self.G[:, j]
                    change = max(change, abs(delta))
            if change < 1e-8:
                break
        self._candidate(w)
        self._polish(w != 0)

    def _bound(self, c, a, free, precise=False):
        if precise:
            c = c.astype(np.longdouble)
            r = self.yl - self.xl @ c
            g = self.xl.T @ r
            dc = self.d * c
            z = abs(g + dc)
            constant = .5 * (r @ r + dc @ c) + g @ c
        else:
            Rc = self.R @ c
            z = abs(self.h - Rc)
            constant = self.base - .5 * (c @ Rc)
        b = np.minimum(z / self.q, self.box)
        on = self.lam - b * (z - .5 * self.q * b)
        # Off bits may use the uncharged interval in the literal objective.
        # This can only decrease the allowed witness value (off=0 there).
        small = np.minimum(b, self.tau)
        off = -small * (z - .5 * self.q * small)
        value = constant + on[a].sum() + off[~(a | free)].sum() + np.minimum(on[free], off[free]).sum()
        if precise:
            # Bound accumulated dot-product and scalar arithmetic error.
            u = np.finfo(c.dtype).eps
            scale = abs(self.yl) + abs(self.xl) @ abs(c)
            scale2 = self.xl.shape[0] + self.p + 8
            mass = (scale @ scale + (abs(self.xl).T @ scale) @ np.full(self.p, self.box)
                    + (abs(dc) @ (abs(c) + self.box)) + self.p * self.lam + z @ b)
            return float(np.nextafter(float(value - 32 * scale2 * u * mass), -np.inf))
        return float(value)

    def _relax(self, zero, active, free, start):
        """Proximal gradient for the separable convex envelopes on free bits."""
        c = np.clip(start, -self.box, self.box).copy()
        c[zero] = np.clip(c[zero], -self.tau, self.tau)
        v = c.copy()
        t = 1.
        q, step = self.q, self.step
        den = 1 + step * q
        threshold = step * self.slope
        lo = self._bound(c, active, free)
        witness = c.copy()
        for k in range(400):
            u = v + step * (self.h - self.R @ v)
            mag = abs(u)
            quad = mag / den
            shrunk = np.maximum(mag - threshold, self.tau)
            mixed = np.where(quad <= self.tau, quad,
                             np.where((shrunk < self.breakpoint) | (self.breakpoint == self.box), shrunk, quad))
            mag = np.where(free, mixed, quad)
            new = np.copysign(np.minimum(mag, np.where(zero, self.tau, self.box)), u)
            if k % 8 == 0 or k == 399:
                lb = self._bound(new, active, free)
                if lb > lo:
                    lo, witness = lb, new.copy()
                if lo >= self.ub - self.tol:
                    break
                abs_new = abs(new)
                penalty = .5 * q * new * new + self.lam * ~zero
                small = free & (abs_new <= self.tau)
                line = free & ~small & (abs_new <= self.breakpoint)
                penalty[small] = .5 * q[small] * new[small]**2
                penalty[line] = .5 * q[line] * self.tau**2 + self.slope[line] * (abs_new[line] - self.tau)
                upper = self.base + new @ (.5 * (self.R @ new) - self.h) + penalty.sum()
                if upper - lo <= 2e-9:
                    break
            next_t = .5 * (1 + np.sqrt(1 + 4*t*t))
            if (v - new) @ (new - c) > 0:
                next_t = 1.
            v = new + ((t - 1) / next_t) * (new - c)
            c, t = new, next_t
        return lo, witness, new

    def run(self):
        self._coordinate_incumbent(self.best)
        self._coordinate_incumbent(np.clip(np.linalg.solve(self.G + self.ridge*np.eye(self.p), self.h), -self.box, self.box))
        empty = np.zeros(self.p, dtype=bool)
        queue = []
        leaves = []
        counter = itertools.count()

        def record(z, a, lb, c):
            leaves.append({'fixed_zero': z, 'fixed_active': a, 'lb': lb,
                           'minorant_diagonal': self.d.copy(), 'minorant_center': c})

        def visit(z, a, start, preferred=-1):
            free = ~(z | a)
            lb = self._bound(start, a, free, precise=True)
            if lb >= self.ub - self.tol:
                record(z, a, lb, start)
                return
            _, witness, x = self._relax(z, a, free, start)
            lb = self._bound(witness, a, free, precise=True)
            if lb < self.ub - self.tol:
                support = ((abs(x) >= .5 * (self.tau + self.breakpoint)) & ~z) | a
                self._polish(support)
                rounded = x.copy()
                rounded[~support] = np.clip(rounded[~support], -self.tau, self.tau)
                self._candidate(rounded)
                if lb < self.ub - self.tol and np.any(abs(rounded[~support]) > 0):
                    self._polish(support, include_small=True)
            if not free.any():
                self._polish(a, include_small=True)
                if lb < self.ub - self.tol:
                    # At a fixed support the polished ridge solution is also a dual center.
                    c = self.best.copy() if np.array_equal(self.best != 0, a) else x
                    lb2 = self._bound(c, a, free, precise=True)
                    if lb2 > lb:
                        lb, witness = lb2, c
            if lb >= self.ub - self.tol or not free.any():
                record(z, a, lb, witness)
            else:
                heapq.heappush(queue, (lb, next(counter), z, a, witness, x, preferred))

        visit(empty, empty.copy(), self.best)
        while queue:
            lb, _, z, a, witness, x, preferred = heapq.heappop(queue)
            if lb >= self.ub - self.tol:
                record(z, a, lb, witness)
                continue
            free = ~(z | a)
            width = self.breakpoint - self.tau
            frac = np.divide(np.maximum(abs(x) - self.tau, 0), width,
                             out=np.zeros_like(x), where=width > 0).clip(0, 1)
            height = self.lam + .5 * self.q * width * (self.breakpoint + self.tau)
            scores = np.where(free, frac * (1-frac) * height, -1)
            j = preferred if preferred >= 0 and free[preferred] else int(np.argmax(scores))
            zz, aa = z.copy(), a.copy()
            zz[j] = aa[j] = True
            if len(leaves) + len(queue) + 2 > 2048:
                raise RuntimeError('Proof search needs more than 2048 subcubes')
            visit(zz, a, witness, self.pair[j])
            visit(z, aa, witness)
        lower = min(self.ub, min(leaf['lb'] for leaf in leaves))
        if self.ub - lower > 1e-5:
            raise RuntimeError('Could not certify the requested objective gap')
        if self.clean_ub - lower <= 1e-5:
            self.best, self.ub = self.clean_best, self.clean_ub
        return {'weights': self.best, 'objective': float(self.ub),
                'certified_lower_bound': float(lower), 'subcube_leaves': leaves}


def solve_proof_carrying_box_l0(problem: dict) -> dict:
    """Return a feasible minimizer and a partition of the binary support cube."""
    return _Search(problem).run()
