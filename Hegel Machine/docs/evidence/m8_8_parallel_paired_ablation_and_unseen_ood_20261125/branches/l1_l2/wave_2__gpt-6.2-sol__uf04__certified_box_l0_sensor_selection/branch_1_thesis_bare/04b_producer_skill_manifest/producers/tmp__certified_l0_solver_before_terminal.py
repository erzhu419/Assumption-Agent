"""Best-bound search with diagonal-perspective dual certificates."""
import heapq
import itertools

import numpy as np
from scipy.optimize import lsq_linear


_EPS = 1e-9
_TOL = 9e-6


class _Problem:
    def __init__(self, instance):
        self.X = np.asarray(instance['X'], dtype=float)
        self.y = np.asarray(instance['y'], dtype=float)
        self.ridge = float(instance['ridge'])
        self.box = float(instance['box'])
        self.lam = float(instance['l0_penalty'])
        self.n, self.p = self.X.shape
        self.e = min(_EPS, self.box)
        self.I = np.arange(self.p)
        # Accumulate certificate data in extended precision. The error budget
        # below uses absolute products, not the possibly cancelled Gram terms.
        X = self.X.astype(np.longdouble)
        y = self.y.astype(np.longdouble)
        self.Hx = X.T @ X + np.longdouble(self.ridge) * np.eye(self.p, dtype=np.longdouble)
        self.gx = X.T @ y
        self.cx = (y @ y) / 2
        self.H = self.Hx.astype(float)
        self.g = self.gx.astype(float)
        self.c = float(self.cx)
        u = np.finfo(np.longdouble).eps
        a = np.abs(y) + self.box * np.sum(np.abs(X), axis=1)
        self.roundoff = float(128 * (self.n + self.p + 4) * u * (
            1 + a @ a + self.p * (self.ridge * np.longdouble(self.box)**2 + self.lam)))
        self.gram_error = float(8 * (self.n + 2) * u * np.sum(np.abs(X).T @ np.abs(X)))
        self.design = np.vstack((self.X, np.sqrt(self.ridge) * np.eye(self.p)))
        self.target = np.concatenate((self.y, np.zeros(self.p)))
        self.best = np.inf
        self.w = np.zeros(self.p)
        self.cache = {}

    def objective(self, w):
        r = self.X @ w - self.y
        return float(0.5 * (r @ r + self.ridge * (w @ w)) + self.lam * np.count_nonzero(np.abs(w) > _EPS))

    def fit(self, paid):
        key = tuple(np.flatnonzero(paid))
        if key not in self.cache:
            bounds = np.where(paid, self.box, self.e)
            w = np.zeros(self.p)
            J = np.flatnonzero(paid)
            if len(J):
                w[J] = np.linalg.solve(self.H[np.ix_(J, J)], self.g[J])
            # Pin inactive coordinates to the appropriate side of the free
            # interval, then solve once on the resulting face. If that face
            # fails the box KKT conditions, let BVLS find the correct face.
            F = ~paid
            w[F] = -bounds[F] * np.sign((self.H @ w - self.g)[F])
            if len(J) and np.any(F):
                w[J] = np.linalg.solve(self.H[np.ix_(J, J)], self.g[J] - self.H[np.ix_(J, F)] @ w[F])
            grad = self.H @ w - self.g
            violation = np.where(w <= -bounds, np.minimum(grad, 0.),
                                 np.where(w >= bounds, np.maximum(grad, 0.), grad))
            if np.any(np.abs(w) > bounds) or np.max(np.abs(violation), initial=0.) > 1e-10:
                w = lsq_linear(self.design, self.target, bounds=(-bounds, bounds),
                               method='bvls', tol=1e-12).x
            w = np.clip(w, -bounds, bounds)
            value = self.objective(w)
            self.cache[key] = value, w
            if value < self.best:
                self.best, self.w = value, w
        return self.cache[key]

    def improve(self, paid, swaps=False):
        paid = paid.copy()
        value, _ = self.fit(paid)
        while True:
            vbest, pbest = value, None
            for j in self.I:
                q = paid.copy()
                q[j] = not q[j]
                v, _ = self.fit(q)
                if v < vbest - 1e-10:
                    vbest, pbest = v, q
            if pbest is None and swaps:
                for j in self.I[paid]:
                    for k in self.I[~paid]:
                        q = paid.copy()
                        q[j], q[k] = False, True
                        v, _ = self.fit(q)
                        if v < vbest - 1e-10:
                            vbest, pbest = v, q
            if pbest is None:
                return
            paid, value = pbest, vbest

    def split(self, H, undecided, weights=None):
        """A heuristic diagonal allocation, followed by a PSD check."""
        diag = np.diag(H)
        d = np.zeros(len(diag))
        if not np.any(undecided):
            return d, self.ridge / 2
        # Remove the curvature already belonging to the mandatory columns
        # by using their Schur complement when allocating the diagonal.
        U = undecided
        B = H[np.ix_(U, U)].copy()
        gamma = max(self.ridge * .005, 1e-12 * np.max(diag))
        if np.any(~U):
            C = H[np.ix_(U, ~U)]
            B -= C @ np.linalg.solve(H[np.ix_(~U, ~U)], C.T)
        scales = np.sqrt(np.maximum(np.diag(B), self.ridge))
        M = B / scales[:, None] / scales
        alpha = max(0., np.linalg.eigvalsh(M)[0])
        diag_u = np.full(len(B), .78 * alpha)
        R = M - np.diag(diag_u)
        # Coordinate allocations exploit features with unusually large
        # conditional curvature. Maintaining R positive avoids overshoot.
        order = np.argsort(np.diag(np.linalg.inv(R))) if alpha > 1e-12 else np.arange(len(B))
        if weights is not None:
            order = np.argsort(-weights[U], kind='stable')
        if alpha > 1e-12:
            for j in order:
                inc = .75 / np.linalg.inv(R)[j, j]
                R[j, j] -= inc
                diag_u[j] += inc
        d[U] = diag_u * scales**2
        # Prove positivity of H - diag(d) - gamma*I by reconstructing its
        # Cholesky factor: a positive factor alone is not an error bound.
        # The inverse triangular factor is also checked by its residual.
        for _ in range(8):
            A = H - np.diag(d + gamma)
            try:
                L = np.linalg.cholesky(A)
                K = np.linalg.inv(L).astype(np.longdouble)
                L = L.astype(np.longdouble)
                err = np.sum(np.abs(L @ K - np.eye(len(L))))
                err += 16 * len(L) * np.finfo(np.longdouble).eps * (np.sum(np.abs(L)) * np.sum(np.abs(K)) + len(L))
                residual = np.sum(np.abs(A.astype(np.longdouble) - L @ L.T))
                residual += self.gram_error + 8 * np.finfo(float).eps * (np.sum(np.abs(H)) + np.sum(d) + len(L)*gamma)
                residual += 16 * len(L) * np.finfo(np.longdouble).eps * np.sum(np.abs(L))**2
                if err < .5 and residual * np.sum(K * K) < .24:
                    return d, gamma
            except np.linalg.LinAlgError:
                pass
            d *= .5
        # The original ridge supplies a known spectral lower bound.
        d[U] = self.ridge / 2
        return d, self.ridge / 2

    def dual(self, S, off, on, d, gamma, x):
        """Evaluate a feasible affine-minorant dual, independent of convergence."""
        LD = np.longdouble
        x = x.astype(LD)
        d = d.astype(LD)
        e, box, lam = LD(self.e), LD(self.box), LD(self.lam)
        h = self.Hx[np.ix_(S, S)]
        rx = h @ x - d * x
        s = np.abs(self.gx[S] - rx)
        def conjugate(B):
            t = B * np.ones_like(d)
            np.divide(s, d, out=t, where=d > 0)
            np.minimum(t, B, out=t)
            return t * (s - d * t / 2)
        a, b = conjugate(e), conjugate(box) - lam
        lower = self.cx - x @ rx / 2 - np.sum(np.where(on, b, np.maximum(a, b)))
        if np.any(off) and e > 0:
            C = self.Hx[np.ix_(off, S)]
            lower -= e * np.sum(np.abs(C @ x - self.gx[off]))
            # Complete the square against the reserved gamma*I curvature.
            c = np.sum(np.abs(C), axis=0) + self.gram_error
            lower -= e**2 * (c @ c) / (2 * LD(gamma))
        return float(np.nextafter(float(lower - self.roundoff), -np.inf))

    def relax(self, status, warm, cutoff, weights=None):
        S = status >= 0
        off = ~S
        on = status[S] == 1
        U = ~on
        H = self.H[np.ix_(S, S)]
        g = self.g[S]
        if not len(g):
            self.fit(status == 1)
            return self.dual(S, off, on, np.zeros(0), self.ridge, np.zeros(0)), np.zeros(self.p), np.zeros(self.p)
        d, gamma = self.split(H, U, weights)
        R = H - np.diag(d)
        B, e, lam = self.box, self.e, self.lam
        L = max(np.linalg.eigvalsh(R)[-1], self.ridge)
        if not np.any(U):
            _, w = self.fit(status == 1)
            return self.dual(S, off, on, d, gamma, w[S]), w, np.zeros(self.p)
        b = np.minimum(B, e + np.sqrt(2 * lam / d[U]))
        m = lam / (b - e) + .5 * d[U] * (b + e)
        # Prox of the perspective envelope: quadratic, connecting chord,
        # then quadratic again. Paid coordinates have a simple box prox.
        den = 1 + d / L
        def prox(z):
            a = np.abs(z[U]); t = a - m / L
            v = np.where(t < e, np.minimum(e, a / den[U]),
                         np.where(t <= b, t, np.maximum(b, a / den[U])))
            z[U] = np.sign(z[U]) * v
            return np.clip(z, -B, B)
        x = warm[S].copy()
        z = x.copy()
        acc = 1.
        low = -np.inf
        for k in range(220):
            new = prox(z - (R @ z - g) / L)
            move = new - x
            nacc = (1 + np.sqrt(1 + 4 * acc**2)) / 2
            if (z - new) @ move > 0:
                nacc = 1.
            z = new + ((acc - 1) / nacc if nacc > 1 else 0.) * move
            x, acc = new, nacc
            if k % 12 == 0 or k == 219:
                lb = self.dual(S, off, on, d, gamma, x)
                low = max(low, lb)
                if low >= cutoff or np.max(np.abs(move)) < 1e-10:
                    break
        w = np.zeros(self.p)
        w[S] = x
        frac = np.zeros(self.p)
        frac[S] = np.where(on, 1., 0.)
        frac[np.flatnonzero(S)[U]] = np.clip((np.abs(x[U]) - e) / (b - e), 0., 1.)
        return low, w, frac


def solve_certified_box_l0(instance: dict) -> dict:
    """Minimize the stated box-L0 objective and return its global dual bound."""
    P = _Problem(instance)
    status = np.zeros(P.p, dtype=np.int8)  # 0 undecided, -1 free, +1 paid
    paid = np.ones(P.p, dtype=bool)
    P.fit(~paid)
    if P.box <= _EPS or P.p == 0:
        _, w = P.fit(paid)
        lo = P.dual(paid, ~paid, ~paid, np.zeros(P.p), P.ridge, w)
        return dict(weights=P.w.tolist(), objective=P.best, certified_lower_bound=lo, nodes_explored=1)
    _, w = P.fit(paid)
    P.improve(paid, swaps=True)
    serial = itertools.count()
    heap = []
    closed = np.inf
    nodes = 0

    def enqueue(st, warm, inherited):
        nonlocal nodes, closed
        nodes += 1
        lb, w, frac = P.relax(st, warm, P.best - _TOL)
        lb = max(lb, inherited)
        if lb < P.best - _TOL:
            U = st == 0
            if np.any(U):
                for threshold in (1e-5, .25, .55, .9):
                    P.fit((st == 1) | (U & (frac > threshold)))
                if lb < P.best - _TOL:
                    score = np.where(U, np.minimum(frac, 1 - frac) + 1e-4 * frac, -1)
                    j = int(np.argmax(score))
                    heapq.heappush(heap, (lb, next(serial), st, w, j))
                    return
        closed = min(closed, lb)

    enqueue(status, w, -np.inf)
    while heap and heap[0][0] < P.best - _TOL and nodes + 2 <= 600:
        lb, _, st, w, j = heapq.heappop(heap)
        for v in (-1, 1):
            child = st.copy()
            child[j] = v
            enqueue(child, w, lb)
    lower = min(closed, heap[0][0] if heap else np.inf, P.best)
    return dict(weights=P.w.tolist(), objective=P.best, certified_lower_bound=float(lower), nodes_explored=nodes)
