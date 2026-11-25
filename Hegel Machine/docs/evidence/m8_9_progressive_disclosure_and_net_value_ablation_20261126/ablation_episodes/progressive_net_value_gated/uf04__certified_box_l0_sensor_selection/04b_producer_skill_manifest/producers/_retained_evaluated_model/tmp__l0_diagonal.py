"""Best-bound search for thresholded, boxed sparse ridge regression."""
import heapq
import numpy as np
from scipy.linalg import eigvalsh

_TOL = 1e-9
_GAP = 1e-5
_LIMIT = 600


class _Search:
    def __init__(self, instance):
        self.X = np.asarray(instance['X'], dtype=float)
        self.y = np.asarray(instance['y'], dtype=float)
        self.ridge = float(instance['ridge'])
        self.m = float(instance['box'])
        self.lam = float(instance['l0_penalty'])
        self.t = min(_TOL, self.m)
        self.n, self.p = self.X.shape
        self.G = self.X.T @ self.X + self.ridge * np.eye(self.p)
        self.abs_gram = np.abs(self.X).T @ np.abs(self.X) + self.ridge*np.eye(self.p)
        self.c = self.X.T @ self.y
        self.y2 = 0.5 * (self.y @ self.y)
        # Allow for cancellation when evaluating the supporting quadratic.
        self.err = 16 * np.finfo(float).eps * (self.n + self.p + 4) * (
            1 + self.y2 + self.m * np.sum(np.abs(self.c)) +
            self.m**2 * (np.linalg.norm(self.X, 'fro')**2 * self.p + self.ridge*self.p)
            + self.p * self.lam)
        self.best = np.inf
        self.w = np.zeros(self.p)
        self.nodes = 0
        self.full_diag = self._minorant(np.arange(self.p)) if self.p else self.w

    def _minorant(self, ids):
        G = self.G[np.ix_(ids, ids)]
        h = np.diag(G).copy()
        q = np.sqrt(h)
        gram = G / q[:, None] / q
        off = np.abs(gram - np.eye(len(ids)))
        # Spectral and row-sum estimates propose curvature; the Cholesky
        # reconstruction and a rounding margin certify it independently.
        g = np.maximum(0., 1. - off.sum(axis=1))
        e = max(0., float(eigvalsh(gram, subset_by_index=[0, 0])[0]))
        scale = self.abs_gram[np.ix_(ids, ids)].sum(axis=1).max()
        pad = 64*np.finfo(float).eps*(self.n+self.p+4)*scale
        cushion = pad/h
        eye = np.eye(len(ids))
        for estimate in (0.5*g + 0.5*e, e):
            d = h * np.maximum(0., estimate-4*cushion)
            if not np.any(d):
                continue
            shifted = G-np.diag(d)-pad*eye
            try:
                L = np.linalg.cholesky(shifted)
            except np.linalg.LinAlgError:
                continue
            if np.abs(shifted-L@L.T).sum(axis=1).max() <= pad/2:
                return d
        return np.zeros(len(ids))

    def _accept(self, w):
        r = self.X @ w - self.y
        value = 0.5*(r@r + self.ridge*(w@w)) + self.lam*np.count_nonzero(np.abs(w) > _TOL)
        if value < self.best:
            self.best, self.w = float(value), w.copy()

    def _polish(self, support, start):
        bounds = np.where(support, self.m, self.t)
        g = self.G
        w = np.clip(start, -bounds, bounds)
        for _ in range(100):
            old = w.copy()
            for j in range(self.p):
                w[j] = np.clip(w[j] + (self.c[j] - g[j] @ w)/g[j,j], -bounds[j], bounds[j])
            if np.max(np.abs(w-old)) <= 2e-12:
                break
        self._accept(w)

    def _local(self, start):
        w = start.copy()
        h = self.G.diagonal()
        for _ in range(75):
            old = w.copy()
            for j in range(self.p):
                a = abs(w[j]*h[j] + self.c[j] - self.G[j]@w)
                u = min(a/h[j], self.m)
                v = min(a/h[j], self.t)
                b = u if (u-v)*(a-0.5*h[j]*(u+v)) > self.lam else v
                w[j] = np.copysign(b, w[j]*h[j] + self.c[j] - self.G[j]@w)
            if np.max(np.abs(w-old)) <= 1e-10:
                break
        self._accept(w)
        self._polish(np.abs(w)>self.t, w)

    def _relax(self, state, start):
        free = state == 0
        on = state == 1
        off = state == -1
        ids = np.flatnonzero(~off)
        # Keep a globally valid split; a face-restricted split may sharpen it
        # when inactive intervals have already been fixed (handled below).
        d = self.full_diag.copy()
        if ids.size and ids.size != self.p:
            face = self._minorant(ids)
            # Difference between the face tangent and the full tangent is
            # controlled by a bounded cross term involving inactive variables.
            if face[free[ids]].sum() > d[free].sum()*1.05:
                d[ids] = face
                d[off] = 0.
                face_split = True
            else:
                face_split = False
        else:
            face_split = False
        H = self.G - np.diag(d)
        diag = H.diagonal()
        a = np.maximum(d, np.finfo(float).tiny)
        # Contact of the outer quadratic with the line from the inactive kink.
        b = np.minimum(self.m, np.hypot(self.t, np.sqrt(2*self.lam/a)))
        slope = self.lam/np.maximum(b-self.t, np.finfo(float).tiny) + 0.5*d*(b+self.t)
        bound = np.where(off, self.t, self.m)
        w = np.clip(start, -bound, bound)
        grad = H @ w - self.c
        for sweep in range(260):
            delta = 0.
            for j in range(self.p):
                f = diag[j]*w[j]-grad[j]
                af = abs(f)
                h = self.G[j,j]
                if not free[j]:
                    z = min(af/h, bound[j])
                elif af <= h*self.t:
                    z = af/h
                elif af <= slope[j] + diag[j]*self.t:
                    z = self.t
                elif af < slope[j] + diag[j]*b[j]:
                    z = (af-slope[j])/diag[j]
                else:
                    z = min(self.m, max(b[j], af/h))
                step = np.copysign(z, f)-w[j]
                if step:
                    w[j] += step
                    grad += step*H[:,j]
                    delta = max(delta, abs(step))
            if delta <= 2e-11*(1+self.m):
                break
        grad = H@w-self.c
        g = np.abs(grad)
        # Exact minima of the separated quadratics plus the binary charge.
        v = np.minimum(self.t, g/a)
        u = np.minimum(self.m, g/a)
        lo0 = v*(0.5*d*v-g)
        lo1 = u*(0.5*d*u-g)+self.lam
        terms = np.where(off, lo0, np.where(on, lo1, np.minimum(lo0, lo1)))
        lower = self.y2 - 0.5*(w@(H@w)) + terms.sum() - self.err
        if face_split:
            # H on the face and on the inactive block is PSD. Only their
            # cross term can be negative. Bound it for all feasible x-w.
            Hcross = np.abs(H[np.ix_(off, ~off)])
            lower -= (self.t+np.abs(w[off])) @ Hcross @ (self.m+np.abs(w[~off]))
        self._accept(w)
        if self.nodes <= 30 or not np.any(free):
            for factor in (0.25, 0.8):
                support = on | (free & (np.abs(w) >= np.maximum(2*self.t, factor*b)))
                self._polish(support, w)
        # Branch on the largest discrepancy between original and relaxed cost.
        z = np.abs(w)
        envelope = np.where(z <= self.t, 0.5*d*z*z,
                      np.where(z < b, 0.5*d*self.t**2 + slope*(z-self.t), 0.5*d*z*z+self.lam))
        off_loss = np.maximum(0., z-self.t)*np.maximum(g, self.G.diagonal()*np.maximum(0., z-self.t)/2)
        on_loss = np.maximum(0., .5*d*z*z+self.lam-envelope)
        score = np.minimum(on_loss, off_loss) + 0.15*on_loss*np.minimum(1., z/np.maximum(b, 1e-300))
        branch = int(np.argmax(np.where(free, score, -np.inf))) if np.any(free) else -1
        return float(lower), branch, w

    def solve(self):
        self._accept(self.w)
        if not self.p:
            return self.result(max(0., self.best-self.err))
        if self.m <= _TOL:
            self._polish(np.ones(self.p, dtype=bool), self.w)
            grad = self.G@self.w-self.c
            self.nodes = 1
            return self.result(max(0., self.best + np.sum(-self.m*np.abs(grad)-grad*self.w)-self.err))
        start = np.clip(np.linalg.solve(self.G, self.c), -self.m, self.m)
        self._local(start)
        self._local(np.zeros(self.p))
        queue = []
        closed = np.inf
        def add(state, w, parent):
            nonlocal closed
            self.nodes += 1
            lb, j, w = self._relax(state, w)
            lb = max(parent, lb)
            if lb >= self.best-_GAP:
                closed = min(closed, lb)
            elif j >= 0:
                heapq.heappush(queue, (lb, self.nodes, j, state, w))
            else:
                closed = min(closed, lb)
        add(np.zeros(self.p, dtype=np.int8), start, 0.)
        while queue and self.best-min(queue[0][0],closed) > _GAP and self.nodes+2 <= _LIMIT:
            lb, _, j, state, w = heapq.heappop(queue)
            if lb >= self.best-_GAP:
                closed = min(closed, lb)
                continue
            for value in (-1, 1):
                child = state.copy(); child[j] = value
                add(child, w, lb)
        return self.result(min(closed, queue[0][0] if queue else np.inf))

    def result(self, lower):
        return {'weights': self.w.tolist(), 'objective': self.best,
                'certified_lower_bound': float(min(self.best, lower)),
                'nodes_explored': self.nodes}


def solve_certified_box_l0(instance: dict) -> dict:
    return _Search(instance).solve()
