"""Reference-panel alignment, robust potency regression, and certified L0 selection."""
import numpy as np

_SUPPORT_EPS = 1e-8


def _align(cohort):
    _, centers = np.unique(cohort['center_id'], return_inverse=True)
    strata = np.asarray(cohort['stratum'])
    raw = np.asarray(cohort['X_raw'], dtype=float)
    controls = np.asarray(cohort['controls'], dtype=float)
    aligned = np.empty_like(raw)
    for k in range(centers.max() + 1):
        rows = centers == k
        panel = controls[rows]
        finite = np.isfinite(panel)
        valid = finite.sum(axis=0) >= 2
        panel = np.where(finite[:, valid], panel[:, valid], np.nan)
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        aligned[rows] = (raw[rows] - offset) / gain
    _, cells, counts = np.unique(np.column_stack((centers, strata)), axis=0,
                                  return_inverse=True, return_counts=True)
    weights = (len(raw) / len(counts)) / counts[cells]
    finite = np.isfinite(aligned)
    means = weights @ np.where(finite, aligned, 0.) / (weights @ finite)
    return np.where(finite, aligned, means), weights


def _potency(aligned, strata, y):
    design = np.column_stack((np.ones(len(y)), strata == 1, strata == 2, aligned))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    for iteration in range(100):
        residual = y - design @ beta
        # Update the scale during initialization, then minimize a fixed Cauchy loss.
        if iteration < 8:
            mad = np.median(np.abs(residual - np.median(residual)))
            scale = max(2.0 * 1.4826 * mad, 1e-10)
        root_weight = 1 / np.hypot(1, residual / scale)
        updated = np.linalg.lstsq(root_weight[:, None] * design,
                                   root_weight * y, rcond=None)[0]
        delta = np.linalg.norm(updated - beta)
        beta = updated
        if iteration >= 8 and delta < 1e-10 * (1 + np.linalg.norm(beta)):
            break
    return beta[3:].copy()


class _BoxSearch:
    def __init__(self, problem):
        self.X = np.asarray(problem['X'], dtype=float)
        self.y = np.asarray(problem['y'], dtype=float)
        self.r = float(problem['lam2'])
        self.k = float(problem['lam0'])
        self.M = float(problem['M'])
        self.e = min(self.M, _SUPPORT_EPS)
        self.G = self.X.T @ self.X
        self.c = self.X.T @ self.y
        self.full_h = self.G.diagonal() + self.r
        self.p = len(self.c)
        scale = max(1., np.max(self.G.diagonal()))
        self.diag = np.full(self.p, max(0., np.linalg.eigvalsh(self.G)[0] - 1e-12 * scale))
        self.best = np.zeros(self.p)
        self.upper = self.objective(self.best)
        self.tol = 5e-6

    def objective(self, w):
        residual = self.X @ w - self.y
        return float(.5 * (residual @ residual + self.r * (w @ w))
                     + self.k * np.count_nonzero(np.abs(w) > _SUPPORT_EPS))

    def improve(self, start, state):
        """Find a box-feasible incumbent by exact one-coordinate updates."""
        w = start.copy()
        off = state == 0
        w[off] = np.clip(w[off], -self.e, self.e)
        residual = self.G @ w - self.c
        for _ in range(100):
            change = 0.
            for j, bit in enumerate(state):
                h = self.full_h[j]
                b = residual[j] - self.G[j, j] * w[j]
                t = abs(b)
                v0 = min(self.e, t / h)
                v = min(self.M, max(self.e, t / h))
                if bit == 0 or (bit < 0 and self.k >= (v - v0) * (t - .5 * h * (v + v0))):
                    v = v0
                if b > 0:
                    v = -v
                delta = v - w[j]
                if delta:
                    w[j] = v
                    residual += delta * self.G[:, j]
                    change = max(change, abs(delta))
            if change < 2e-11:
                break
        obj = self.objective(w)
        if obj < self.upper:
            self.best, self.upper = w, obj

    def relax(self, state, start):
        """Solve a convex envelope relaxation; any iterate gives a valid bound."""
        d = self.diag
        G = self.G
        h = d + self.r
        R = G - np.diag(d)
        rdiag = R.diagonal()
        # The free-coordinate envelope follows the inactive quadratic to e,
        # then joins the active quadratic by a line tangent at pivot (or ending at M).
        e = self.e
        pivot = np.minimum(self.M, e + np.sqrt(2 * self.k / h))
        if e < self.M and np.all(pivot > e):
            slope = self.k / (pivot - e) + .5 * h * (pivot + e)
        else:
            slope = np.full(self.p, np.inf)
        a = start.copy()
        off = state == 0
        a[off] = np.clip(a[off], -e, e)
        grad = R @ a - self.c
        for _ in range(170):
            change = 0.
            for j, bit in enumerate(state):
                b = grad[j] - rdiag[j] * a[j]
                t = abs(b)
                v = t / self.full_h[j]
                if bit == 0 or (bit < 0 and v <= e):
                    v = min(e, v)
                elif bit == 1:
                    v = min(self.M, v)
                elif t <= slope[j] + rdiag[j] * e:
                    v = e
                elif t < slope[j] + rdiag[j] * pivot[j]:
                    v = (t - slope[j]) / rdiag[j]
                else:
                    v = min(self.M, v)
                if b > 0:
                    v = -v
                delta = v - a[j]
                if delta:
                    a[j] = v
                    grad += delta * R[:, j]
                    change = max(change, abs(delta))
            if change < 2e-10:
                break
        residual = self.X @ a - self.y
        loss_grad = self.X.T @ residual
        b = loss_grad - d * a
        constant = .5 * (residual @ residual + (d * a) @ a) - loss_grad @ a
        v = np.clip(-b / h, -self.M, self.M)
        active = .5 * h * v * v + b * v + self.k
        # A zero support bit includes the entire unpenalized interval, not
        # just the origin. For active bits, relaxing the interval exclusion
        # in the quadratic minimum gives a valid lower bound.
        z = np.clip(-b / h, -self.e, self.e)
        inactive = .5 * h * z * z + b * z
        lb = (constant + inactive[state == 0].sum() + active[state == 1].sum()
              + np.minimum(inactive[state < 0], active[state < 0]).sum())
        # Slight downward rounding allowance for cancellation in the expansion.
        magnitude = (1. + residual @ residual + d @ (a * a)
                     + np.abs(loss_grad) @ np.abs(a) + np.abs(b) @ np.abs(v)
                     + h @ (v * v) + self.p * self.k)
        lb -= 64 * np.finfo(float).eps * magnitude
        return float(lb), a, pivot, active

    def search(self):
        root = np.full(self.p, -1, dtype=np.int8)
        self.improve(self.best, root)
        self.improve(np.clip(np.linalg.solve(self.G + self.r * np.eye(self.p), self.c),
                             -self.M, self.M), root)
        leaves = []
        stack = [(root, self.best)]
        while stack:
            state, start = stack.pop()
            lb, a, pivot, active = self.relax(state, start)
            free = np.flatnonzero(state < 0)
            if lb < self.upper - self.tol:
                self.improve(a, state)
            if lb >= self.upper - self.tol or not len(free):
                leaves.append({'fixed_zero': np.flatnonzero(state == 0).tolist(),
                               'fixed_active': np.flatnonzero(state == 1).tolist(),
                               'lb': lb, 'minorant_diagonal': self.diag.copy(),
                               'minorant_center': a})
                continue
            frac = np.minimum(1., np.abs(a[free]) / pivot[free])
            j = free[np.argmax(frac * (1. - frac) + 1e-7 / (1. + np.abs(active[free])))]
            preferred = int(frac[np.argmax(free == j)] >= .5)
            for value in (1 - preferred, preferred):
                child = state.copy()
                child[j] = value
                stack.append((child, a))
        return {'weights': self.best, 'objective': self.upper,
                'certified_lower_bound': min(leaf['lb'] for leaf in leaves),
                'subcube_leaves': leaves}


def solve_multicenter_certified_l0(study: dict) -> dict:
    cohort = study['cohort']
    aligned, _ = _align(cohort)
    beta = _potency(aligned, np.asarray(cohort['stratum']), np.asarray(cohort['y_potency']))
    return {'aligned_biomarkers': aligned, 'robust_potency_weights': beta,
            'certified_selection': _BoxSearch(study['sparse_selection']).search()}
