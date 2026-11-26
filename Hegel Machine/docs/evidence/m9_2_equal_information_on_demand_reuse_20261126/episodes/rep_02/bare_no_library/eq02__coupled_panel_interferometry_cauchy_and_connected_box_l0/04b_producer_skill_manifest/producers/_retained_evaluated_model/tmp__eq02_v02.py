"""Reference-panel calibration and certified symmetric-box L0 selection."""

import heapq

import numpy as np
from scipy.linalg import solve as linear_solve
from scipy.optimize import minimize


def _align(panel):
    x = np.array(panel['X_panel_raw'], dtype=np.float64, copy=True)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    centers = np.asarray(panel['center_id'])
    strata = np.asarray(panel['stratum'])
    cell_size = np.empty(len(x), dtype=np.float64)
    for center in np.unique(centers):
        mask = centers == center
        c = controls[mask]
        valid = np.isfinite(c)
        cols = valid.sum(axis=0) >= 2
        c = np.where(valid[:, cols], c[:, cols], np.nan)
        offset = np.median(np.nanmean(c, axis=0))
        gain = np.median(np.nanstd(c, axis=0, ddof=0))
        x[mask] = (x[mask] - offset) / gain
        for stratum in (0, 1, 2):
            cell = mask & (strata == stratum)
            cell_size[cell] = cell.sum()
    weight = 1 / cell_size
    weight /= weight.mean()
    finite = np.isfinite(x)
    means = (weight @ np.where(finite, x, 0)) / (weight @ finite)
    return np.where(finite, x, means)


def _calibrate(h, y):
    beta = np.linalg.lstsq(h, y, rcond=None)[0]
    # An L1 start keeps large calibration glitches from setting the Cauchy scale.
    for _ in range(35):
        r = y - h @ beta
        sw = (r * r + 1e-6) ** -0.25
        nxt = np.linalg.lstsq(h * sw[:, None], y * sw, rcond=None)[0]
        if np.linalg.norm(nxt - beta) <= 1e-8 * (1 + np.linalg.norm(beta)):
            beta = nxt
            break
        beta = nxt
    r = y - h @ beta
    scale = max(1.5 * 1.4826 * np.median(np.abs(r - np.median(r))), 1e-7)
    for _ in range(100):
        sw = 1 / np.hypot(1, (y - h @ beta) / scale)
        nxt = np.linalg.lstsq(h * sw[:, None], y * sw, rcond=None)[0]
        if np.linalg.norm(nxt - beta) <= 1e-10 * (1 + np.linalg.norm(beta)):
            return nxt
        beta = nxt
    return beta


def _diagonal(g):
    # Begin with a scale-invariant eigenvalue bound, then use the remaining
    # PSD budget to increase individual coordinates via rank-one downdates.
    norm = np.sqrt(np.maximum(g.diagonal(), np.finfo(float).tiny))
    scaled = g / norm[:, None] / norm
    smallest = np.linalg.eigvalsh(scaled)[0]
    if smallest <= 1e-10:
        return np.zeros(len(g))
    lo = 0.45 * smallest
    r = scaled - lo * np.eye(len(g))
    inv = np.linalg.inv(r)
    d = np.full(len(g), lo)
    for _ in range(4):
        for j in np.argsort(-1 / inv.diagonal()):
            v = inv[:, j].copy()
            inc = 0.38 / inv[j, j]
            d[j] += inc
            inv += (inc / (1 - inc * v[j])) * np.outer(v, v)
    d *= norm**2
    # Strict slack for the certificate's PSD check.
    return 0.999999 * d


class _Selector:
    def __init__(self, x, y, lam2, lam0, box):
        self.x, self.y = x, y
        self.lam2, self.lam0, self.box = lam2, lam0, box
        self.p = x.shape[1]
        self.g = x.T @ x
        self.b = x.T @ y
        self.const = 0.5 * (y @ y)
        self.d = _diagonal(self.g)
        self.q = self.g - np.diag(self.d)
        self.a = self.d + lam2
        self.ridge = self.g + lam2 * np.eye(self.p)
        self.slope = np.full(self.p, lam0 / box)
        self.knee = np.full(self.p, box)
        positive = self.a > 0
        self.knee[positive] = np.minimum(box, np.sqrt(2 * lam0 / self.a[positive]))
        if lam0 > 0:
            self.slope = lam0 / self.knee + 0.5 * self.a * self.knee
        else:
            self.slope.fill(0)
        self.best = self.const
        self.w = np.zeros(self.p)
        self.fits = {}
        self.tol = 5e-6
        self.pad = min(1e-7, 32 * np.finfo(float).eps * (1 + self.const + box * np.linalg.norm(self.b, 1) + box**2 * np.sum(np.abs(self.g))))

    def _objective(self, w):
        r = self.x @ w - self.y
        return float(0.5 * (r @ r + self.lam2 * (w @ w)) + self.lam0 * np.count_nonzero(np.abs(w) > 1e-8))

    def _fit_support(self, active):
        key = tuple(np.flatnonzero(active))
        if not key:
            return np.zeros(self.p)
        if key in self.fits:
            return self.fits[key]
        jj = np.array(key)
        a = self.ridge[np.ix_(jj, jj)]
        b = self.b[jj]
        try:
            t = linear_solve(a, b, assume_a='pos', check_finite=False)
        except np.linalg.LinAlgError:
            t = np.linalg.lstsq(a, b, rcond=None)[0]
        box = self.box
        if np.any(np.abs(t) > box):
            def loss(v):
                av = a @ v
                return 0.5 * v @ av - b @ v, av - b
            t = minimize(loss, np.clip(t, -box, box), method='L-BFGS-B', jac=True,
                         bounds=[(-box, box)] * len(t),
                         options={'ftol': 1e-15, 'gtol': 2e-11, 'maxiter': 240}).x
            # Polish the free coordinates after identifying saturated ones.
            free = (np.abs(t) < box - 1e-9) | (t * (a @ t - b) > 0)
            if free.any():
                try:
                    u = linear_solve(a[np.ix_(free, free)], b[free] - a[np.ix_(free, ~free)] @ t[~free],
                                     assume_a='pos', check_finite=False)
                    if np.all(np.abs(u) <= box):
                        t[free] = u
                except np.linalg.LinAlgError:
                    pass
        w = np.zeros(self.p)
        w[jj] = t
        self.fits[key] = w
        value = self._objective(w)
        if value < self.best:
            self.best, self.w = value, w
        return w

    def _bound(self, c, state):
        qc = self.q @ c
        lin = qc - self.b
        t = np.clip(np.divide(-lin, self.a, out=-np.sign(lin) * self.box,
                              where=self.a > 0), -self.box, self.box)
        v = self.lam0 + t * (lin + 0.5 * self.a * t)
        # An inactive coefficient pays no penalty anywhere in the threshold band.
        tiny = min(self.box, 1e-8)
        z = np.clip(t, -tiny, tiny)
        zero_cost = z * (lin + 0.5 * self.a * z)
        fixed_zero = state == 0
        v[fixed_zero] = zero_cost[fixed_zero]
        free = state == -1
        v[free] = np.minimum(v[free], zero_cost[free])
        return float(self.const - 0.5 * (c @ qc) + v.sum() - self.pad)

    def _node(self, state, start):
        free = state == -1
        active = state == 1
        allowed = np.flatnonzero(state != 0)
        c = start.copy()
        c[state == 0] = 0
        if not free.any():
            w = self._fit_support(active)
            return self._bound(w, state), w, np.zeros(self.p)
        q, a, b, box = self.q, self.a, self.b, self.box
        slope, knee = self.slope, self.knee
        denom = q.diagonal() + a
        qdiag = q.diagonal()
        qcols = np.ascontiguousarray(q.T)
        residual = b - q @ c
        best_lb = self._bound(c, state)
        best_c = c.copy()
        sweeps = 130 if len(allowed) == self.p else 85
        for sweep in range(sweeps):
            largest = 0.0
            for j in allowed:
                old = c[j]
                rhs = residual[j] + qdiag[j] * old
                if active[j]:
                    new = rhs / denom[j] if denom[j] > 0 else np.sign(rhs) * box
                else:
                    mag = abs(rhs)
                    if mag <= slope[j]:
                        new = 0.0
                    elif mag <= qdiag[j] * knee[j] + slope[j] and qdiag[j] > 0:
                        new = np.copysign((mag - slope[j]) / qdiag[j], rhs)
                    else:
                        new = rhs / denom[j] if denom[j] > 0 else np.sign(rhs) * box
                new = min(box, max(-box, new))
                delta = new - old
                if delta:
                    c[j] = new
                    residual -= delta * qcols[j]
                    largest = max(largest, abs(delta))
            if sweep % 5 == 4 or largest < 2e-10 or sweep + 1 == sweeps:
                lb = self._bound(c, state)
                if lb > best_lb:
                    best_lb, best_c = lb, c.copy()
                if best_lb >= self.best - self.tol or largest < 2e-10:
                    break
        if best_lb < self.best - self.tol:
            frac = np.divide(np.abs(c), knee, out=np.zeros(self.p), where=knee > 0)
            for threshold in (0.5, 0.12, 0.88):
                self._fit_support(active | (free & (frac >= threshold)))
        else:
            frac = np.zeros(self.p)
        return best_lb, best_c, frac

    def _leaf(self, state, lb, center):
        return {'fixed_zero': (state == 0).tolist(), 'fixed_active': (state == 1).tolist(),
                'lb': lb, 'minorant_diagonal': self.d.tolist(), 'minorant_center': center.tolist()}

    def solve(self):
        root = np.full(self.p, -1, dtype=np.int8)
        lb, center, frac = self._node(root, self.w)
        # Improve the initial incumbent by true one-flip box-ridge refits.
        for _ in range(4):
            previous = self.best
            support = np.abs(self.w) > 1e-8
            for j in range(self.p):
                neighbor = support.copy()
                neighbor[j] = not neighbor[j]
                self._fit_support(neighbor)
            if self.best >= previous - 1e-10:
                break
        heap = [(lb, 0, root, center, frac)]
        leaves = []
        serial = 0
        while heap:
            lb, _, state, center, frac = heapq.heappop(heap)
            free = state == -1
            if lb >= self.best - self.tol:
                leaves.append(self._leaf(state, lb, center))
                continue
            if not free.any():
                raise RuntimeError('Optimality gap remains on a fully fixed support')
            score = np.where(free, np.minimum(frac, 1 - frac), -np.inf)
            j = int(np.argmax(score))
            if score[j] <= 0:
                j = int(np.argmax(np.where(free, np.abs(center), -np.inf)))
            for value in (int(frac[j] >= 0.5), int(frac[j] < 0.5)):
                child = state.copy()
                child[j] = value
                child_lb, child_c, child_frac = self._node(child, center)
                if child_lb >= self.best - self.tol:
                    leaves.append(self._leaf(child, child_lb, child_c))
                else:
                    serial += 1
                    heapq.heappush(heap, (child_lb, serial, child, child_c, child_frac))
        if len(leaves) > 2048:
            raise RuntimeError(f'Certificate requires {len(leaves)} leaves')
        lower = min(leaf['lb'] for leaf in leaves)
        return {'weights': self.w.tolist(), 'objective': self.best,
                'certified_lower_bound': min(lower, self.best), 'subcube_leaves': leaves}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    """Align the panel, remove calibration effects, and certify feature selection."""
    panel = study['panel']
    strata = np.asarray(panel['stratum'])
    aligned = _align(panel)
    optics = study['interferometry']
    signal = np.asarray(optics['signal_phasor'], dtype=np.complex128)
    ref = np.asarray(optics['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(ref))
    h = np.column_stack((np.ones(len(strata)), strata == 1, strata == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    beta = _calibrate(h[cal], y[cal])
    deconfounded = y[target] - h[target] @ beta
    selection = _Selector(np.asarray(study['X_cand'], dtype=np.float64), deconfounded,
                          float(study['lam2']), float(study['lam0']), float(study['M'])).solve()
    return {'aligned_panel': aligned, 'calibration_weights': beta,
            'deconfounded_target': deconfounded, 'certified_selection': selection}
