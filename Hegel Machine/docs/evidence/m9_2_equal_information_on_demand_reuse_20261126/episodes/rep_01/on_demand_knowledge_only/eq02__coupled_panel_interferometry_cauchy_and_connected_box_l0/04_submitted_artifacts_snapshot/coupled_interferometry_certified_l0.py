"""Reference-calibrated interferometry and certified boxed sparse regression."""

import heapq

import numpy as np
from scipy.optimize import lsq_linear


def _align(panel, strata):
    groups, center = np.unique(panel['center_id'], return_inverse=True)
    values = np.array(panel['X_panel_raw'], dtype=np.float64, copy=True)
    controls = np.asarray(panel['controls'], dtype=np.float64)
    for c in range(groups.size):
        rows = center == c
        ref = controls[rows]
        finite = np.isfinite(ref)
        valid = finite.sum(axis=0) >= 2
        ref = np.where(finite[:, valid], ref[:, valid], np.nan)
        offset = np.median(np.nanmean(ref, axis=0))
        gain = np.median(np.nanstd(ref, axis=0))
        values[rows] = (values[rows] - offset) / gain
    cell = 3 * center + strata
    counts = np.bincount(cell, minlength=3 * groups.size)
    w = (cell.size / (3 * groups.size * counts[cell]))[:, None]
    finite = np.isfinite(values)
    fill = np.sum(w * np.where(finite, values, 0), axis=0) / np.sum(w * finite, axis=0)
    return np.where(finite, values, fill)


def _cauchy(design, y):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    # Re-estimate scale during warm-up so glitches cannot permanently inflate it.
    for i in range(100):
        r = y - design @ beta
        if i < 15:
            sigma = max(1e-8, 1.4826 * np.median(np.abs(r - np.median(r))))
        scale = 1.5 * sigma
        root_w = scale / np.hypot(scale, r)
        new = np.linalg.lstsq(design * root_w[:, None], y * root_w, rcond=None)[0]
        delta = np.linalg.norm(new - beta)
        beta = new
        if i >= 15 and delta < 1e-10 * (1 + np.linalg.norm(beta)):
            break
    return beta


class _BoxSearch:
    def __init__(self, x, y, l0, l2, box):
        self.x, self.y = x, y
        self.l0, self.l2, self.box = l0, l2, box
        self.p = x.shape[1]
        self.bits = 1 << np.arange(self.p)
        self.full = (1 << self.p) - 1
        g = x.T @ x
        self.g = (g + g.T) * 0.5
        self.b = x.T @ y
        self.h = self.g + l2 * np.eye(self.p)
        self.hdiag = self.h.diagonal().copy()
        # Normalize the Gram matrix before extracting a PSD diagonal minorant.
        norm = np.sqrt(np.maximum(self.g.diagonal(), np.finfo(float).tiny))
        corr = self.g / norm[:, None] / norm[None, :]
        rho = max(0., np.linalg.eigvalsh(corr)[0] - 1e-11)
        self.d = rho * self.g.diagonal()
        self.k = self.d + l2
        self.koff = self.g - np.diag(self.d)
        self.q = self.koff.diagonal().copy()
        self.t = np.minimum(box, np.sqrt(np.divide(2*l0, self.k, out=np.full(self.p, np.inf), where=self.k > 0)))
        self.slope = l0 / np.where(self.t > 0, self.t, 1.0) + 0.5 * self.k * self.t
        self.switch = self.slope + self.q * self.t
        self.xl = x.astype(np.longdouble)
        self.yl = y.astype(np.longdouble)
        self.dl = self.d.astype(np.longdouble)
        self.upper = float(0.5 * (y @ y))
        self.best = np.zeros(self.p)
        self.seen = set()

    def _try_support(self, support):
        mask = int(self.bits[support].sum())
        if mask in self.seen:
            return
        self.seen.add(mask)
        if not mask:
            return
        hs = self.h[np.ix_(support, support)]
        try:
            w = np.linalg.solve(hs, self.b[support])
        except np.linalg.LinAlgError:
            w = np.full(int(support.sum()), np.inf)
        if np.max(np.abs(w)) > self.box:
            xs = self.x[:, support]
            ys = self.y
            if self.l2:
                xs = np.vstack((xs, np.sqrt(self.l2)*np.eye(xs.shape[1])))
                ys = np.r_[ys, np.zeros(xs.shape[1])]
            w = lsq_linear(xs, ys, bounds=(-self.box, self.box), method='bvls', tol=1e-12).x
        w = np.clip(w, -self.box, self.box)
        w[np.abs(w) <= 1e-8] = 0.
        r = self.x[:, support] @ w - self.y
        val = float(0.5*(r @ r + self.l2*(w @ w)) + self.l0*np.count_nonzero(w))
        if val < self.upper:
            self.upper = val
            self.best = np.zeros(self.p)
            self.best[support] = w

    def _relax(self, zero, active, initial):
        # Convex envelope of each diagonal quadratic plus its activation cost.
        # Its tangent dual is exactly the separable minorant recorded in the leaf.
        w = initial.copy()
        w[zero] = 0.
        residual = self.b - self.koff @ w
        order = np.flatnonzero(~zero).tolist()
        q, h, slope, switch, box = self.q, self.hdiag, self.slope, self.switch, self.box
        for _ in range(100):
            movement = 0.
            for j in order:
                z = residual[j] + q[j] * w[j]
                az = abs(z)
                if active[j] or az > switch[j]:
                    v = min(box, az / h[j]) if h[j] > 0 else box
                elif az <= slope[j]:
                    v = 0.
                else:
                    v = min(box, (az - slope[j]) / q[j]) if q[j] > 0 else box
                if z < 0.:
                    v = -v
                dv = v - w[j]
                if dv:
                    residual -= dv * self.koff[:, j]
                    w[j] = v
                    movement = max(movement, abs(dv))
            if movement < 2e-10 * (1. + box):
                break
        return w

    def _leaf(self, z, a, initial):
        zero = (z & self.bits) != 0
        active = (a & self.bits) != 0
        free = ~(zero | active)
        w = self._relax(zero, active, initial)
        magnitude = np.abs(w)
        self._try_support(active | (free & (magnitude > np.maximum(1e-8, 0.5*self.t))))
        if z == 0 and a == 0:
            self._try_support(magnitude > 1e-8)
        c = w.astype(np.longdouble)
        r = self.yl - self.xl @ c
        g = self.xl.T @ r
        grad = np.abs(g + self.dl*c)
        k = self.dl + self.l2
        v = np.minimum(self.box, np.divide(grad, k, out=np.full(self.p, self.box, dtype=np.longdouble), where=k>0))
        gain = v * (grad - 0.5*k*v)
        on = self.l0 - gain
        const = 0.5*(r @ r + (self.dl*c) @ c) + g @ c
        val = const + on[active].sum() + np.minimum(on[free], 0).sum()
        guard = 4e-13*(1 + abs(const) + np.abs(g) @ np.abs(c) + gain.sum() + self.p*self.l0)
        lb = float(val - guard)
        leaf = {'fixed_zero': zero.tolist(), 'fixed_active': active.tolist(), 'lb': lb,
                'minorant_diagonal': self.d.tolist(), 'minorant_center': w.tolist()}
        frac = np.divide(magnitude, self.t, out=np.zeros(self.p), where=self.t>0)
        score = np.minimum(frac, 1.-frac)
        score[~free] = -np.inf
        branch = int(np.argmax(score)) if np.any(free) else None
        return lb, leaf, branch, w

    def solve(self):
        heap = []

        def add(z, a, initial):
            lb, leaf, branch, w = self._leaf(z, a, initial)
            heapq.heappush(heap, (lb, z, a, leaf, branch, w))

        add(0, 0, self.best)
        while heap[0][0] < self.upper - 8e-6:
            if len(heap) >= 2048:
                raise RuntimeError('Box-L0 partition exceeded the certificate budget')
            _, z, a, _, j, w = heapq.heappop(heap)
            if j is None:
                raise RuntimeError('Box-L0 full-support bound did not close')
            bit = 1 << j
            add(z | bit, a, w)
            add(z, a | bit, w)
        lower = min(self.upper, heap[0][0])
        return {'weights': self.best.tolist(), 'objective': self.upper,
                'certified_lower_bound': lower, 'subcube_leaves': [node[3] for node in heap]}


def solve_coupled_interferometry_l0(study: dict) -> dict:
    """Align references, estimate the optical calibration, and certify support selection."""
    panel = study['panel']
    strata = np.asarray(panel['stratum'], dtype=int)
    aligned = _align(panel, strata)
    optics = study['interferometry']
    signal = np.asarray(optics['signal_phasor'], dtype=np.complex128)
    reference = np.asarray(optics['ref_phasor'], dtype=np.complex128)
    phase = np.angle(signal * np.conj(reference))
    h = np.column_stack((np.ones(strata.size), strata == 1, strata == 2, aligned, phase))
    y = np.asarray(study['y_assay_raw'], dtype=np.float64)
    cal = np.asarray(study['calibration_indices'])
    target = np.asarray(study['stage3_indices'])
    theta = _cauchy(h[cal], y[cal])
    deconfounded = y[target] - h[target] @ theta
    cert = _BoxSearch(np.asarray(study['X_cand'], dtype=np.float64), deconfounded,
                      float(study['lam0']), float(study['lam2']), float(study['M'])).solve()
    return {'aligned_panel': aligned, 'calibration_weights': theta,
            'deconfounded_target': deconfounded, 'certified_selection': cert}
