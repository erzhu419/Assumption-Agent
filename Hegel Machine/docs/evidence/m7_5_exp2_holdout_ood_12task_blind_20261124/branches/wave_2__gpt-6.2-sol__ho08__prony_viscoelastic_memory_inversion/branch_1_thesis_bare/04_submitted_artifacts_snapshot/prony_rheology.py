"""Recover a passive Prony spectrum from stress measurements with affine drift."""

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter

_TAU_MIN, _TAU_MAX = 0.02, 20.0


class _History:
    def __init__(self, t, strain):
        self.strain = np.asarray(strain, dtype=float)
        self.increment = np.diff(self.strain)
        self.dt = np.diff(np.asarray(t, dtype=float))
        self.uniform = self.dt.size > 0 and np.allclose(
            self.dt, self.dt[0], rtol=1e-10, atol=0)

    def design(self, tau, derivative=False):
        """Integrate each memory state exactly for piecewise-linear strain."""
        if self.uniform:
            z = self.dt[0] / tau
        else:
            z = self.dt[:, None] / tau
        a = np.exp(-z)
        f = -np.expm1(-z) / z
        h = np.zeros((self.strain.size, tau.size))
        dh = np.zeros_like(h) if derivative else None
        if self.uniform:
            for j, (aj, fj, zj) in enumerate(zip(a, f, z)):
                h[1:, j] = lfilter([fj], [1, -aj], self.increment)
                if derivative:
                    # Differentiate the recurrence with respect to log(tau).
                    u = zj * aj * h[:-1, j] + (fj - aj) * self.increment
                    dh[1:, j] = lfilter([1], [1, -aj], u)
        else:
            for i, dx in enumerate(self.increment):
                h[i + 1] = a[i] * h[i] + f[i] * dx
                if derivative:
                    dh[i + 1] = (a[i] * dh[i] + z[i] * a[i] * h[i]
                                 + (f[i] - a[i]) * dx)
        return np.column_stack((self.strain, h)), dh


def _weighted_fit(design, drift, stress, weight):
    """Project out the unrestricted drift before imposing modulus bounds."""
    w = np.sqrt(weight)
    a, d, y = design * w[:, None], drift * w[:, None], stress * w
    q, r = np.linalg.qr(d, mode='reduced')
    qa = np.einsum('ni,nj->ij', q, a)
    qy = q.T @ y
    moduli, _ = nnls(a - np.einsum('ni,ij->nj', q, qa), y - q @ qy,
                     maxiter=30 * a.shape[1])
    nuisance = np.linalg.solve(r, qy - qa @ moduli)
    residual = stress - np.einsum('ij,j->i', design, moduli) - drift @ nuisance
    return moduli, nuisance, residual


def _spot_spikes(design, drift, stress):
    """Use a Huber pilot, then remove only gross residual outliers."""
    weight = np.ones(stress.size)
    floor = 1e-10 * max(np.max(np.abs(stress)), 1e-100)
    for _ in range(20):
        _, _, residual = _weighted_fit(design, drift, stress, weight)
        sigma = max(1.4826 * np.median(np.abs(residual - np.median(residual))), floor)
        new_weight = (1.345 * sigma) / np.maximum(1.345 * sigma, np.abs(residual))
        if np.max(np.abs(weight - new_weight)) < 1e-4:
            break
        weight = new_weight
    good = np.abs(residual) <= 5 * sigma
    _, _, residual = _weighted_fit(design, drift, stress, good)
    sigma = max(np.sqrt(np.mean(residual[good] ** 2)), floor)
    return good, sigma


def _fit_modes(history, t, drift, stress, good, sigma, grid, grid_moduli):
    """Select a sparse model by BIC, refining times continuously in range."""
    y, d = stress[good], drift[good]
    weights = np.ones(y.size)
    log_min, log_max = np.log([_TAU_MIN, _TAU_MAX])
    # Weakly damp tails approaching the observation span: a very slow
    # exponential can trade off against both affine drift and equilibrium.
    time_scale = (t[-1] - t[0]) / 4
    elastic = history.strain[good, None]
    equilibrium, _, residual = _weighted_fit(elastic, d, y, weights)
    best_score = y.size * np.log(max(np.mean(residual ** 2), np.finfo(float).tiny))
    best_score += 3 * np.log(y.size)
    best = np.empty(0), equilibrium
    # Weight pilot modes by the response produced by a unit modulus.
    mass = grid_moduli[1:] * np.linalg.norm(history.design(grid)[0][:, 1:], axis=0)
    for m in range(1, min(5, (y.size - 4) // 2) + 1):
        low = np.r_[np.full(m, log_min), np.zeros(m + 1), -np.inf, -np.inf]
        high = np.r_[np.full(m, log_max), np.full(m + 3, np.inf)]
        # Broad and central starts, plus log-time quantiles from the pilot.
        starts = [np.geomspace(.08, 13, m), np.geomspace(.32, 7, m)] if m > 1 else [np.array([.4]), np.array([4.])]
        active = mass > 0
        if np.count_nonzero(active) >= m:
            cum = np.cumsum(mass[active]) - 0.5 * mass[active]
            starts.append(np.exp(np.interp((np.arange(m) + 0.5) / m * mass.sum(),
                                          cum, np.log(grid[active]))))

        def fun(p):
            tau = np.exp(p[:m])
            a, _ = history.design(tau)
            return np.r_[a[good] @ p[m:-2] + d @ p[-2:] - y, sigma * tau / time_scale]

        def jac(p):
            tau = np.exp(p[:m])
            a, da = history.design(tau, derivative=True)
            penalty = np.zeros((m, p.size))
            penalty[np.arange(m), np.arange(m)] = sigma * tau / time_scale
            return np.vstack((np.column_stack((da[good] * p[m+1:-2], a[good], d)), penalty))

        for start in starts:
            a, _ = history.design(start)
            b, nuisance, _ = _weighted_fit(a[good], d, y, weights)
            initial = np.r_[np.clip(np.log(start), log_min, log_max), b, nuisance]
            result = least_squares(fun, initial, jac=jac, bounds=(low, high),
                                   x_scale='jac', max_nfev=100)
            score = y.size * np.log(max(2 * result.cost / y.size, np.finfo(float).tiny))
            score += (2 * m + 3) * np.log(y.size)
            if score < best_score:
                best_score = score
                best = np.exp(result.x[:m]), result.x[m:-2]
    return best


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Fit the training data and return drift-free test stress and G_inf."""
    t = np.asarray(experiment['train_t'], dtype=float)
    y = np.asarray(experiment['train_stress_raw'], dtype=float)
    history = _History(t, experiment['train_strain'])
    drift = np.column_stack((np.ones(t.size), (t - t.mean()) / np.ptp(t)))
    grid = np.geomspace(_TAU_MIN, _TAU_MAX, 41)
    design, _ = history.design(grid)
    good, sigma = _spot_spikes(design, drift, y)
    grid_moduli, _, _ = _weighted_fit(design, drift, y, good)
    tau, moduli = _fit_modes(history, t, drift, y, good, sigma, grid, grid_moduli)
    test = _History(experiment['test_t'], experiment['test_strain'])
    return {'test_stress': test.design(tau)[0] @ moduli, 'G_inf': float(moduli[0])}
