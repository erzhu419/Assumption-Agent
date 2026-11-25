"""Robust identification of a nonnegative Prony relaxation spectrum."""

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter


class _History:
    """Convolve the strain rate of a piecewise-linear strain history."""

    def __init__(self, t, strain):
        self.dt = np.diff(t, prepend=t[0])
        self.de = np.diff(strain, prepend=strain[0])
        self.uniform = t.size > 1 and np.allclose(self.dt[1:], self.dt[1], rtol=1e-10, atol=0)

    def basis(self, tau, derivative=False):
        if self.uniform:
            x = self.dt[1] / tau
            decay = np.exp(-x)
            weight = -np.expm1(-x) / x
            states = np.column_stack([
                lfilter([w], [1., -a], self.de) for a, w in zip(decay, weight)
            ])
            if not derivative:
                return states
            previous = np.vstack([np.zeros(tau.size), states[:-1]])
            drive = self.de[:, None] * (weight - decay) + previous * (x * decay)
            jac = np.column_stack([
                lfilter([1.], [1., -a], drive[:, j]) for j, a in enumerate(decay)
            ])
            return states, jac

        x = self.dt[1:, None] / tau
        decay = np.exp(-x)
        weight = -np.expm1(-x) / x
        states = np.zeros((self.de.size, tau.size))
        jac = np.zeros_like(states) if derivative else None
        for i, (a, w, de) in enumerate(zip(decay, weight, self.de[1:]), 1):
            states[i] = a * states[i - 1] + w * de
            if derivative:
                jac[i] = a * jac[i - 1] + x[i - 1] * a * states[i - 1] + (w - a) * de
        return (states, jac) if derivative else states


def _multiply(a, x):
    return np.einsum('ij,j->i', a, x)


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Fit physical moduli and training-only affine drift, then predict clean stress."""
    t, strain, y = (np.asarray(experiment['train_' + key], dtype=float)
                    for key in ('t', 'strain', 'stress_raw'))
    history = _History(t, strain)
    duration = t[-1] - t[0]
    drift = np.column_stack([np.ones(t.size), (t - t[0]) / duration])
    log_min, log_max = np.log([0.02, 20.0])
    grid = np.linspace(log_min, log_max, 73)

    def linear_fit(a, keep=slice(None)):
        d, a, target = drift[keep], a[keep], y[keep]
        q, _ = np.linalg.qr(d)
        ap = a - q @ (q.T @ a)
        norms = np.linalg.norm(ap, axis=0)
        norms[norms == 0.] = 1.
        elastic = nnls(ap / norms, target - q @ (q.T @ target),
                       maxiter=20 * a.shape[1])[0] / norms
        b = np.linalg.lstsq(d, target - _multiply(a, elastic), rcond=None)[0]
        return np.r_[elastic, b]

    a = np.column_stack([strain, history.basis(np.exp(grid))])
    full = np.column_stack([a, drift])
    c = linear_fit(a)
    floor = 1e-10 * max(1., np.max(np.abs(y)))
    # Estimate the inlier noise scale after removing gross spikes. Otherwise
    # the contaminated pilot fit can make real relaxation modes look like noise.
    for _ in range(3):
        residual = y - _multiply(full, c)
        scale = max(_mad(residual), floor)
        c = linear_fit(a, np.abs(residual - np.median(residual)) <= 4. * scale)
    scale = max(_mad(y - _multiply(full, c)), floor)
    cutoff = scale
    # Weakly downweight modes persisting well past the training window. Their
    # nearly linear tail is hard to distinguish from load-cell drift.
    horizon = duration / 4.

    def loss(squares, mixture):
        rho = np.ones((3, squares.size))
        u = squares[:t.size]
        if mixture:
            # Gaussian inliers plus a broad, approximately constant spike density.
            log_inlier = -0.5 * u
            log_total = np.logaddexp(log_inlier, -6.5)
            prob = np.exp(log_inlier - log_total)
            rho[0, :t.size] = -2. * (log_total - np.logaddexp(0., -6.5))
            rho[1, :t.size] = prob
            rho[2, :t.size] = -0.5 * prob * (1. - prob)
        else:
            rho[0, :t.size] = np.log1p(u)
            rho[1, :t.size] = 1. / (1. + u)
            rho[2, :t.size] = -rho[1, :t.size] ** 2
        rho[0, t.size:] = squares[t.size:]
        rho[2, t.size:] = 0.
        return rho

    def optimize(logs, refit=None):
        k = len(logs)
        a = np.column_stack([strain, history.basis(np.exp(logs))]) if k else strain[:, None]
        z0 = np.r_[logs, linear_fit(a)] if refit is None else refit.x
        noise = scale if refit is None else 1.2 * scale
        lower = np.r_[np.full(k, log_min), np.zeros(k + 1), -np.inf, -np.inf]
        upper = np.r_[np.full(k, log_max), np.full(k + 3, np.inf)]

        def residuals(z):
            tau = np.exp(z[:k])
            a = np.column_stack([strain, history.basis(tau), drift]) if k else np.column_stack([strain, drift])
            return np.r_[_multiply(a, z[k:]) - y, scale * tau / horizon]

        def jacobian(z):
            if not k:
                return np.column_stack([strain, drift])
            tau = np.exp(z[:k])
            a, derivatives = history.basis(tau, derivative=True)
            top = np.column_stack([derivatives * z[k + 1:-2], strain, a, drift])
            bottom = np.column_stack([np.diag(scale * tau / horizon), np.zeros((k, k + 3))])
            return np.vstack([top, bottom])

        return least_squares(residuals, z0, jac=jacobian, bounds=(lower, upper),
                             loss=lambda u: loss(u, refit is None), f_scale=noise,
                             x_scale='jac', max_nfev=120)

    # Grow the spectrum from both the previous fit and independent grid starts.
    # Compare orders at a common noise scale so that large spikes do not drive
    # selection of extra relaxation branches.
    best = previous = optimize(np.array([]))
    best_order = 0
    best_score = 2 * best.cost / cutoff**2 + 3 * np.log(t.size)
    ranked = np.argsort(c[1:-2] * np.linalg.norm(a[:, 1:], axis=0))[::-1]
    active = grid[ranked[c[1:-2][ranked] > 0.]]
    for k in range(1, 7):
        old = previous.x[:k - 1]
        starts = [np.r_[old, x] for x in np.linspace(log_min, log_max, 9)
                  if not old.size or np.min(np.abs(old - x)) >= .25]
        starts.append(np.linspace(log_min, log_max, k + 2)[1:-1])
        selected = []
        for x in active:
            if all(abs(x - j) >= .35 for j in selected):
                selected.append(x)
                if len(selected) == k:
                    starts.append(np.array(selected))
                    break
        previous = min((optimize(x) for x in starts), key=lambda fit: fit.cost)
        score = 2 * previous.cost / cutoff**2 + (2 * k + 3) * np.log(t.size)
        if score < best_score:
            best, best_order, best_score = previous, k, score
        elif k >= best_order + 2:
            break

    k = best_order
    best = optimize(best.x[:k], refit=best)
    tau = np.exp(best.x[:k])
    equilibrium = float(best.x[k])
    test_t = np.asarray(experiment['test_t'], dtype=float)
    test_strain = np.asarray(experiment['test_strain'], dtype=float)
    prediction = equilibrium * test_strain
    if k:
        prediction += _multiply(_History(test_t, test_strain).basis(tau), best.x[k + 1:-2])
    return {'test_stress': prediction, 'G_inf': equilibrium}
