"""Positive Prony inversion with affine load-cell drift and sparse spikes."""

import numpy as np
from scipy.optimize import least_squares, nnls
from scipy.signal import lfilter


_LOG_BOUNDS = np.log([0.02, 20.0])


class _History:
    def __init__(self, time, strain):
        self.strain = np.asarray(strain, dtype=float)
        dt = np.diff(time)
        self.increment = np.diff(self.strain)
        self.uniform = dt.size > 0 and np.allclose(dt, dt[0], rtol=1e-10, atol=0)
        self.dt = dt[0] if self.uniform else dt[:, None]
        self.elapsed = np.asarray(time) - time[0]

    def columns(self, log_times, derivative=False):
        x = self.dt * np.exp(-log_times)
        decay = np.exp(-x)
        drive = -np.expm1(-x) / x
        state = np.zeros((len(self.strain), len(log_times)))
        slope = np.zeros_like(state) if derivative else None
        if self.uniform:
            for j, a in enumerate(decay):
                state[1:, j] = lfilter([drive[j]], [1., -a], self.increment)
                if derivative:
                    forcing = x[j] * a * state[:-1, j] + (drive[j] - a) * self.increment
                    slope[1:, j] = lfilter([1.], [1., -a], forcing)
        else:
            # Restart the integrating factor in short blocks to keep its range bounded.
            inv_tau = np.exp(-log_times)
            forcing = drive * self.increment[:, None]
            blocks = []
            for start in range(0, len(self.increment), 32):
                stop = min(start + 32, len(self.increment))
                lag = self.elapsed[start+1:stop+1, None] - self.elapsed[start]
                factor = np.exp(-lag * inv_tau)
                sl = slice(start+1, stop+1)
                if np.min(factor, initial=1.) < 1e-80:
                    factor = None
                    for i in range(start, stop):
                        state[i+1] = decay[i]*state[i] + forcing[i]
                else:
                    state[sl] = factor * (state[start] + np.cumsum(forcing[start:stop]/factor, axis=0))
                blocks.append((start, stop, sl, factor))
            if derivative:
                forcing = x * decay * state[:-1] + (drive-decay) * self.increment[:, None]
                for start, stop, sl, factor in blocks:
                    if factor is None:
                        for i in range(start, stop):
                            slope[i+1] = decay[i]*slope[i] + forcing[i]
                    else:
                        slope[sl] = factor * (slope[start] + np.cumsum(forcing[start:stop]/factor, axis=0))
        return np.column_stack((self.strain, state)), slope


def _project(q, x):
    return x - q @ (q.T @ x)


def _positive_fit(a, y):
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0.] = 1.
    return nnls(a / scale, y, maxiter=20 * a.shape[1], atol=1e-10)[0] / scale


def _screen(a, drift, y):
    """Fit the affine nuisance independently of positivity and screen spikes."""
    keep = np.ones(len(y), dtype=bool)
    for _ in range(18):
        q, r = np.linalg.qr(drift[keep], mode='reduced')
        g = _positive_fit(_project(q, a[keep]), _project(q, y[keep]))
        b = np.linalg.solve(r, q.T @ (y[keep] - a[keep] @ g))
        resid = y - a @ g - drift @ b
        median = np.median(resid)
        noise = max(1.4826 * np.median(np.abs(resid - median)), 1e-12)
        new_keep = np.abs(resid - median) < 4. * noise
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return keep, g


def _spectrum(history, grid, a, drift, y, keep, g):
    q = np.linalg.qr(drift[keep], mode='reduced')[0]
    z = _project(q, y[keep])

    def columns(log_times, derivative=False):
        c, d = history.columns(log_times, derivative)
        return _project(q, c[keep]), _project(q, d[keep]) if derivative else None

    lo, hi = _LOG_BOUNDS
    power = g[1:] * np.linalg.norm(a[keep, 1:], axis=0)
    active = power > 0
    rng = np.random.default_rng(213)
    best_score = np.inf
    for m in range(5):
        if m == 0:
            elastic = _project(q, a[keep, :1])
            modulus = _positive_fit(elastic, z)
            rss = np.sum((elastic @ modulus - z)**2)
            best_score = len(z) * np.log(max(rss / len(z), 1e-30)) + 3*np.log(len(z))
            best, best_fit = modulus, None
            continue
        def residual(v):
            return columns(v[:m])[0] @ v[m:] - z

        def jacobian(v):
            c, d = columns(v[:m], True)
            return np.column_stack((d * v[m + 1:], c))

        base = np.linspace(lo + (hi-lo)*0.23, hi - (hi-lo)*0.10, m) if m > 1 else np.array([(lo+hi)/2])
        starts = [base]
        if np.any(active):
            w = power[active] / power[active].sum()
            starts.append(np.interp((np.arange(m)+.5)/m, np.cumsum(w)-.5*w, grid[active]))
        starts.extend(base + rng.normal(0., .9, m) for _ in range(4))
        lower = np.r_[np.full(m, lo), np.zeros(m+1)]
        upper = np.r_[np.full(m, hi), np.full(m+1, np.inf)]
        for guess in starts:
            logs = np.sort(np.clip(guess, lo+1e-4, hi-1e-4))
            coeff = _positive_fit(columns(logs)[0], z)
            fit = least_squares(residual, np.r_[logs, np.maximum(coeff, .01)],
                                jac=jacobian, bounds=(lower, upper), x_scale='jac',
                                max_nfev=160, ftol=1e-9)
            score = len(z)*np.log(max(2*fit.cost/len(z), 1e-30)) + (2*m+3)*np.log(len(z))
            if score < best_score:
                best_score, best, best_fit = score, fit.x, fit
    return best, best_fit, columns


def _equilibrium(v, result, columns, samples):
    """Second-order bias correction away from active spectral bounds."""
    m = (len(v)-1)//2
    if result is None or np.min(v[m:]) <= 1e-6 or np.any(result.active_mask):
        return float(v[m])
    lo, hi = _LOG_BOUNDS
    if np.min(np.minimum(v[:m]-lo, hi-v[:m])) < .10:
        return float(v[m])
    j = result.jac
    p = np.linalg.pinv(j, rcond=1e-9)
    cov = (p @ p.T) * (2*result.cost / max(1, samples-len(v)-2))
    _, d = columns(v[:m], True)
    h = 1e-4
    second = (columns(v[:m]+h, True)[1] - columns(v[:m]-h, True)[1])/(2*h)
    ids = np.arange(m)
    curvature = second @ (v[m+1:] * np.diag(cov)[:m]) + 2*d @ cov[ids, m+1+ids]
    delta = .5 * (p[m] @ curvature)
    return float(max(0., v[m] + np.clip(delta, -.20*v[m], .20*v[m])))


def fit_prony_and_predict_stress(experiment: dict) -> dict:
    """Fit the training history and predict drift-free stress from zero memory."""
    t = np.asarray(experiment['train_t'], dtype=float)
    y = np.asarray(experiment['train_stress_raw'], dtype=float)
    train = _History(t, experiment['train_strain'])
    test = _History(np.asarray(experiment['test_t'], dtype=float), experiment['test_strain'])
    drift = np.column_stack((np.ones_like(t), (t-t.mean())/np.ptp(t)))
    grid = np.linspace(*_LOG_BOUNDS, 61)
    a = train.columns(grid)[0]
    keep, g = _screen(a, drift, y)
    v, result, columns = _spectrum(train, grid, a, drift, y, keep, g)
    m = (len(v)-1)//2
    prediction = test.columns(v[:m])[0] @ v[m:]
    return {'test_stress': prediction, 'G_inf': _equilibrium(v, result, columns, int(keep.sum()))}
