"""Identify a shared product-inhibited enzymatic cascade from trajectories."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import log_ndtr

_NAMES = ('Vmax1', 'Km1', 'alpha_p', 'Vmax2', 'Km2', 'k_leak')


def _simulate(params, y0, t, amp, window):
    """Integrate each constant-feed segment separately at its actual breakpoints."""
    vmax1, km1, alpha, vmax2, km2, leak = params
    start, stop = map(float, window)
    breaks = [t[0], *sorted({x for x in (start, stop) if amp and t[0] < x < t[-1]}), t[-1]]
    out = np.empty((len(t), 3))
    out[0] = y0
    for left, right in zip(breaks, breaks[1:]):
        if left == right:
            continue
        feed = amp if start <= (left + right) / 2 < stop else 0.0

        def rhs(_, y):
            s, i, p = y
            v1 = vmax1 * s / (km1 + s + alpha * p)
            v2 = vmax2 * i / (km2 + i)
            return (feed - v1, v1 - v2 - leak * i, v2)

        j = np.searchsorted(t, left, side='right')
        k = np.searchsorted(t, right, side='right')
        nodes = t[j:k]
        if not len(nodes) or nodes[-1] != right:
            nodes = np.append(nodes, right)
        sol = solve_ivp(rhs, (left, right), y0, t_eval=nodes,
                        method='DOP853', rtol=3e-8, atol=2e-10)
        if not sol.success:
            raise RuntimeError(sol.message)
        out[j:k] = sol.y[:, :k-j].T
        y0 = sol.y[:, -1]
    return out


def _fit(traces):
    """Fit shared rates and latent starts under additive, zero-clipped noise."""
    prepared = [(np.asarray(tr['t_obs'], float), np.asarray(tr['y_obs'], float),
                 tr['feed_amp'], tr['feed_window']) for tr in traces]
    initial = np.maximum(0., np.concatenate([tr['y0_approx'] for tr in traces]))
    x0 = np.r_[np.log([1.5, .7, 1., 1.2, .45, .1]), initial]

    def residual(x):
        p = np.exp(x[:6])
        errors = []
        for n, (t, obs, amp, window) in enumerate(prepared):
            pred = _simulate(p, x[6+3*n:9+3*n], t, amp, window)
            r = (pred - obs) / .025
            # A recorded zero means the pre-clipping measurement was nonpositive.
            z = obs == 0
            r[z] = np.sqrt(-2 * log_ndtr(-r[z]))
            errors.append(r.ravel())
        return np.concatenate(errors)

    lo = np.r_[np.log([.005]*6), np.zeros(initial.size)]
    hi = np.r_[np.log([100.]*6), np.full(initial.size, np.inf)]
    fit = least_squares(residual, x0, bounds=(lo, hi), x_scale='jac',
                        ftol=2e-8, gtol=2e-7, diff_step=1e-5, max_nfev=120)
    return np.exp(fit.x[:6]), fit.x[6:].reshape(-1, 3), fit


def identify_and_forecast(dataset: dict) -> dict:
    """Fit common rate parameters and simulate the known test initial state."""
    p, _, _ = _fit(dataset['train_trajectories'])
    forecast = _simulate(p, np.asarray(dataset['y0_test'], float),
                         np.asarray(dataset['t_test'], float),
                         dataset['test_feed_amp'], dataset['test_feed_window'])
    return {'params': dict(zip(_NAMES, map(float, p))), 'forecast_SIP': forecast}
