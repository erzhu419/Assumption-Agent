"""Joint identification and forecasting for a fed, product-inhibited cascade."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import log_ndtr

_NAMES = ('Vmax1', 'Km1', 'alpha_p', 'Vmax2', 'Km2', 'k_leak')


class _Experiment:
    """Precompute constant-feed intervals and observation indices."""

    def __init__(self, times, amplitude, window):
        self.times = np.asarray(times, dtype=float)
        t = self.times
        start, end = window
        edges = sorted({t[0], t[-1]} | {
            x for x in (start, end) if amplitude and t[0] < x < t[-1]
        })
        self.intervals = []
        for a, b in zip(edges, edges[1:]):
            i, j = np.searchsorted(t, [a, b], side='right')
            grid = t[i:j]
            if not len(grid) or grid[-1] != b:
                grid = np.append(grid, b)
            u = amplitude if start <= (a + b) / 2 < end else 0.0
            self.intervals.append((a, b, i, j, grid, u))

    def simulate(self, params, y0):
        v1max, km1, alpha, v2max, km2, leak = params
        y = np.empty((len(self.times), 3))
        y[0] = state = y0

        def rhs(t, state):
            s, i, p = state
            v1 = v1max * s / (km1 + s + alpha * p)
            v2 = v2max * i / (km2 + i)
            return (u - v1, v1 - v2 - leak * i, v2)

        for a, b, i, j, grid, u in self.intervals:
            sol = solve_ivp(rhs, (a, b), state, t_eval=grid,
                            method='DOP853', rtol=2e-7, atol=1e-9)
            y[i:j] = sol.y[:, :j-i].T
            state = sol.y[:, -1]
        return y


def _fit(experiments, observations, y0s):
    # The data have approximately constant additive noise in concentration units.
    # Initial-state observations enter the likelihood once, at the first time.
    sigma = 0.025
    def residual(x):
        params = np.exp(x[:6])
        initial = x[6:].reshape(-1, 3)
        blocks = []
        for exp, obs, y0 in zip(experiments, observations, initial):
            pred = exp.simulate(params, y0)
            r = (pred - obs) / sigma
            # A recorded zero represents all latent readings at or below zero.
            clipped = obs == 0
            r[clipped] = np.sqrt(-2 * log_ndtr(-r[clipped]))
            blocks.append(r.ravel())
        return np.concatenate(blocks)

    x0 = np.r_[np.log([1.5, 0.8, 1.0, 1.0, 0.5, 0.15]), np.ravel(y0s)]
    lower = np.r_[np.full(6, -7.0), np.zeros(x0.size - 6)]
    upper = np.r_[np.full(6, 4.0), np.full(x0.size - 6, np.inf)]
    return least_squares(residual, x0, bounds=(lower, upper), x_scale='jac',
                         ftol=1e-8, max_nfev=120)


def identify_and_forecast(dataset: dict) -> dict:
    """Fit shared rate parameters and forecast from the supplied test state."""
    train = dataset['train_trajectories']
    experiments = [_Experiment(d['t_obs'], d['feed_amp'], d['feed_window'])
                   for d in train]
    observations = [np.asarray(d['y_obs'], dtype=float) for d in train]
    y0s = [np.maximum(d['y0_approx'], 0.0) for d in train]
    fit = _fit(experiments, observations, y0s)
    params = np.exp(fit.x[:6])
    test = _Experiment(dataset['t_test'], dataset['test_feed_amp'],
                       dataset['test_feed_window'])
    return {'params': dict(zip(_NAMES, map(float, params))),
            'forecast_SIP': test.simulate(params, dataset['y0_test'])}
