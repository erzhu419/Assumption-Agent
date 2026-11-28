"""Calibrate the input, estimate the panel-B response, and fit separate readouts."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    keep = np.linalg.norm(v, axis=1) < 5
    u, v = u[keep], v[keep]
    eye = np.eye(u.shape[1])
    d = np.column_stack((np.tile(eye, (len(u), 1)),
                         (u[:, :, None]*eye).reshape(-1, u.shape[1]),
                         u.ravel()**3, ((u-1)*np.roll(u, -1, axis=1)).ravel()))
    p = _ls(d, v.ravel())
    offset, gain, cubic, cross = p[:4], p[4:8], p[8], p[9]
    w = (x-offset)/gain
    for _ in range(16):
        nxt = np.roll(w, -1, axis=1)
        error = offset + gain*w + cubic*w**3 + cross*(w-1)*nxt - x
        jac = (gain+3*cubic*w*w+cross*nxt)[:, :, None]*eye
        jac += cross*(w-1)[:, :, None]*np.roll(eye, 1, axis=1)
        w -= np.clip(np.linalg.solve(jac, error[..., None])[..., 0], -.5, .5)
    return w


def _features(x, params):
    h = x**3 / (1 + params[0]*x*x)
    for amplitude, frequency in zip(params[1::2], params[2::2]):
        h += amplitude*np.sin(frequency*x)
    return np.column_stack((_aff(x), h))


def _design(w, z):
    gate = np.sum(w*np.roll(z, 1, axis=1), axis=1)
    state = np.column_stack((_aff(z), gate))
    pair = z[:, 0]*z[:, 2] + z[:, 1]*z[:, 3]
    return np.column_stack((state, pair)), state


def _link(d, params):
    h = params[0] + d[:, 1:]@params[1:]
    return np.column_stack((d[:, 0], h + np.arctan(h)))


def _fit_readouts(panel, w, query, y, trajectory):
    u, v = panel['u'], panel['v']
    lower = [.05] + [-.2, 2]*3
    upper = [3] + [.2, 40]*3

    def panel_fit(params):
        d = _features(u, params)
        return d, _ls(d, v)

    def panel_residual(params):
        d, operator = panel_fit(params)
        return (d@operator-v).ravel()

    starts = ([.72, .008, 10, .004, g, .002, k]
              for g in (5, 8, 13, 16, 20, 25, 29, 34)
              for k in (14, 22, 31, 37) if abs(g-k) > 2)
    fits = sorted((least_squares(panel_residual, p, bounds=(lower, upper),
                                 max_nfev=130) for p in starts), key=lambda r: r.cost)
    _, operator = panel_fit(fits[0].x)
    scalar, _ = _design(w, _features(w, fits[0].x)@operator)
    initial_link = np.r_[0., _ls(scalar, y)[1:]]
    # Fit the shared transform primarily to the panel, with weaker supervision
    # from each training readout. Profile the linear coefficients at every step.
    state_weight = .20 / np.sqrt(trajectory.shape[1])

    def profile(params, predict=False):
        p, b = params[:7], params[7:]
        panel_design, operator = panel_fit(p)
        scalar, state = _design(w, _features(w, p)@operator)
        linked = _link(scalar, b)
        cy, ct = _ls(linked, y), _ls(state, trajectory)
        if predict:
            scalar_q, state_q = _design(query, _features(query, p)@operator)
            return _link(scalar_q, b)@cy, state_q@ct
        return np.r_[(panel_design@operator-v).ravel(),
                     .09*(linked@cy-y),
                     (state_weight*(state@ct-trajectory)).ravel()]

    bounds = (lower + [-np.inf]*len(initial_link), upper + [np.inf]*len(initial_link))
    best = min((least_squares(profile, np.r_[fit.x, scale*initial_link],
                              bounds=bounds, max_nfev=140)
                for fit in fits[:2] for scale in (1, 2)), key=lambda r: r.cost)
    return profile(best.x, predict=True)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    w = _calibrate(study['panel_a'], np.concatenate((train['x'], test['x'])))
    y, state = _fit_readouts(study['panel_b'], w[:n], w[n:], train['y'], train['traj'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
