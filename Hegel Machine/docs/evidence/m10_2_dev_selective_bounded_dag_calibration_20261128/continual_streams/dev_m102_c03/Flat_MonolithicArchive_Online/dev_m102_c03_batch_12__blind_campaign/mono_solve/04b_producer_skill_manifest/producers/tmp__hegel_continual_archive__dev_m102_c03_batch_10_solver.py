"""Identify the cyclic calibration, two nonlinear branches and relaxation readout."""
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    a = _aff(u)
    r = np.linalg.norm(v - a @ np.linalg.lstsq(a, v, rcond=None)[0], axis=1)
    keep = r < max(3*np.median(r), 1e-3)
    def design(z, j):
        w = z[:, j]
        p = z[:, (j+1)%4]
        return np.column_stack((np.ones(len(z)), w, p, w*p, w**3))
    def estimate(mask):
        return np.column_stack([np.linalg.lstsq(design(u[mask], j), v[mask, j], rcond=None)[0] for j in range(4)])
    def forward(z, b):
        return np.column_stack([design(z, j) @ b[:, j] for j in range(4)])
    b = estimate(keep)
    r = np.linalg.norm(v - forward(u, b), axis=1)
    keep = r < max(5*np.median(r), .001)
    b = estimate(keep)
    lin = np.diag(b[1]); lin[(np.arange(4)+1)%4, np.arange(4)] = b[2]
    z0 = np.linalg.solve(lin.T, (x-b[0]).T).T
    return np.array([least_squares(lambda z: forward(z[None,:], b)[0]-row, init, gtol=1e-12).x for row, init in zip(x,z0)])




def _design(z,c,d):
    return np.column_stack((_aff(z), c, d, (c[:,:,None]*d[:,None,:]).reshape(len(z),16)))


def _time(t, p):
    return -np.expm1(-t[:,None]*np.exp(p))

def _lstsq(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _project_operator(u, v, z, basis, start, bounds):
    def residual(rate):
        a = basis(u, rate)
        return (a @ _lstsq(a, v) - v).ravel()

    rate = least_squares(residual, [start], bounds=bounds, gtol=1e-12).x
    return basis(z, rate) @ _lstsq(basis(u, rate), v)


def _odd_basis(x, rate):
    return np.column_stack((_aff(np.tanh(x * rate)), x, x**3, x**5))


def _even_basis(x, rate):
    return _aff(np.cos(x * rate))


def solve_batch(study: dict) -> dict:
    train = study['train']
    test_x = study['test']['x']
    if len(test_x) == 0:
        return dict(y_hat=np.empty(0), state_pred=np.empty((0, len(study['eval_times']))), diagnostics_vec=np.zeros(4))
    z = _calibrate(study['panel_a'], np.vstack((train['x'], test_x)))
    panel = study['panel_c']
    c = _project_operator(panel['u'], panel['v1'], z, _odd_basis, 1.72, (0.5, 3.0))
    d = _project_operator(panel['u'], panel['v2'], z, _even_basis, 2.4, (1.5, 3.2))
    features = _design(z, c, d)
    times = np.asarray(study['eval_times'], dtype=float)
    traj = np.asarray(train['traj'], dtype=float)

    def temporal_residual(log_rates):
        a = _time(times, log_rates)
        return (a @ _lstsq(a, traj.T) - traj.T).ravel()

    log_rates = least_squares(temporal_residual, np.log([0.4, 1.8]),
                              bounds=(np.log([0.12, 1.0]), np.log([0.9, 4.0]))).x
    modes = np.linalg.qr(_time(times, log_rates), mode='reduced')[0]
    n = len(traj)
    target = np.column_stack((train['y'], traj @ modes))
    pred = features[n:] @ _lstsq(features[:n], target)
    y_hat = pred[:, 0]
    state = pred[:, 1:] @ modes.T
    last = state[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])
    return dict(y_hat=y_hat, state_pred=state, diagnostics_vec=diagnostics)
