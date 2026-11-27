import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares, minimize

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']
y = b._calibrator(s['sensor_calibration_sweeps'])
f, B, nom = b._transport(s)
pulses = s['reference_tracer_pulses']
W = np.column_stack([p['w_pulse'] for p in pulses])
Y = np.column_stack([y(p['y_pulse_raw']) for p in pulses])
R = B @ W

def fit(mask):
    return least_squares(lambda wall: (f(wall, R)[0] - Y)[mask], nom,
                         bounds=(0, np.inf), gtol=1e-12).x

idx = np.arange(Y.shape[0]); n = len(s['sensor_ij'])
groups = [np.broadcast_to(((idx % n // 4) == g)[:, None], Y.shape) for g in range(n // 4)]
shifts = [np.broadcast_to((idx // n == g)[:, None], Y.shape) for g in range(len(s['mu_shifts']))]
pulse_slices = [np.broadcast_to(np.arange(Y.shape[1]) == g, Y.shape) for g in range(Y.shape[1])]
for name, slices in [('group', groups), ('shift', shifts), ('pulse', pulse_slices)]:
    worst_fit = max(np.sqrt(np.mean(((f(fit(~mask), R)[0]-Y)[mask])**2)) for mask in slices)
    worst_nom = max(np.sqrt(np.mean(((f(nom, R)[0]-Y)[mask])**2)) for mask in slices)
    assert worst_fit < worst_nom
    print('tracer', name, 'worst-fit', worst_fit, 'worst-nominal', worst_nom)
wall = fit(np.ones_like(Y, dtype=bool)); S, U = f(wall)
wout = b.solve_batch(s); w = wout['weights']; cert = wout['kkt_certificate']; budget = s['total_budget']; u = np.broadcast_to(s['u_box'], w.shape)
Z = s['Z_nuisance']; P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)
X = P @ S; yr = P @ y(s['y_obs_raw']); r = X @ w - yr; ridge = s['source_ridge']
stationarity = np.max(np.abs(X.T @ r + ridge*w + cert['nu_sum'] - cert['lam_lower'] + cert['lam_upper']))
field_error = np.max(np.abs(wout['terminal_field'].ravel(order='F') - U @ w))
gap = cert['primal_objective'] - cert['dual_lower_bound']
assert np.allclose(w, cert['weights']) and abs(w.sum()-budget) < 1e-12 and np.all((w>=0)&(w<=u))
assert abs(cert['primal_objective'] - .5*(r@r+ridge*(w@w))) < 1e-12
assert max(stationarity, field_error, gap) < 1e-10
print('public certificate stationarity, gap, field error', stationarity, gap, field_error)
blind = b.solve_batch(dict(s, reference_tracer_pulses=None)); wb = blind['weights']; cb = blind['kkt_certificate']
assert abs(wb.sum()-budget) < 1e-12 and np.all((wb>=0)&(wb<=u))
assert np.isfinite(blind['terminal_field']).all() and 0 <= cb['primal_objective']-cb['dual_lower_bound'] < 1e-10
print('public with pulses removed: weight and field deviations', np.linalg.norm(wb-w), np.linalg.norm(blind['terminal_field']-wout['terminal_field']))
rng = np.random.default_rng(327); worst = np.zeros(3)
for t in range(500):
    m = int(rng.integers(2, 40)); A = rng.normal(size=(m,m)); H = A.T @ A / m + .001*np.eye(m); c = rng.normal(size=m)
    u = rng.uniform(.01, 1, m); u[t % m] = 0
    budget = [0, u.sum(), u.sum()*1e-8, u.sum()*(1-1e-8), rng.uniform()*u.sum(), min(u[u>0])][t%6]
    x, nu = b._box_simplex(H,c,u,budget); g = H@x-c+nu
    dual_gap = float(-np.minimum(-g*x, g*(u-x)).sum())
    worst = np.maximum(worst, [abs(x.sum()-budget), max(0., np.max(-x), np.max(x-u)), dual_gap])
assert np.max(worst) < 1e-9, worst
print('500 box-simplex cases worst budget residual, box violation, supporting-plane gap', worst)
