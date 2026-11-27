import importlib.util, pickle
import numpy as np
from scipy.optimize import minimize
path = '/holdout_task/batch_solver.py'
spec = importlib.util.spec_from_file_location('candidate', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
s = item['study']
nx, ny = s['nx'], s['ny']
L = np.diag(s['V_reaction_2d'].ravel(order='F'))
for j in range(ny):
    for i in range(nx):
        row = i + nx*j
        for axis, q, n, stride in [('x', i, nx, 1), ('y', j, ny, nx)]:
            k, v = s['kappa_'+axis], s['vel_'+axis]
            h = s['L'+axis] / (n-1)
            for delta in (-1, 1):
                nb = (q + delta) % n
                a = (k[q] + k[nb])/(2*h*h) + max(-delta*v[q], 0)/h
                L[row, row + stride*(nb-q)] -= a
                L[row, row] += a
modes = np.column_stack([a.ravel(order='F') for a in s['source_modes']])
sensor = s['sensor_ij'] @ np.array([1, nx])
blocks = [np.linalg.solve(L + mu*np.eye(nx*ny), modes) for mu in s['mu_shifts']]
S = np.concatenate([a[sensor] for a in blocks])
Sp, Up = m._forward(s)
forward = max(np.max(np.abs(Sp-S)), np.max(np.abs(Up-blocks[-1])))
assert forward < 1e-11
sweeps = s['sensor_calibration_sweeps']
ref, raw = sweeps['ref_rosette'], sweeps['raw_rosette']
roll = np.arange(len(ref)) % 5
heldout = []
for fold in range(5):
    train = roll != fold
    sw = dict(sweeps, ref_rosette=ref[train], raw_rosette=raw[train])
    recovered = m._calibrate(dict(s, sensor_calibration_sweeps=sw, y_obs_raw=raw[~train])).reshape(-1,4)
    heldout.append(np.sqrt(np.mean((recovered-ref[~train])**2, axis=0)))
assert np.max(heldout) < 3e-5
pulse_errors = []
for pulse in item['campaign_diagnostics']:
    y = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw']))
    r = (S @ pulse['w_pulse'] - y).reshape(len(blocks), -1)
    pulse_errors.append(np.sqrt(np.mean(r*r, axis=1)))
assert np.max(pulse_errors) < 5e-6
Z = s['Z_nuisance']
P = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T@Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
D, ridge = P@S, s['source_ridge']
# Exercise the observed study, then source patterns that drive both box bounds.
ref_design = np.c_[np.ones(len(ref)), ref]
coef = np.linalg.lstsq(ref_design, raw, rcond=None)[0]
cases = [s]
for w_syn in (np.array([.42,.42,.42,.14,0,0,0,0]), np.array([0,0,0,0,.14,.42,.42,.42])):
    clean = (2*S@w_syn + Z@np.array([.05,-.03,.02])).reshape(-1,4)
    cases.append(dict(s, y_obs_raw=(coef[0] + clean@coef[1:]).ravel()))
worst = np.zeros(6)
for case in cases:
    cap, budget = case['u_box'], case['total_budget']
    out = m.solve_batch(case)
    w, cert = out['weights'], out['kkt_certificate']
    nu, lo, hi = cert['nu_sum'], cert['lam_lower'], cert['lam_upper']
    y = P@m._calibrate(case)
    def fun(w):
        r = D@w-y
        return .5*(r@r+ridge*(w@w))
    def grad(w): return D.T@(D@w-y)+ridge*w
    ref_opt = minimize(fun, np.full(len(w),budget/len(w)), jac=grad, method='SLSQP', bounds=[(0,cap)]*len(w), constraints=[{'type':'eq','fun':lambda x:x.sum()-budget,'jac':lambda x:np.ones_like(x)}], options={'ftol':1e-14,'maxiter':200})
    assert ref_opt.success, ref_opt.message
    g = grad(w)
    dual = fun(w) - g@w + nu*budget + cap*np.minimum(0,g-nu).sum()
    assert np.array_equal(cert['weights'], w)
    assert w.min() >= -1e-12 and w.max() <= cap+1e-12 and abs(w.sum()-budget) < 1e-12
    assert lo.min() >= 0 and hi.min() >= 0
    worst = np.maximum(worst, [np.max(np.abs(g-nu-lo+hi)), max(np.max(abs(w*lo)),np.max(abs((cap-w)*hi))), abs(fun(w)-dual), max(abs(cert['primal_objective']-fun(w)),abs(cert['dual_lower_bound']-dual)), abs(ref_opt.fun-fun(w)), np.max(abs(out['terminal_field'].ravel(order='F')-blocks[-1]@w))])
    print('active bounds', int(np.sum(w<1e-9)), int(np.sum(w>cap-1e-9)))
assert np.max(worst)<1e-9, worst
print('worst calibration-channel held-out RMSE', np.max(heldout), 'pulse by shift RMSE', np.array(pulse_errors).tolist(), 'dense forward max error', forward)
print('worst stationarity, complementarity, gap, certificate value, reference objective, terminal field:', worst.tolist())
