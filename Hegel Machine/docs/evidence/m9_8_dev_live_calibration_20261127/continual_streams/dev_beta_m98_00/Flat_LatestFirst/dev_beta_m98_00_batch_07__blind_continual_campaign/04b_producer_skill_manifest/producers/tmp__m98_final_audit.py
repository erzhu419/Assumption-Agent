import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item, = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))
s = item['study']
c = s['sensor_calibration_sweeps']
x, raw = c['ref_rosette'], c['raw_rosette']
a = m._rosette_terms(x)
from itertools import combinations
pair = [(x[:, i], x[:, j]) for i, j in combinations(range(4), 2)]
extra = np.column_stack([f for u, v in pair for q in (u-v, u+v) for f in (q/(1+q*q), (u*v)/(1+q*q))])
for name, design in [('current', a), ('pairwise rational extension', np.column_stack([a, extra]))]:
    errors = []
    for hold in np.array_split(np.random.default_rng(98).permutation(len(x)), 5):
        fit = np.setdiff1d(np.arange(len(x)), hold)
        coef = np.linalg.lstsq(design[fit], raw[fit], rcond=None)[0]
        errors.append(np.sqrt(np.mean((design[hold]@coef-raw[hold])**2, axis=0)))
    print(name, 'worst held-out fold/channel calibration RMSE:', np.max(errors))
S, U = m._design(s)
for j, pulse in enumerate(item['campaign_diagnostics']):
    yy = m._calibrate(dict(s, y_obs_raw=pulse['y_pulse_raw']))
    e = (S@pulse['w_pulse']-yy).reshape(len(s['mu_shifts']), -1, 4)
    y = yy.reshape(e.shape)
    print('pulse', j, 'worst shift/channel relative L2:', np.max(np.linalg.norm(e,axis=1)/np.linalg.norm(y,axis=1)))
out = m.solve_batch(s)
w = out['weights']; cert = out['kkt_certificate']; nu = cert['nu_sum']
z = s['Z_nuisance']; y = m._calibrate(s)
p = np.linalg.solve(z.T@z+s['nuisance_ridge']*np.eye(z.shape[1]), z.T)
A = S-z@(p@S); b = y-z@(p@y)
cap, total, ridge = (s[k] for k in ('u_box','total_budget','source_ridge'))
H = A.T@A+ridge*np.eye(len(w)); t = A.T@b
J = lambda v: 0.5*(np.linalg.norm(A@v-b)**2+ridge*(v@v))
grad = lambda v: H@v-t
ref = minimize(J, np.full(len(w),total/len(w)), jac=grad, bounds=[(0,cap)]*len(w), constraints=[{'type':'eq','fun':lambda v:v.sum()-total,'jac':lambda v:np.ones_like(v)}], method='SLSQP', options={'ftol':1e-14})
assert ref.success, ref.message
lo, hi = np.array(cert['lam_lower']), np.array(cert['lam_upper'])
g = grad(w)
checks = {'weight sum error':abs(w.sum()-total), 'stationarity max':np.max(abs(g-nu-lo+hi)), 'complementarity max':max(np.max(abs(w*lo)),np.max(abs((cap-w)*hi))), 'internal duality gap':J(w)-cert['dual_lower_bound'], 'objective disagreement':abs(J(w)-ref.fun), 'certificate objective disagreement':abs(J(w)-cert['primal_objective']), 'terminal assembly max error':np.max(abs(out['terminal_field']-np.tensordot(w,U,axes=1)))}
print(checks)
assert w.min() >= -1e-10 and w.max() <= cap+1e-10
assert min(lo.min(),hi.min()) >= 0 and np.max(abs(np.asarray(cert['weights'])-w)) <= 1e-12
assert max(map(abs,checks.values())) < 1e-9
