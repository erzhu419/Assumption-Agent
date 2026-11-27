import importlib.util
import pickle
import time

import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'beta_m96_w04_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open(root + 'public_beta_m96_w04_s00.pkl', 'rb'))[0]
s, gt = item['study'], item['gt']
rng = np.random.default_rng(431)
start = time.perf_counter()
out = m.solve_beta_m96_w04_s00(s)
elapsed = time.perf_counter() - start
rel = lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
S,U = m._forward(s)
D,r = m._profile(s,S,m._calibrate(s))
print('public seconds, weight max abs error, terminal relative error, design relative error, target relative error:', elapsed, max(abs(out['weights']-gt['w_opt'])), rel(out['terminal_field'],gt['terminal_field_A']), rel(D,gt['S_tilde']), rel(r,gt['r_y']))
assert max(abs(out['weights']-gt['w_opt'])) < 1e-6
assert rel(out['terminal_field'],gt['terminal_field_A']) < 1e-6
assert rel(D,gt['S_tilde']) < 1e-10

# Check clean affine gains and offsets, two-point sweeps, and two large
# outliers shared by the banks in either wing. All should preserve the inversion.
c = s['sensor_calibration_sweeps']
bank, x = c['bank_id'], c['ref_signal']
gain = np.array([.78,1.24,.86,1.31]); off = np.array([-.13,.24,.18,-.21])
rows = s['row_bank_id']
base_y = s['y_obs_raw']
errors = []
for mode in ('clean','two_points','wing_a','wing_b'):
    readings = gain[bank]*x + off[bank]
    keep = np.ones(x.size,dtype=bool)
    for b in range(4):
        idx = np.flatnonzero(bank==b)
        if mode=='two_points':
            keep[idx] = False
            keep[idx[[np.argmin(x[idx]),np.argmax(x[idx])]]] = True
        if (mode=='wing_a' and b<2) or (mode=='wing_b' and b>=2):
            readings[idx[[2,8]]] += [50.,-65.]
    a = dict(s,sensor_calibration_sweeps=dict(bank_id=bank[keep],ref_signal=x[keep],raw_readout=readings[keep]),y_obs_raw=gain[rows]*base_y+off[rows])
    got = m.solve_beta_m96_w04_s00(a)
    e = np.max(abs(m._calibrate(a)-base_y))
    errors.append(e)
    assert e < 1e-12
    assert max(abs(got['weights']-gt['w_opt'])) < 1e-6
print('calibration worst absolute recovery error:',max(errors))

# Known optimizers constructed from independent KKT gradients, including
# exact-budget vertices and partially active cases under varied designs.
worst = np.zeros(4)
for trial in range(250):
    n = 12
    S = rng.normal(size=(25,n))/5
    u = rng.uniform(.1,2.,n)
    ridge = 10.**rng.uniform(-4,1)
    typ = trial%5
    k = rng.permutation(n)
    lo,hi = (k[:4],k[4:8]) if typ==0 else ((k[:7],k[7:]) if typ==1 else (k[:0],k[:0]))
    if typ==3: lo = k
    if typ==4: hi = k
    w0 = u*rng.uniform(.05,.95,n)
    w0[lo] = 0.
    w0[hi] = u[hi]
    budget = w0.sum()
    g = np.full(n,rng.uniform(-2,2))
    g[lo] += rng.uniform(.02,2.,lo.size)
    g[hi] -= rng.uniform(.02,2.,hi.size)
    y = S@w0 + np.linalg.lstsq(S.T,ridge*w0-g,rcond=None)[0]
    w,c = m._box_simplex(S,y,ridge,u,budget)
    grad = S.T@(S@w-y)+ridge*w
    stat = grad-c['nu_sum']-np.array(c['lam_lower'])+np.array(c['lam_upper'])
    worst = np.maximum(worst,[max(abs(w-w0)),abs(w.sum()-budget),max(abs(stat)),c['primal_objective']-c['dual_lower_bound']])
print('250 QPs: worst weight, budget, stationarity and gap errors:',worst)
assert np.max(worst)<1e-9
