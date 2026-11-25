import copy
import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('hereditary_deconv', root + 'hereditary_deconv.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(root + 'public_hereditary_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']

rng = np.random.default_rng(17)
worst = np.zeros(3)
worst_shift = 0.0
for case in cases:
    p = case['problem']
    gt = case['gt']
    base = solver.deconvolve_hereditary_driver(p)
    cal = p['calibration_run']
    t = cal['timestamps']
    past_t = cal['past_timestamps']
    past_u = cal['past_driver_history']
    u = cal['driver_u']
    tau = np.asarray(p['taus'])
    state = np.full(2, past_u[0])
    for dt, value in zip(np.diff(past_t), past_u[1:]):
        a = np.exp(-dt / tau)
        state = a * state + (1 - a) * value
    physical = gt['moduli'][0] * u.copy()
    for k, dt in enumerate(np.diff(t, prepend=past_t[-1])):
        a = np.exp(-dt / tau)
        state = a * state + (1 - a) * u[k]
        physical[k] += state @ gt['moduli'][1:]
    residual = cal['y_obs'] - physical
    trend = np.column_stack((np.ones(t.size), t-t[0]))
    beta = np.linalg.lstsq(trend, residual, rcond=None)[0]
    for _ in range(3):
        ordinary = np.abs(residual-trend @ beta) < 1
        beta = np.linalg.lstsq(trend[ordinary], residual[ordinary], rcond=None)[0]
    glitches = np.flatnonzero(~ordinary)
    assert 0 < glitches.size < t.size // 5
    print('public glitch indices:', glitches.tolist())
    q = copy.deepcopy(p)
    for signs in (np.ones(glitches.size), -np.ones(glitches.size), *rng.choice([-1,1], size=(32, glitches.size))):
        yy = q['calibration_run']['y_obs']
        yy[glitches] = physical[glitches] + (trend @ beta)[glitches] + signs*rng.uniform(2.5,5.,glitches.size)
        out = solver.deconvolve_hereditary_driver(q)
        e = np.asarray(out['moduli'])
        err = [np.max(np.abs(e/gt['moduli']-1)), np.sqrt(np.mean((out['recovered_driver']-gt['u_blind'])**2)), np.sqrt(np.mean((out['predicted_blind_response']-gt['y_blind_clean'])**2))]
        worst = np.maximum(worst, err)
        worst_shift = max(worst_shift, np.max(np.abs((e-base['moduli'])/gt['moduli'])))
print('68 public-trace glitch interventions; worst modulus relative error, driver RMSE, response RMSE:', worst.tolist())
print('worst relative modulus shift from unchanged trace:', worst_shift)
