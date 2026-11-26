import importlib.util
import pickle
import numpy as np
from numpy.testing import assert_allclose

path = '/holdout_task/alpha_w01_batch_05_solver.py'
spec = importlib.util.spec_from_file_location('coupon_candidate', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_alpha_w01_batch_05.pkl', 'rb'))[0]['dataset']
tr, te = d['train'], d['test']
public_maps = s._load_cells(d['calibration_load_cells'])
assert all(not np.any(term) for block in (tr, te) for term in s._inputs(block, np.zeros((len(block['lab_id']), 4)), public_maps, d['past_dt'], d['yield_stress'])[-2:])
assert_allclose(s.solve_alpha_w01_batch_05(d)['viscoplastic_weights'], 0, atol=0)

# Supply known lab maps, with clean sweeps on one wing and one EMI glitch per lab on the other.
maps = np.array([[.7536, -.1397], [1.2378, .0943], [.7075, -.1103], [1.2802, .1335]])
x = np.tile(np.linspace(-2.6, 2.6, 9), 4)
labs = np.repeat(np.arange(4), 9)
y = maps[labs, 0] * x + maps[labs, 1]
y[[21, 32]] += [0.7, -0.6]
checks = dict(lab_id=labs, cmd_stress=x, true_stress=y)
assert_allclose(s._load_cells(checks), maps, atol=1e-9)

dt, t = d['past_dt'], d['eval_times']
taus = np.array([.54, 3.91]); w = np.array([.68,.56,.08,.15,-.13,.11,-.09,.063,.037])
xa, xb = s._alignment(tr, te)
def truth(block, cov, threshold):
    g,b = maps[block['lab_id']].T
    ph = g[:,None]*block['past_stress_cmd'] + b[:,None]
    hold = g*block['stress_hold_cmd'] + b
    # Advance internal state sequentially over piecewise-constant stress history.
    state = np.zeros((len(g),2))
    for duration, stress in zip(dt, ph.T):
        state = stress[:,None] + (state-stress[:,None])*np.exp(-duration/taus)
    amplitude = (hold[:,None] - state)*w[:2]
    work = np.sum(np.maximum(np.abs(ph)-threshold,0)**2*dt,axis=1)
    amplitude[:,0] += w[8]*work*hold
    flow = w[2] + cov@w[3:7] + w[7]*np.sign(hold)*np.maximum(np.abs(hold)-threshold,0)**2
    return block['strain_0'][:,None] + amplitude@(1-np.exp(-t[None,:]/taus[:,None])) + flow[:,None]*t, amplitude@(1/taus)+flow

ys = .85
c, r = truth(tr, xa, ys)
ct, rt = truth(te, xb, ys)
train = dict(tr, strain_curves_obs=c, instant_rate_obs=r)
test = {k:v for k,v in te.items() if k!='instant_rate_obs'}
out = s.solve_alpha_w01_batch_05(dict(d, train=train, test=test, yield_stress=ys, calibration_load_cells=checks))
for key, ref in [('estimated_taus',taus),('hereditary_weights',w[:7]),('viscoplastic_weights',w[7:]),('predicted_strain',ct),('initial_rate_pred',rt)]:
    error = np.max(np.abs(out[key]-ref))
    print(key, 'above-yield max abs error', error)
    assert_allclose(out[key], ref, atol=2e-6, rtol=0)
