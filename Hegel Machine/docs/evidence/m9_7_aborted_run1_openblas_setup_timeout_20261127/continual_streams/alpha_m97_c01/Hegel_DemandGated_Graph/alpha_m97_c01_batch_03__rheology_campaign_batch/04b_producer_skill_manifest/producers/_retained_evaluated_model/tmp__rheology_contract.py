import copy, importlib.util, pickle
from time import perf_counter
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c01_batch_03_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
d = pickle.load(open(root+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']
cal = d['calibration_load_cells']; nominal = m._calibrate(cal)
for c in range(4):
    idx = np.flatnonzero(cal['lab_id'] == c)
    keep = (cal['lab_id'] != c) | (np.arange(len(cal['lab_id'])) == idx[0])
    sparse = m._calibrate({k: v[keep] for k,v in cal.items()})
    np.testing.assert_allclose(sparse[c], m._historical_line(c), atol=1e-12)
    np.testing.assert_allclose(sparse[np.arange(4) != c], nominal[np.arange(4) != c], atol=1e-12)
    worst = 0.
    for i in idx:
        for spike in (-12., -3.01, 3.01, 12.):
            noisy = {k: v.copy() for k,v in cal.items()}; noisy['true_stress'][i] += spike
            worst = max(worst, float(np.max(np.abs(m._calibrate(noisy)-nominal))))
    print('channel', c, 'sparse archive versus full current max coefficient delta', np.max(abs(sparse[c]-nominal[c])), 'worst EMI coefficient error', worst)
    assert worst < 0.001

def forward(data, dt, times, tau, w):
    past, hold, x = data
    ends = np.cumsum(dt)
    age = ends[-1] - ends
    kernel = np.exp(-age[:,None]/tau) - np.exp(-(age+dt)[:,None]/tau)
    g = hold[:,None] - past@kernel
    drift = w[2] + x@w[3:7] + w[7]*np.sign(hold)*np.sqrt(abs(hold))
    f = 1 - np.exp(-times[:,None]/tau)
    return (g*w[:2])@f.T + drift[:,None]*times/(1+0.85*times) + w[8]*(g[:,0]-g[:,1])[:,None]*f[:,0]

start = perf_counter()
out = m.solve_alpha_m97_c01_batch_03(d)
print('default-process solve seconds', perf_counter()-start)
n = len(d['train']['lab_id']); data = m._prepare(d); dt = d['past_dt']; t = d['eval_times']
test = tuple(v[n:] for v in data)
tau = out['estimated_taus']; w = np.r_[out['hereditary_weights'],out['regime_mode_weights']]
np.testing.assert_allclose(out['predicted_strain'] - d['test']['strain_0'][:,None], forward(test,dt,t,tau,w), atol=2e-12)
h = 1e-5
v = forward(test,dt,np.array([-h,h]),tau,w); rate = (v[:,1]-v[:,0])/(2*h)
print('max independent onset derivative error', np.max(abs(rate-out['initial_rate_pred'])))
np.testing.assert_allclose(rate, out['initial_rate_pred'], atol=2e-8)
clean = copy.deepcopy(d); tau0 = np.array([0.52, 4.6]); w0 = np.array([0.72,0.46,0.18,0.13,-0.16,0.08,-0.1,0.28,-0.18]); train = tuple(v[:n] for v in data)
clean['train']['strain_curves_obs'] = d['train']['strain_0'][:,None] + forward(train,dt,t,tau0,w0)
v = forward(train,dt,np.array([-h,h]),tau0,w0)
clean['train']['instant_rate_obs'] = (v[:,1]-v[:,0])/(2*h)
ret = m.solve_alpha_m97_c01_batch_03(clean)
print('noiseless max tau/weight errors', np.max(abs(ret['estimated_taus']-tau0)), np.max(abs(np.r_[ret['hereditary_weights'],ret['regime_mode_weights']]-w0)))
np.testing.assert_allclose(ret['estimated_taus'],tau0,atol=1e-5)
np.testing.assert_allclose(np.r_[ret['hereditary_weights'],ret['regime_mode_weights']],w0,atol=1e-5)
