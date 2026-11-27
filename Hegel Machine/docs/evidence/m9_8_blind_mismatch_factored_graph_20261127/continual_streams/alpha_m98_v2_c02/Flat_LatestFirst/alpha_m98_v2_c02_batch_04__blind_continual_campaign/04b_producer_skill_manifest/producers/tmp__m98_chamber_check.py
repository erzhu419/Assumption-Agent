import importlib.util, json, pickle
import numpy as np
spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = item['dataset']
o = m.solve_batch(d)
assert set(o) == {'estimated_taus','hereditary_weights','predicted_strain','initial_rate_pred'}
tau = o['estimated_taus']; theta = o['hereditary_weights']; test = d['test']; dt = d['past_dt']; t = d['eval_times']
assert .05 < tau[0] < tau[1] < 25
_, x = m._standardize(d['train'], test)
stress = m._stress_calibration(d['calibration_load_cells'])
q = stress(test['past_channel_raw']); hold = stress(test['hold_channel_raw'])
z = np.zeros((len(q), 2))
for h, pulse in zip(dt, q.T):
    a = np.exp(-h / tau)
    z = a * z + (1 - a) * pulse[:, None]
gap = hold[:, None] - z
branch = gap * theta[:2]
drift = theta[2] + x @ theta[3:]
expected = test['strain_0'][:,None] + branch @ (1 - np.exp(-t[None,:] / tau[:,None])) + drift[:,None] * (1 - np.exp(-t))
err = np.max(abs(o['predicted_strain'] - expected))
rate_err = np.max(abs(o['initial_rate_pred'] - (np.sum(branch / tau,axis=1) + drift)))
assert err < 1e-12 and rate_err < 1e-12
h = 1e-6
near = np.array([-h, 0., h])
change, rate = m._chamber(m._design(test,x,stress,dt,near),tau,theta,near)
fd_err = np.max(abs((change[:,2]-change[:,0])/(2*h)-rate))
assert np.max(abs(change[:,1])) < 1e-12 and fd_err < 1e-8
ungated = expected + drift[:,None] * (t + np.expm1(-t))
print('deployment coupons',len(test['lab_id']),'max recurrence strain/rate errors',err,rate_err,'max initial derivative error',fd_err,'max gated vs ungated final-time strain difference',np.max(abs(expected[:,-1]-ungated[:,-1])))
rows = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
z = np.array([r['facility_telemetry_trace'] for r in rows]); cols = [4,7]
a = z[:,:,cols]; dist = np.sqrt(np.mean((a[:-1]-a[-1])**2,axis=(1,2)))
print('boundary-telemetry distances to prior entries',dist,'signed means',a.mean(axis=1))
