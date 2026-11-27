import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root+'public_batch.pkl','rb'))[0]['dataset']
c = d['calibration_load_cells']; raw = c['bridge_raw']; true = c['true_channel_loads'].sum(1)
aff = np.c_[np.ones(len(raw)), raw]
quad = np.column_stack([aff]+[raw[:,i]*raw[:,j] for i in range(4) for j in range(i,4)])
for name, a in [('affine', aff), ('quadratic incl. cross-channel', quad)]:
    resid = []
    for j in range(len(a)):
        keep = np.arange(len(a)) != j
        resid.append(a[j] @ np.linalg.lstsq(a[keep], true[keep], rcond=None)[0] - true[j])
    print(name, 'calibration LOO RMSE/max', np.sqrt(np.mean(np.square(resid))), max(abs(np.array(resid))))
tr, te = d['train'], d['test']; dt, t = d['past_dt'], d['eval_times']
B = m._bridge(c); n = len(tr['lab_id']); x = m._features(tr,te); past, hold = m._loads(tr,B)
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]; r = tr['instant_rate_obs']; labs = tr['lab_id']
tele = tr['aux_chamber_telemetry']
# Within-lab contrasts remove laboratory offsets without using response labels.
z = tele.copy()
for lab in np.unique(labs):
    mask = labs == lab
    z[mask] -= z[mask].mean(axis=0)
for lab in np.unique(labs):
    k = labs != lab; h = ~k
    tau, theta = m._fit(x[:n][k], past[k], hold[k], dt, t, y[k], r[k])
    A, R = m._design(x[:n], past, hold, dt, t, tau)
    ec = y - (A@theta).reshape(n,len(t)); er = r - R@theta
    # Diagnostic challenger: fit extra coupon drift from auxiliary telemetry only
    # on the outer training labs. Clip sensor spikes using training residuals only.
    dc = np.clip(ec[k], -4*m._mad(ec[k].ravel()), 4*m._mad(ec[k].ravel())) @ t / (t@t)
    dr = np.clip(er[k], -4*m._mad(er[k]), 4*m._mad(er[k]))
    for name, zz in [('raw', tele), ('lab-centered', z)]:
        coef = np.linalg.lstsq(zz[k], (dc+dr)/2, rcond=None)[0]
        delta = zz[h]@coef
        print('held lab', lab, name, 'curve MAE base/extra', np.mean(abs(ec[h])), np.mean(abs(ec[h]-delta[:,None]*t)), 'rate RMSE base/extra', np.sqrt(np.mean(er[h]**2)), np.sqrt(np.mean((er[h]-delta)**2)))
out = m.solve_batch(d); tau, theta = out['estimated_taus'], out['hereditary_weights']
assert 0.05 < tau[0] < tau[1] < 25
p, h = m._loads(te, B); mem = np.zeros((len(h),2))
for j, step in enumerate(dt):
    decay = np.exp(-step/tau)
    mem = decay*mem + (1-decay)*p[:,j,None]
gap = h[:,None]-mem; drift = theta[2]+x[n:]@theta[3:]
curve = te['strain_0'][:,None]+(gap*theta[:2])@(1-np.exp(-t[:,None]/tau)).T+drift[:,None]*t
rate = (gap*theta[:2]/tau).sum(1)+drift
err = max(np.max(abs(curve-out['predicted_strain'])), np.max(abs(rate-out['initial_rate_pred'])))
print('independent recurrence and initial derivative max error', err)
assert err < 1e-12
