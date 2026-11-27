import json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from downstream_solver import _features, _material, _basis, _loads, _gap
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = calibrate_upstream(item['dataset'])
tr, te = d['train'], d['test']
x, _ = _features(tr, te)
p, h = _loads(tr)
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:,None]
r = tr['instant_rate_obs']
lab = tr['lab_id']
aux = tr['aux_chamber_telemetry']
results = {k: [] for k in ('intrinsic', 'telemetry', 'nonlinear_gap')}
for label in np.unique(lab):
    train = lab != label
    val = ~train
    tau, theta = _material({k:v[train] for k,v in tr.items()}, x[train], dt, t)
    c, q = _basis(tau,p,h,x,dt,t)
    ec, er = y - c@theta, r - q@theta
    g = _gap(tau,p,h,dt)
    # Added terms preserve zero strain increment at t=0 and differentiate to the rate basis.
    a = -np.expm1(-t[:,None]/tau)
    gap_delta = g[:,0]-g[:,1]
    terms = np.column_stack((h*p[:,-1]*gap_delta, gap_delta/(1+gap_delta**2)))
    variants = {'intrinsic': (None,None), 'telemetry': (aux[:,None,:]*t[None,:,None],aux), 'nonlinear_gap': (a[None,:,:]*terms[:,None,:],terms/tau)}
    for name, (u,v) in variants.items():
        cp, rp = ec[val], er[val]
        if u is not None:
            from scipy.optimize import least_squares
            mat = np.concatenate((u[train].reshape(-1,u.shape[-1]),v[train]))
            target = np.r_[ec[train].ravel(),er[train]]
            b = least_squares(lambda b: mat@b-target,np.zeros(mat.shape[1]),loss='soft_l1',f_scale=.013).x
            cp, rp = cp-u[val]@b, rp-v[val]@b
        results[name].append((np.max(np.median(abs(cp),axis=0)),np.median(abs(rp))))
for name, scores in results.items():
    print(name, 'per-lab [worst horizon strain MedAE, rate MedAE]', np.round(scores,6), 'worst', np.round(np.max(scores,axis=0),6))
log = json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json'))
for e in log:
    print(e['completed_campaign_short_name'], np.round(np.median(e['facility_telemetry_trace'],axis=0),3))
out = solve_batch(item['dataset'])
for key, shape in {'estimated_taus':(2,), 'hereditary_weights':(7,), 'predicted_strain':(len(te['lab_id']),len(t)), 'initial_rate_pred':(len(te['lab_id']),)}.items():
    assert out[key].shape == shape and np.isfinite(out[key]).all(), key
