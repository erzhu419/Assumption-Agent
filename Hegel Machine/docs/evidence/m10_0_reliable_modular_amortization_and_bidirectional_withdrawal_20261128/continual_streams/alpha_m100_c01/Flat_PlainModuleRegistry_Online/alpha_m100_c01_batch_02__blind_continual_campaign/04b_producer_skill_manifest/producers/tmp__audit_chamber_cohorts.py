import sys, pickle, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
import downstream_solver as m
root = Path('/holdout_task')
with (root/'public_batch.pkl').open('rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
cal = calibrate_upstream(raw)
payload = json.loads((root/'_upstream_param_payload.json').read_text())
s = calibrate_upstream(dict(raw,test=item['campaign_diagnostics']), payload)['test']
tr, te = cal['train'], cal['test']
dt, t = cal['past_dt'], cal['eval_times']
x = m._features(s,s)[1]
import downstream_solver as m
from scipy.optimize import minimize_scalar
from collections import defaultdict

# Use the reference panel supplied with each cohort: train/test pooled for the material fit, diagnostic panel for the separate chamber experiment.
X, _ = m._features(tr, te)
pasts, holds, feats = m._inputs(tr, X)
labels = np.asarray(s['lab_id'])
truth = np.c_[s['strain_curves_obs']-s['strain_0'][:,None], s['instant_rate_obs']]
results = defaultdict(list)
for lab in np.unique(labels):
    keep = np.asarray(tr['lab_id']) != lab
    tau_l, th_l = m._fit(pasts[keep], holds[keep], feats[keep], dt, t, (tr['strain_curves_obs']-tr['strain_0'][:,None])[keep], tr['instant_rate_obs'][keep])
    pp, hh, ff = m._inputs(s, x)
    vc, vr = m._design(tau_l, pp, hh, ff, dt, t)
    branches = np.concatenate((vc[:,:,:2], vr[:,None,:2]), axis=1)
    base = branches @ th_l[:2]
    drift = ff @ th_l[2:]
    val = labels == lab
    def pred(kind, h):
        if kind in ('unit', 'exponential', 'memory difference'):
            g = -h*np.expm1(-t/h)
        elif kind == 'rational':
            g = t/(1+t/h)
        elif kind == 'quadratic denominator':
            g = t/(1+(t/h)**2)
        else:
            g = t
        return base + drift[:,None]*np.r_[g,1.]
    for kind in ('unit', 'exponential', 'rational', 'quadratic denominator', 'memory difference', 'linear'):
        h = 1. if kind in ('unit','linear') else minimize_scalar(lambda q: np.mean((pred(kind,q)[~val]-truth[~val])**2),bounds=(.2,8.),method='bounded').x
        p = pred(kind,h)
        if kind == 'memory difference':
            diff = branches[:,:,0]*th_l[0]-branches[:,:,1]*th_l[1]
            c = np.sum(diff[~val]*(truth-p)[~val])/np.sum(diff[~val]**2)
            p = p+c*diff
        e = np.abs((truth-p)[val])
        results[kind].append((np.quantile(e[:,:-1],.9),np.quantile(e[:,-1],.9),np.quantile(e[:,-2],.9),e[:,:-1].max()))
    print('lab',lab,'diagnostic rows',val.sum(),'unit terminal bias',round(float(np.median((truth-pred('unit',1.))[val,-2])),5))
for kind, vals in results.items():
    print(kind, 'worst-lab strain P90 / rate P90 / terminal P90 / strain max',np.round(np.max(vals,axis=0),6))
