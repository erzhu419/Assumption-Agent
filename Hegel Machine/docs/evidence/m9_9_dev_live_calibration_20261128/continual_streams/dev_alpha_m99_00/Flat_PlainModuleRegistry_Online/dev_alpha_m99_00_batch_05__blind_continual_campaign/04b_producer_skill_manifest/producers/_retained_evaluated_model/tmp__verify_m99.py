import sys
sys.path.insert(0, '/holdout_task')
import pickle
import numpy as np
from _verified_upstream import calibrate_upstream
from downstream_solver import _features, _fit, _design, _predict_chamber, solve_downstream
from batch_solver import solve_batch
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
raw = item['dataset']
d = calibrate_upstream(dict(raw, chamber_verification_trials=item['campaign_diagnostics']))
a,b,q = (d[k] for k in ('train','test','chamber_verification_trials'))
dt,t = d['past_dt'],d['eval_times']
x = _features(a,b)[:len(a['lab_id'])]
out = solve_batch(raw)
direct = solve_downstream(d)
expected = {'estimated_taus':(2,), 'hereditary_weights':(7,), 'predicted_strain':(len(b['lab_id']),len(t)), 'initial_rate_pred':(len(b['lab_id']),)}
assert out.keys() == direct.keys() == expected.keys()
for k,shape in expected.items():
    assert out[k].shape == shape and np.isfinite(out[k]).all()
    np.testing.assert_allclose(out[k], direct[k])
tau,th = out['estimated_taus'],out['hereditary_weights']
assert .05 < tau[0] < tau[1] < 25
print('material parameters:',tau,th)
def report(name,e,r,s):
    print(name,'pooled strain/rate RMSE',*(round(np.sqrt(np.mean(z*z)),5) for z in (e,r)), 'worst horizon RMSE',round(np.sqrt(np.mean(e*e,axis=0)).max(),5))
    print('strain/rate RMSE by lab',[(int(l),*[round(np.sqrt(np.mean(z[s['lab_id']==l]**2)),5) for z in (e,r)]) for l in np.unique(s['lab_id'])])
xd = _features(q)
yp,rp = _predict_chamber(tau,th,q,xd,t)
y,r = q['strain_curves_obs'],q['instant_rate_obs']
report('independent chamber response, verification cohort normalized separately',yp-y,rp-r,q)
drift = th[2]+xd@th[3:]
g = 2*t/(np.sqrt(1+t)+1)
report('archived log control',yp+drift[:,None]*(np.log1p(t)-g)-y,rp-r,q)
report('linear drift control',yp+drift[:,None]*(t-g)-y,rp-r,q)
_,r0 = _predict_chamber(tau,th,q,xd,np.array([0.]))
ypulse,_ = _predict_chamber(tau,th,q,xd,np.array([0.,1.e-6]))
np.testing.assert_array_equal(ypulse[:,0],q['strain_0'])
np.testing.assert_allclose(np.diff(ypulse,axis=1)[:,0]/1.e-6,r0,rtol=2.e-6)
ce,re = np.empty_like(a['strain_curves_obs']),np.empty_like(a['instant_rate_obs'])
for lab in np.unique(a['lab_id']):
    m = a['lab_id']==lab
    fit = {k:v[~m] for k,v in a.items()}
    ta,tw = _fit(fit,x[~m],dt,t)
    c,rr = _design(ta,a['past_channel_raw'][m].sum(-1),a['hold_channel_raw'][m].sum(-1),x[m],dt,t)
    ce[m] = a['strain_0'][m,None] + c@tw - a['strain_curves_obs'][m]
    re[m] = rr@tw - a['instant_rate_obs'][m]
    print('held-out lab',lab,'taus',ta,'max weight change',round(max(abs(tw-th)),5),'strain median absolute',round(np.median(abs(ce[m])),5))
report('benchtop leave-one-lab-out response',ce,re,a)
