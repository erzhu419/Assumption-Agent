import pickle
import sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from prony_rheology import fit_prony_and_predict_stress

# Independent direct interval integration, with no recursive state update.
def reference(t, x, tau, c):
    u = t[:, None] - t[None, 1:]
    age = np.maximum(u, 0)
    dt = np.diff(t)
    delta = np.diff(x)
    out = c[0] * x.copy()
    for q, a in zip(tau, c[1:]):
        interval = -np.expm1(-dt/q) * (q/dt) * delta
        out += a * ((np.exp(-age/q) * (u >= 0)) @ interval)
    return out

with open('/holdout_task/public_rheology.pkl', 'rb') as f:
    public = pickle.load(f)
e = public['experiment']
model = fit_prony_and_predict_stress(e)
ref = public['gt_test_stress']
assert set(model) == {'test_stress', 'G_inf'}
assert model['test_stress'].shape == ref.shape
assert np.isfinite(model['test_stress']).all()
assert np.linalg.norm(model['test_stress']-ref)/np.linalg.norm(ref) <= .022
assert abs(model['G_inf']/public['gt_G_inf']-1) <= .06
shift = fit_prony_and_predict_stress(dict(e, train_stress_raw=e['train_stress_raw'] - 2.1 + .37*e['train_t']))
np.testing.assert_allclose(model['test_stress'], shift['test_stress'], rtol=1e-6, atol=1e-7)
np.testing.assert_allclose(model['G_inf'], shift['G_inf'], rtol=1e-6)
print('public schema, both thresholds, and affine drift invariance passed', flush=True)

for i, (ttau, jitter) in enumerate([([.02,.23,2.4,20.], False), ([.02,.23,2.4,20.], True), ([.055,.11,1.9,11.8], True), ([.035,.2,3.8,6.3,18.], False)]):
    rng = np.random.default_rng(632+i)
    t = e['train_t'].copy()
    if jitter:
        t[1:-1] += rng.uniform(-.019,.019,len(t)-2)
    x = np.interp(t, e['train_t'], e['train_strain'])
    tt = np.linspace(0,24,377)
    xx = np.interp(tt, e['test_t'], e['test_strain'])
    c = np.r_[14., rng.uniform(9., 30., len(ttau))]
    ref = reference(tt, xx, ttau, c)
    clean = reference(t, x, ttau, c)
    for spikes in (False, True):
        y = clean + 1.22 - .17*t
        if spikes:
            inds = rng.choice(len(t), 12, replace=False)
            y[inds] += rng.choice([-1,1], 12) * rng.uniform(.7, 6, 12)
        fit = fit_prony_and_predict_stress(dict(train_t=t,train_strain=x,train_stress_raw=y,test_t=tt,test_strain=xx))
        d = fit['test_stress']-ref
        print('exact spectrum',i,'jitter',jitter,'spikes',spikes,'modulus/stress/worst quartile %',np.round(100*np.r_[abs(fit['G_inf']/14-1),np.linalg.norm(d)/np.linalg.norm(ref),max(np.linalg.norm(d[q])/np.linalg.norm(ref[q]) for q in np.array_split(np.arange(len(tt)),4))],4),flush=True)
