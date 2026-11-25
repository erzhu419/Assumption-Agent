import pickle
import time
import numpy as np
from prony_rheology import fit_prony_and_predict_stress, _basis, _history, _spectrum

p = pickle.load(open('/holdout_task/public_rheology.pkl', 'rb'))
x = p['experiment']
t, e, tt = (x[k] for k in ('train_t', 'train_strain', 'test_t'))

def reference(t, e, G, tau, amp):
    # Independent direct integration over completed linear segments.
    result = G * e.copy()
    rate = np.diff(e) / np.diff(t)
    age = t[1:, None] - t[None, 1:]
    valid = age >= 0
    for T, A in zip(tau, amp):
        local = rate * T * (-np.expm1(-np.diff(t) / T))
        result[1:] += A * np.sum(np.where(valid, np.exp(-np.maximum(age, 0) / T) * local, 0), axis=1)
    return result

def metric(y, z):
    return np.linalg.norm(y-z) / np.linalg.norm(z)

def predict(data):
    start = time.monotonic()
    out = fit_prony_and_predict_stress(data)
    assert out['test_stress'].shape == np.shape(data['test_t'])
    assert np.isfinite(out['test_stress']).all() and np.isfinite(out['G_inf']) and out['G_inf'] >= 0
    return out, time.monotonic()-start

public, elapsed = predict(x)
print('public', metric(public['test_stress'], p['gt_test_stress']), abs(public['G_inf']/p['gt_G_inf']-1), 'seconds', elapsed)
rng = np.random.default_rng(87231)
factor = rng.uniform(.65, 1.35, size=t.size-1)
tj = np.r_[0., np.cumsum(np.diff(t) * factor)]
tj *= t[-1]/tj[-1]
te = np.r_[0., np.cumsum(np.diff(tt) * rng.uniform(.5, 1.5, size=tt.size-1))]
et = .035*np.sin(2*np.pi*(.10*te+.014*te**2)) + .024*(te>8) - .031*(te>16)
regimes = [(8.,[],[]), (12.,[.02],[40.]), (10.,[.035,.9,4.],[25.,10.,6.]), (10.,[.17,1.3,8.],[30.,14.,9.]), (14.,[.055,.4,3.,17.],[21.,19.,13.,12.]), (11.,[.11,3.,19.],[15.,11.,17.])]
for i, (G,tau,amp) in enumerate(regimes):
    T = tj if i%2 else t
    clean = reference(T, e, G, tau, amp)
    z = reference(te, et, G, tau, amp)
    np.testing.assert_allclose(_basis(_history(te, et), np.asarray(tau)) @ np.r_[G,amp], z, rtol=3e-13, atol=3e-13)
    err=[]; wall=[]
    for seed in range(8):
        r = np.random.default_rng(6000+seed)
        y = clean + (-2.4+.17*T if seed%2 else 1.3-.22*T) + r.normal(0, .015, T.size)
        ix = r.choice(T.size, round(.02*T.size), replace=False)
        y[ix] += r.choice([-1,1], size=ix.size) * (4.5 if seed%2 else r.uniform(.6,7,ix.size))
        out, dur = predict(dict(train_t=T, train_strain=e, train_stress_raw=y, test_t=te, test_strain=et))
        wall.append(dur)
        err.append((metric(out['test_stress'],z), (out['G_inf']-G)/G))
    a = np.array(err)
    print('modes',tau,'max stress/G',np.max(abs(a),axis=0).round(5),'median signed G',round(np.median(a[:,1]),5),'fails',np.sum(abs(a)>[.022,.06],axis=0),'max seconds',round(max(wall),3),flush=True)
