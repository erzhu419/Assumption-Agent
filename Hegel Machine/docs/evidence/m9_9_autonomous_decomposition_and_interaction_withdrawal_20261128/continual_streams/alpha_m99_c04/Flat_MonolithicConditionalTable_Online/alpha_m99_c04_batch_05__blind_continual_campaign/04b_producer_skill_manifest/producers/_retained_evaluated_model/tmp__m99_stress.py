import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
ds, d = item['dataset'], item['campaign_diagnostics']
tr, te = ds['train'], ds['test']
t, dt = ds['eval_times'], ds['past_dt']
cal = ds['calibration_load_cells']
bridge = m._fit_bridge(cal)
b, mat, terms = bridge
L, raw = cal['true_channel_loads'], cal['bridge_raw']
raw_fit = b + L @ mat
for ix, coef in terms:
    raw_fit += L[:, ix].prod(1)[:, None] * coef
err = np.linalg.norm(raw_fit - raw, axis=1)
clean_abs = []
for k in range(4):
    val = np.arange(len(L)) % 4 == k
    sub = dict(cal, true_channel_loads=L[~val], bridge_raw=raw[~val])
    bh, mh, th = m._fit_bridge(sub)
    held_fit = bh + L[val] @ mh
    for ix, coef in th:
        held_fit += L[val][:, ix].prod(1)[:, None] * coef
    good = np.linalg.norm(held_fit - raw[val], axis=1) < .01
    clean_abs.extend(np.abs(m._decode(raw[val][good], (bh, mh, th)).sum(1) - L[val][good].sum(1)))
print('bridge sweep OOF clean total-load MedAE/p95/max, clean/total:', np.quantile(clean_abs, [.5,.95,1]), len(clean_abs), len(L))
for group, name in [(tr, 'train'), (te, 'test'), (d, 'diagnostic')]:
    loads = m._decode(group['hold_channel_raw'], bridge)
    print(name, 'hold channel range', loads.min(), loads.max(), 'outside calibration channel bounds', np.sum((loads < L.min(0)) | (loads > L.max(0))))

x, _ = m._features(tr, te)
tau, theta = m._fit(m._calibrate(tr, bridge), x, dt, t)
xd, _ = m._features(d, d)
dc = m._calibrate(d, bridge)
h = dc['hold_stress']
z = theta[2] + xd @ theta[3:]
lag = dt.sum() - np.cumsum(dt)
mem = dc['past_stress'] @ (-np.expm1(-dt[:, None] / tau) * np.exp(-lag[:, None] / tau))
A = np.vstack([np.column_stack([-np.expm1(-t[:, None] / tau), t]), np.r_[1/tau, 1.]])
base = (h[:, None] * theta[:2]) @ A[:, :2].T
y = np.column_stack([d['strain_curves_obs'] - d['strain_0'][:, None], d['instant_rate_obs']])
lab = d['lab_id']
clean = np.isfinite(d['coupon_features_raw']).all(1)
scale = np.r_[np.full(len(t), .003), .004]

def report(name, p):
    e = np.abs(p - y)
    def pair(E):
        return [np.median(E[:, :-1]), np.median(E[:, -1])]
    print(name, 'worst labs', np.round(np.max([pair(e[lab == l]) for l in np.unique(lab)], axis=0), 6), 'complete', np.round(pair(e[clean]), 6), 'imputed', np.round(pair(e[~clean]), 6), 'worst horizon MedAE', round(np.median(e[:, :-1], axis=0).max(), 6))

report('artifact (diagnostic in-sample reference)', base + z[:, None] * (A @ m._CHAMBER_DRIFT))
features = {'linear': [z], 'hold product': [z, z * h], 'drift square': [z, z*z], 'history contrast': [z, mem[:,0] - mem[:,1]], 'history interaction': [z, z * (mem[:,0] - mem[:,1])]}
for a in (.25, 1., 4.):
    features[f'drift saturation a={a}'] = [z / (1 + a * z*z)]
    features[f'hold saturation a={a}'] = [z / (1 + a * ((h-3.5)/3.5)**2)]
for name, feats in features.items():
    design = np.einsum('ij,kb->ikjb', np.array(feats).T, A).reshape(len(h), len(scale), -1)
    pred = np.empty_like(y)
    for l in np.unique(lab):
        fit = (lab != l) & clean
        M = (design[fit] / scale[None,:,None]).reshape(-1, design.shape[-1])
        v = ((y-base)[fit] / scale).ravel()
        p = least_squares(lambda u: M @ u - v, np.linalg.lstsq(M,v,rcond=None)[0], loss='soft_l1', f_scale=2).x
        pred[lab == l] = base[lab == l] + design[lab == l] @ p
    report(name, pred)
    if name == 'linear':
        for l in np.unique(lab):
            ix = (lab == l) & clean
            print('  lab', l, 'n complete', ix.sum(), 'signed curve/rate median', np.median((pred-y)[ix,:-1]), np.median((pred-y)[ix,-1]))
