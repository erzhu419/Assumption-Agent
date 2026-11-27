import importlib.util
import pickle
import numpy as np
from scipy.optimize import minimize_scalar

s = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']; g = item['campaign_diagnostics']; t = d['eval_times']; K = len(t)
bridge = m._axial_load_map(d['calibration_load_cells'])
out = m.solve_batch(d); tau = out['estimated_taus']; w = out['hereditary_weights']
x, _ = m._features(g, g); p, h, a = m._specimen(g, x, bridge); drift = a @ w[2:]
xj = m._features(d['train'], g)[1]; joint = w[2] + xj @ w[3:]
y = np.c_[g['strain_curves_obs'] - g['strain_0'][:,None], g['instant_rate_obs']]
R = np.r_[-np.expm1(-t[:,None] / tau), (1/tau)[None,:]]
base = h[:,None] * (R @ w[:2]); log = np.r_[np.log1p(t), 1.]
curves, rates = m._chamber(g, bridge, t, tau, w)
fixed = np.c_[curves-g['strain_0'][:,None], rates]
np.testing.assert_allclose(fixed, base + drift[:,None]*log, atol=1e-12, rtol=0)
c, r = m._branches((p,h,a), d['past_dt'], t, tau)
bench = np.c_[c @ w[:2] + drift[:,None]*t, r @ w[:2] + drift]
contrast = (r*tau - h[:,None]) * w[:2]
W = np.r_[np.full(K, 1/np.sqrt(K)), 1.]
labels = g['lab_id']; missing = np.isnan(g['coupon_features_raw']).any(1)

def predict(kind, train, test):
    if kind == 'reset_log': return fixed[test]
    if kind == 'joint_log': return base[test]+joint[test,None]*log
    if kind == 'benchtop': return bench[test]
    if kind in ('rational_time', 'rational_drift'):
        def prediction(q):
            if kind == 'rational_time':
                return base + drift[:,None]*np.r_[t/(1+q*t*t),1.]
            return base + (drift/(1+q*drift**2))[:,None]*log
        f = lambda q: np.sum(((prediction(q)[train]-y[train])*W)**2)
        q = min((0.,minimize_scalar(f,bounds=(0,5),method='bounded').x),key=f)
        return prediction(q)[test]
    if kind == 'memory_pair':
        B = contrast[:,None,:]*R[None,:,:]
    else:
        B = np.stack(((h-p[:,-1])[:,None]*R[:,0], (h-p[:,0])[:,None]*R[:,1], (h-p[:,-1])[:,None]*drift[:,None]*log),axis=2)
    A = (B[train]*W[None,:,None]).reshape(-1,B.shape[-1]); b=((y-fixed)[train]*W).ravel()
    return fixed[test] + B[test] @ np.linalg.lstsq(A,b,rcond=None)[0]

def metric(e):
    return np.sqrt(np.mean(e[:,:K]**2)), np.sqrt(np.mean(e[:,K]**2))

kinds = ['reset_log','joint_log','benchtop','rational_time','rational_drift','memory_pair','boundary_interaction']
for kind in kinds:
    e = np.empty_like(y)
    for lab in np.unique(labels):
        test=labels==lab; e[test] = predict(kind,~test,test)-y[test]
    print(kind, 'strain/rate', np.round(metric(e),6), 'worst lab strain/rate', np.max([metric(e[labels==l]) for l in np.unique(labels)],axis=0).round(6), 'missing strain/rate',np.round(metric(e[missing]),6),'worst coupon strain', round(np.sqrt(np.mean(e[:,:K]**2,1)).max(),6))
selected = np.empty_like(y)
for lab in np.unique(labels):
    test=labels==lab; pool=~test
    def inner_loss(k):
        return sum(np.sum(((predict(k,pool & (labels!=l),labels==l)-y[labels==l])*W)**2) for l in np.unique(labels[pool]))
    k=min(kinds,key=inner_loss); selected[test]=predict(k,pool,test)-y[test]; print('outer lab',lab,'inner choice',k)
print('nested strain/rate',np.round(metric(selected),6))
