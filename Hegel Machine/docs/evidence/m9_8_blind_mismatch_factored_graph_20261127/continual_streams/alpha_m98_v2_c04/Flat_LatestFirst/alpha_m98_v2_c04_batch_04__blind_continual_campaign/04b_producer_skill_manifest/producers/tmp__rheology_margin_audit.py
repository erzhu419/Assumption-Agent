import importlib.util
import itertools
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']
x, raw = c['true_channel_loads'], c['bridge_raw']
b = m._fit_bridge(c)
ordinary = np.linalg.norm(m._bridge_basis(x) @ b - raw, axis=1) < .01
# Evaluate only ordinary sweeps, while giving every training fold its gross errors.
# Both additions are specified before any held-out observations are scored.
def extras(a, family):
    qs = [a[:,i] + sign*a[:,j] for i,j in itertools.combinations(range(4),2) for sign in (-1,1)]
    if family == 'signed_pair':
        return np.column_stack([q*abs(q) for q in qs])
    return np.column_stack([q**3/(1+q*q) for q in [*a.T,*qs]])

norm = np.linalg.norm(x[ordinary], axis=1)
high = norm >= np.quantile(norm, .8)
errors = {key: [] for key in ('cubic_pair','signed_pair','rational')}
for j in np.flatnonzero(ordinary):
    v = np.arange(len(x)) != j
    bh = m._fit_bridge({'true_channel_loads':x[v], 'bridge_raw':raw[v]})
    keep = v & (np.linalg.norm(m._bridge_basis(x) @ bh - raw, axis=1) < .01)
    est = m._decode_bridge(raw[j:j+1], bh)[0]
    errors['cubic_pair'].append((est-x[j]).sum())
    # A correction this small can be evaluated via the local inverse Jacobian.
    h = 1e-5
    A = m._bridge_basis(x[keep])
    for family in ('signed_pair','rational'):
        def basis(a):
            return np.column_stack((m._bridge_basis(a),extras(a,family)))
        C = np.linalg.lstsq(basis(x[keep]),raw[keep],rcond=None)[0]
        J = (basis(est+np.eye(4)*h)-basis(est-np.eye(4)*h))@C/(2*h)
        dx = np.linalg.solve(J.T, raw[j]-basis(est[None,:])[0]@C)
        errors[family].append((est+dx-x[j]).sum())
for name, error in errors.items():
    e = np.array(error)
    print(name, 'axial RMSE overall/high load, max absolute', *(f'{v:.7g}' for v in (np.sqrt(np.mean(e**2)),np.sqrt(np.mean(e[high]**2)),max(abs(e)))))

tr,te = d['train'],d['test']
lab = tr['lab_id']; strata = tr['stratum']; n = len(lab)
features = m._features(tr,te)[:n]
past,hold = m._loads(tr,b)
yc = tr['strain_curves_obs'] - tr['strain_0'][:,None]
yr = tr['instant_rate_obs']
t = d['eval_times']; dt = d['past_dt']
aux = np.asarray(tr['aux_chamber_telemetry'])
errc, errr = np.empty_like(yc), np.empty_like(yr)
for l in np.unique(lab):
    keep = lab != l
    tau,theta = m._fit(features[keep],past[keep],hold[keep],dt,t,yc[keep],yr[keep])
    D,R = m._design(features[~keep],past[~keep],hold[~keep],dt,t,tau)
    errc[~keep] = (D@theta).reshape(-1,len(t)) - yc[~keep]
    errr[~keep] = R@theta - yr[~keep]
# Sample-level robust residual summaries avoid letting a single curve outlier determine group bias.
summary = np.column_stack((np.median(errc,axis=1),errr))
print('worst lab/horizon MAE',max(np.mean(abs(errc[lab==l]),axis=0).max() for l in np.unique(lab)))
for l,s in np.unique(np.c_[lab,strata],axis=0):
    keep=(lab==l)&(strata==s)
    print('lab/stratum/n',l,s,keep.sum(),'median curve/rate bias',np.round(np.median(summary[keep],axis=0),5))
a = np.c_[aux,np.clip(summary,-.03,.03)]
z = a.copy()
for l in np.unique(lab):
    v=lab==l
    z[v]-=z[v].mean(0)
print('max absolute auxiliary correlation per output, pooled',np.max(np.abs(np.corrcoef(a.T)[:-2,-2:]),axis=0),'within-lab centered',np.max(np.abs(np.corrcoef(z.T)[:-2,-2:]),axis=0))
