import importlib.util, pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_w04_batch_08_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_alpha_w04_batch_08.pkl','rb'))[0]['dataset']
a = d['train']; dt, times, sy = (d[k] for k in ('past_dt','eval_times','yield_stress'))
lab = a['lab_id']; missing = ~np.isfinite(a['coupon_features_raw']).all(axis=1)
y, r = a['strain_curves_obs'], a['instant_rate_obs']
models = ('none','missing','archive')
errors = {mode: (np.empty_like(y), np.empty_like(r)) for mode in models}
rng = np.random.default_rng(613)
folds = [[] for _ in range(4)]
for l in np.unique(lab):
    for m in (False, True):
        for fold, group in zip(folds, np.array_split(rng.permutation(np.flatnonzero((lab==l)&(missing==m))),4)):
            fold.extend(group)
for fold in folds:
    t = np.array(fold); u = np.setdiff1d(np.arange(len(lab)),t)
    tr, te = ({k:v[i] for k,v in a.items()} for i in (u,t))
    x, z = s._align(tr,te)
    fit = s._fit_batch(tr,x,dt,times,sy,d['calibration_load_cells'])
    for mode in models:
        yp, rp = s._predict(tr,te,x,z,dt,times,sy,fit,mode)
        ce, re = errors[mode]; ce[t] = yp-y[t]; re[t] = rp-r[t]

def summarize(c,e):
    # A row median separates persistent drift from the isolated strain spikes.
    late = np.abs(np.median(c[:,len(times)//2:]/times[len(times)//2:],axis=1))
    return np.round([np.median(np.abs(c)),np.median(late),np.quantile(late,.9),np.median(np.abs(e)),np.quantile(np.abs(e),.9)],5).tolist()
print('columns: curve MedAE, late slope MedAE/P90, rate MedAE/P90')
for name, mask in [(f'lab {l} {state}',(lab==l)&(missing==m)) for l in range(4) for m,state in [(False,'complete'),(True,'missing')]]+ [('all missing',missing),('all complete',~missing)]:
    print(name,int(mask.sum()),{mode:summarize(c[mask],e[mask]) for mode,(c,e) in errors.items()})
