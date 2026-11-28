import sys, pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/holdout_task')
from upstream_transducer import _fit, _forward, _inverse, _PARTNER

with open('/holdout_task/public_batch.pkl', 'rb') as fh:
    ds = pickle.load(fh)[0]['dataset']
c = ds['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
axis = np.count_nonzero(x, axis=1) == 1
# The gross-spike cutoff is far above the ordinary calibration residuals.
clean = np.max(np.abs(y-x), axis=1) < 2
rng = np.random.default_rng(405)
folds = [('axis', np.flatnonzero(axis))] + [(f'mixed-{i}', v) for i,v in enumerate(np.array_split(rng.permutation(np.flatnonzero(~axis)), 5), 1)]

def shared(a, b, good, guess):
    def project(p):
        h,k,s,t,u = p
        z = a[:,_PARTNER]
        v = a + s*np.sin(h*a) + t*z + u*(np.cos(k*z)-1)
        va, ba = v[good], b[good]
        vc, bc = va-va.mean(axis=0), ba-ba.mean(axis=0)
        gain = np.sum(vc*bc,axis=0)/np.sum(vc**2,axis=0)
        offset = ba.mean(axis=0)-gain*va.mean(axis=0)
        return v*gain+offset, gain, offset
    p = least_squares(lambda p: (project(p)[0][good]-b[good]).ravel(), guess, gtol=1e-12).x
    _,g,o = project(p)
    return p[:2], np.column_stack((o,g,g[:,None]*p[2:])), p

def extra(a, kind):
    z = a[:,_PARTNER]
    return a*z if kind == 'product' else (a-z)/(1+(a-z)**2)

scores = {k: [] for k in ('current', 'shared', 'product', 'rational difference')}
for label, idx in folds:
    tr = ~np.isin(np.arange(len(x)),idx)
    a,b = x[tr],y[tr]
    v = idx[clean[idx]]
    f,p = _fit(dict(true_channel_loads=a,bridge_raw=b))
    err = b-_forward(a,f,p)
    good = np.max(abs(err),axis=1)<0.01
    fs,ps,_ = shared(a,b,good,[*f,*np.median(p[:,2:]/p[:,1,None],axis=0)])
    pred = _forward(x[v],f,p)
    for kind, ff, pp in [('current',f,p),('shared',fs,ps)]:
        e = _forward(x[v],ff,pp)-y[v]
        load = _inverse(y[v],ff,pp)-x[v]
        scores[kind].append((np.max(abs(e)),np.max(abs(load)),np.sum(e**2),np.sum(load**2),e.size))
        print(label,kind,'max bridge/load',*scores[kind][-1][:2])
    for kind in ('product','rational difference'):
        u = extra(a[good],kind)
        z = a[good][:,_PARTNER]
        zv = x[v][:,_PARTNER]
        ec = np.empty_like(pred)
        for j in range(4):
            def design(q,z,u):
                return np.column_stack((np.ones(len(q)),q,np.sin(f[0]*q),z,np.cos(f[1]*z)-1,u))
            w = np.linalg.lstsq(design(a[good,j],z[:,j],u[:,j]), b[good,j],rcond=None)[0]
            ec[:,j] = design(x[v,j],zv[:,j],extra(x[v],kind)[:,j])@w-y[v,j]
        scores[kind].append((np.max(abs(ec)),0,np.sum(ec**2),0,ec.size))
for kind, data in scores.items():
    z = np.array(data)
    print(kind,'worst bridge/load',z[:,:2].max(axis=0),'pooled bridge/load RMSE',np.sqrt(z[:,2:4].sum(axis=0)/z[:,4].sum()))
f,p = _fit(c)
operating = np.vstack([ds[s][k].reshape(-1,4) for s in ('train','test') for k in ('past_channel_raw','hold_channel_raw')])
reference = _inverse(operating,f,p)
# Move the observed spike pattern onto clean calibration locations; check whether
# upstream estimates are stable when a different set of measurements is lost.
spikes = (y-_forward(x,f,p))[~clean]
nominal = _forward(x,f,p)
worst = 0.
for i in range(20):
    b = y.copy()
    b[~clean] = nominal[~clean]
    k = rng.choice(np.flatnonzero(clean),len(spikes),replace=False)
    b[k] += spikes
    ff,pp = _fit(dict(c,bridge_raw=b))
    worst = max(worst,np.max(abs(_inverse(operating,ff,pp)-reference)))
print('relocated observed spikes: worst operating load shift',worst)
assert worst < 0.003
print('calibration fields',sorted(c),'training fields',sorted(ds['train']))
