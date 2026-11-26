import importlib.util, pickle
import numpy as np

p = '/holdout_task/'
s = importlib.util.spec_from_file_location('solver', p+'alpha_w04_batch_04_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
c = pickle.load(open(p+'public_alpha_w04_batch_04.pkl','rb'))[0]
d, g = c['dataset'], c['gt']
tr, te = d['train'], d['test']
t, dt, y = (d[k] for k in ('eval_times','past_dt','yield_stress'))
aff = m._calibrate(d['calibration_load_cells'])
xtr, xte = m._align(tr, te)
raw = np.concatenate([tr['coupon_features_raw'],te['coupon_features_raw']])
finite = np.isfinite(raw)
xall = np.concatenate([xtr,xte]); labs = np.r_[tr['lab_id'],te['lab_id']]; strata = np.r_[tr['stratum'],te['stratum']]
ht, st = m._stress(tr, aff); he, se = m._stress(te, aff)
M = []
for wing in (0,1):
    train = tr['lab_id']//2 != wing
    rows = np.flatnonzero(te['lab_id']//2 == wing)
    tau, w = m._fit(dt,t,ht[train],st[train],xtr[train],(tr['strain_curves_obs']-tr['strain_0'][:,None])[train],tr['instant_rate_obs'][train],y)
    _, b = m._design(tau,dt,t,ht,st,xtr,y)
    rr = (tr['instant_rate_obs'] - b@w)[train & finite[:len(ht)].all(axis=1)]
    sigma = 1.4826*np.median(abs(rr-np.median(rr)))
    print('held wing',wing,'fit tau',tau,'vp',w[7:],'rate noise',sigma,'train rate residual quantiles',np.quantile(abs(rr),[.5,.95,1.]))
    for i in rows:
        k = i + len(ht)
        actual_missing = ~finite[k]; masksets = [actual_missing] if actual_missing.any() else list(np.eye(4,dtype=bool))
        for mask in masksets:
            x = xte[i].copy()
            cell = (labs==labs[k]) & (strata==strata[k]); cell[k] = False
            if not actual_missing.any():
                for j in np.flatnonzero(mask): x[j] = np.mean(xall[cell & finite[:,j],j])
            complete = cell & finite[:,mask].all(axis=1)
            spread = np.std(xall[complete][:,mask]@w[3:7][mask])
            a,b = m._design(tau,dt,t,he[i:i+1],se[i:i+1],x[None,:],y)
            base, rate = (a@w)[0]+te['strain_0'][i], (b@w)[0]
            resid = te['instant_rate_obs'][i]-rate
            corrections = [0., resid, (spread**2/(spread**2+sigma**2))*resid]
            for method, z in enumerate(corrections):
                e = base+z*t-g['te_clean'][i]; r = rate+z-g['te_true_rate0'][i]
                M.append((wing,actual_missing.any(),method,(e*e).mean(),r*r))
M=np.array(M)
for wing in (0,1):
    for real in (0,1):
        print('held wing / actual missing',wing,real,[(int((ix:=(M[:,0]==wing)&(M[:,1]==real)&(M[:,2]==method)).sum()),np.sqrt(M[ix,3:].mean(axis=0)).round(6).tolist()) for method in range(3)])
