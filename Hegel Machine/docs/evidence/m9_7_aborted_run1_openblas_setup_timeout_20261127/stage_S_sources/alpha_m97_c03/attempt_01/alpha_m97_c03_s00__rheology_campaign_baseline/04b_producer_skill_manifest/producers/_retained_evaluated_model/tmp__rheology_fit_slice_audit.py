import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c03_s00.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
tr,te = d['train'],d['test']
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
g,b = m._calibration(d['calibration_load_cells'])
s = tr['past_stress_cmd']*g[lab,None]+b[lab,None]
h = tr['stress_hold_cmd']*g[lab]+b[lab]
x = m._align(tr,te)[:len(lab)]
t,dt = d['eval_times'],d['past_dt']
y = np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
res = {k:np.empty_like(y) for k in ('all rows','complete fit')}
for j in np.unique(lab):
    val = lab==j
    for name,r in res.items():
        fit = ~val if name=='all rows' else ~val & complete
        p,w = m._fit(s[fit],h[fit],dt,t,x[fit],y[fit])
        a,_ = m._design(p,s[val],h[val],dt,t,x[val])
        r[val] = y[val] - a@w
        rc = r[val & complete]
        print(j,name,'tau',np.round(np.exp(p),5),'median signed strain/rate',np.round([np.median(rc[:,:-1]),np.median(rc[:,-1])],6),'median absolute early/late/rate',np.round([np.median(abs(rc[:,:3])),np.median(abs(rc[:,-4:-1])),np.median(abs(rc[:,-1]))],6))
for name,r in res.items():
    for status in (True,False):
        vals=[]
        for j in np.unique(lab):
            e=abs(r[(lab==j)&(complete==status)])
            if len(e): vals.append([np.median(e[:,:-1]),np.median(e[:,-1]),np.quantile(e[:,:-1],.9),np.quantile(e[:,-1],.9)])
        print(name,'complete',status,'worst lab medians and P90s',np.round(np.max(vals,axis=0),6))
