import sys
sys.path.insert(0, '/holdout_task')
import pickle
import numpy as np
from _verified_upstream import calibrate_upstream
from downstream_solver import _features, _loads, _fit, _design

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = calibrate_upstream(pickle.load(f)[0]['dataset'])
a,b=d['train'],d['test']; n=len(a['lab_id']); x=_features(a,b)[:n]; p,h=_loads(a); dt,t=d['past_dt'],d['eval_times']; y=a['strain_curves_obs']-a['strain_0'][:,None]; r=a['instant_rate_obs']; tele=a['aux_chamber_telemetry']
for group in ('lab_id', 'stratum'):
    for g in np.unique(a[group]):
        v=a[group]==g; f=~v
        tau,th=_fit(x[f],p[f],h[f],dt,t,y[f],r[f]); c,k=_design(tau,x[v],p[v],h[v],dt,t)
        e=c@th-y[v]; er=k@th-r[v]
        print(group,g,'n',v.sum(),'taus',tau.round(4),'curve p90 slices',np.quantile(abs(e).reshape(v.sum(),3,-1),.9,axis=(0,2)).round(5),'rate p90',round(np.quantile(abs(er),.9),5))
tau,th=_fit(x,p,h,dt,t,y,r); c,k=_design(tau,x,p,h,dt,t); e=c@th-y; er=k@th-r
print('Residual telemetry correlations',np.corrcoef(np.c_[np.clip(e,-.035,.035).mean(axis=1),er,tele].T)[:2,2:].round(3))
print('Test telemetry correlations with known channels')
p,h=_loads(b); v=np.c_[b['aux_chamber_telemetry'],h,p.mean(axis=1),b['strain_0']]; print(np.corrcoef(v.T)[:4,4:].round(3))
