import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: item=pickle.load(f)[0]
d=item['dataset']; tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; lab=tr['lab_id']; complete=np.isfinite(tr['coupon_features_raw']).all(1)
b=m._bridge(d['calibration_load_cells']); p,h=m._loads(tr,b); x,_=m._features(tr,te); y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
def take(k): return {name:v[k] for name,v in tr.items()}
for option in ('current','complete-only','complete-only balanced noise'):
    scores=[]; spectra=[]
    for held in [None,*np.unique(lab)]:
        fit=np.ones(len(lab),bool) if held is None else lab!=held
        tau,theta=m._material(take(fit),b,x[fit],dt,t)
        if option!='current':
            k=complete&fit
            c0,r0=m._basis(tau,p[k],h[k],x[k],dt,t)
            def sigma(res): return max(1e-5,1.4826*np.median(abs(res-np.median(res))))
            sc,sr=(.004,.004) if option=='complete-only' else (sigma(c0@theta-y[k]),sigma(r0@theta-r[k]))
            def residual(q):
                c,rr=m._basis(np.exp(q[:2]),p[k],h[k],x[k],dt,t)
                return np.r_[((c@q[2:]-y[k])/sc).ravel(),(rr@q[2:]-r[k])/sr]
            q=least_squares(residual,np.r_[np.log(tau),theta],bounds=(np.r_[np.log([.050001]*2),[-np.inf]*7],np.r_[np.log([24.99999]*2),[np.inf]*7]),loss='soft_l1').x
            order=np.argsort(q[:2]); tau=np.exp(q[:2][order]); theta=np.r_[q[2:4][order],q[4:]]
        if held is None:
            print(option,'full tau',tau.round(5)); continue
        spectra.append(tau)
        k=lab==held; c,rr=m._basis(tau,p[k],h[k],x[k],dt,t); ec=abs(c@theta-y[k]); er=abs(rr@theta-r[k]); j=complete[k]
        scores.append([ec.mean(),er.mean(),np.median(ec[j]),np.median(er[j])])
    print(option,'worst lab curve/rate MAE, complete curve/rate median',np.max(scores,axis=0).round(6),'held-out-fit tau span',np.ptp(spectra,axis=0).round(5))
