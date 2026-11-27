import pickle,runpy
import numpy as np
from scipy.optimize import least_squares
m=runpy.run_path('/holdout_task/batch_solver.py')
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
s=i['dataset']; tr,te,d=s['train'],s['test'],i['campaign_diagnostics']; t,dt=s['eval_times'],s['past_dt']
b=m['_bridge'](s['calibration_load_cells']); x,xe=m['_features'](tr,te); p,h=m['_loads'](tr,b)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; lab=tr['lab_id']; labs=np.unique(lab)
def residual(v):
    a,c=m['_basis'](np.exp(v[:2]),p,h,x,dt,t)
    return a@v[2:]-y,c@v[2:]-r
def fit(mask,loss,cs,rs):
    z={k:v[mask] for k,v in tr.items()}; tt,w=m['_material'](z,b,x[mask],dt,t); v=np.r_[np.log(tt),w]
    if loss=='baseline': return v
    def fun(v):
        ee,er=residual(v); return np.r_[ee[mask].ravel()/cs,er[mask]/rs]
    return least_squares(fun,v,loss=loss,bounds=(np.r_[np.log([.050001]*2),[-np.inf]*7],np.r_[np.log([24.99999]*2),[np.inf]*7])).x
for loss,cs,rs in [('baseline',1,1),('cauchy',.0045,.003),('cauchy',.003,.0015),('soft_l1',.0045,.003)]:
    v=fit(lab==lab,loss,cs,rs); e,z=residual(v); print(loss,cs,rs,'full spectrum',np.exp(v[:2]).round(5),'weights',v[2:].round(5),'abs resid q50/q75/q90',np.quantile(abs(e),[.5,.75,.9]).round(5),np.quantile(abs(z),[.5,.75,.9]).round(5))
    fits=[]; metrics=[]
    for j in labs:
        q=lab==j; vf=fit(~q,loss,cs,rs); fits.append(np.exp(vf[:2])); ee,er=residual(vf); metrics.append([np.median(abs(ee[q])),np.median(abs(er[q])),np.quantile(abs(ee[q]),.75)])
    print(' spectrum folds',np.array(fits).round(5).tolist(),'worst held-lab robust errors',np.max(metrics,axis=0).round(6))
