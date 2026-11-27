import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d,q=item['dataset'],item['campaign_diagnostics']; tr,te=d['train'],d['test']; t,dt=d['eval_times'],d['past_dt']; n=len(tr['lab_id'])
x=m._features(tr,te); bridge=m._bridge(d['calibration_load_cells']); p,h=m._loads(tr,bridge)
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    v=tr['lab_id']==lab; f=~v
    tau,th=m._fit(x[:n][f],p[f],h[f],dt,t,y[f],r[f]); a,b=m._design(x[:n][v],p[v],h[v],dt,t,tau)
    ec=abs((a@th).reshape(y[v].shape)-y[v]); er=abs(b@th-r[v]); print('benchtop lab',lab,'counts',v.sum(),'taus',tau.round(4),'median/p90 strain,rate',np.quantile(ec,[.5,.9]).round(5),np.quantile(er,[.5,.9]).round(5))
out=m.solve_batch(d); assert all(np.all(np.isfinite(a)) for a in out.values()); tau=out['estimated_taus']; th=out['hereditary_weights']; assert .05<tau[0]<tau[1]<25
p,h=m._loads(q,bridge); _,ht=m._loads(te,bridge); dr=th[2]+m._features(q,q)[-len(h):]@th[3:]; drt=th[2]+x[n:]@th[3:]; y=q['strain_curves_obs']-q['strain_0'][:,None]; r=q['instant_rate_obs']; lab=q['lab_id']
print('diagnostic and deployment ranges [hold, drift]',[(a.min(0).round(4),a.max(0).round(4)) for a in (np.c_[h,dr],np.c_[ht,drt])]); print('rates and slopes are continuous, no class labels or relay modes in these measurements')
z=np.zeros((len(h),2));
for delta,past in zip(dt,p.T): z=np.exp(-delta/tau)*z-np.expm1(-delta/tau)*past[:,None]
base=np.c_[h,dr]
models={'linear':base,'memory differences':np.c_[base,h[:,None]-z], 'interaction/cubic':np.c_[base,h*dr,h**3,dr**3], 'rational':np.c_[base,h/(1+(h/3)**2),dr/(1+(dr/.35)**2)]}
for k,a in models.items():
    results=[]
    for l in np.unique(lab):
        v=lab==l; f=~v; scale=np.linalg.norm(a[f],axis=0); xq=a/scale
        def predict(vv):
            tc=np.exp(vv[:2]); amp=xq@vv[2:].reshape(3,-1).T
            return amp@np.vstack((-np.expm1(-t[None,:]/tc[:,None]),t)),amp@np.r_[1/tc,1.]
        def residual(vv):
            yc,rc=predict(vv)
            return np.r_[((yc-y)[f]/.005).ravel(),(rc-r)[f]/.004]
        init=np.zeros((3,a.shape[1])); init[:,:2]=m._CHAMBER_COEFFICIENTS*scale[:2]; zz=least_squares(residual,np.r_[np.log(tau*[1,.55]),init.ravel()],loss='soft_l1',max_nfev=160).x
        yp,rp=predict(zz); results.append([np.mean((yp[v]-y[v])**2)**.5,np.mean((rp[v]-r[v])**2)**.5])
    print(k,'per lab RMSE',np.round(results,6).tolist(),'worst',np.max(results,axis=0).round(6))
