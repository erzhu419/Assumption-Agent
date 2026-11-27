import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
s=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl','rb') as f: d=pickle.load(f)[0]['dataset']
tr,te=d['train'],d['test']; dt,t=d['past_dt'],d['eval_times']; x,_=m._features(tr,te); a=m._inputs(tr,x,m._calibration(d['calibration_load_cells'])); past,hold,X=a
Y=np.column_stack((tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']))
lab=tr['lab_id']; telem=tr['aux_chamber_telemetry']; tc=telem.copy()
for j in np.unique(lab):
    mask=lab==j; tc[mask]-=np.median(tc[mask],axis=0)
worst={}
for j in np.unique(lab):
    h=lab==j; f=~h; tau,th,scale=m._material(tr,a,dt,t,f); y,r=m._predict(a,dt,t,tau,th); E=Y-np.column_stack((y,r))
    norm=np.r_[np.full(len(t),scale[0]*np.sqrt(len(t))),scale[1]]
    basis=np.vstack((-np.expm1(-t[:,None]/tau),1/tau)); ramp=np.r_[t,1.]; gap=hold[:,None]-past@m._kernel(dt,tau)
    changes=np.diff(np.c_[past,hold],axis=1)
    den=1+np.sum((gap/np.maximum(np.median(np.abs(gap[f]),axis=0),1e-8))**2,axis=1)
    terms={'baseline':np.empty((len(h),len(t)+1,0)), 'history_differences':np.concatenate([changes[:,None,:]*basis[None,:,b,None] for b in range(2)],axis=2), 'saturating_branches':(gap/den[:,None])[:,None,:]*basis[None,:,:], 'curved_drift':np.stack((np.ones(len(h)),X@th[2:]),axis=1)[:,None,:]*np.r_[t*t/t[-1],0.][None,:,None], 'telemetry_raw':telem[:,None,:]*ramp[None,:,None], 'telemetry_within_lab':tc[:,None,:]*ramp[None,:,None]}
    k=len(t)//3
    for name,D in terms.items():
        error=E[h]
        if D.shape[-1]:
            A=(D[f]/norm[None,:,None]).reshape(-1,D.shape[-1]); b=(E[f]/norm).ravel(); sig=np.maximum(np.linalg.norm(A,axis=0),1e-8)
            fit=least_squares(lambda q: np.r_[A@(q/sig)-b, 0.2*q],np.zeros(len(sig)),loss='soft_l1',f_scale=0.6)
            error=error-D[h]@(fit.x/sig)
        e=np.abs(error); v=np.array([np.median(e[:,:k]),np.median(e[:,k:2*k]),np.median(e[:,2*k:-1]),np.median(e[:,-1])]); worst[name]=np.maximum(worst.get(name,np.zeros(4)),v)
        print(j,name,np.round(v,6))
print('WORST (early, middle, late, rate)')
for name,v in worst.items(): print(name,np.round(v,6))
