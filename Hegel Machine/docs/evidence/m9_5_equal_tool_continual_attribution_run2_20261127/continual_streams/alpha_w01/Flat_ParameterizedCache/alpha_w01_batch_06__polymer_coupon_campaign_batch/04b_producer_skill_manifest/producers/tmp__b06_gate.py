import importlib.util,pickle,numpy as np
from itertools import combinations
p='/holdout_task/'
s=importlib.util.spec_from_file_location('b06',p+'alpha_w01_batch_06_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
d=pickle.load(open(p+'public_alpha_w01_batch_06.pkl','rb'))[0]['dataset']; tr=d['train']; lab=tr['lab_id']; t=d['eval_times']; dt=d['past_dt']; sy=d['yield_stress']; maps=m._calibrate(d['calibration_load_cells']); h,u=m._stress(tr,maps); ov,w=m._plastic(h,u,dt,sy)
x,_,complete,_=m._align(tr,d['test']); v=x[:,1:5]
# Add only terms outside the existing linear span. Fixed rational denominators
# keep this a small falsification probe, not a flexible replacement model.
z=np.column_stack([v[:,i]*v[:,j] for i,j in combinations(range(4),2)]+[v[:,i]/(1+.5*(v[:,i]+sign*v[:,j])**2) for i,j in combinations(range(4),2) for sign in (-1,1)])
y=tr['strain_curves_obs']-tr['strain_0'][:,None]; r=tr['instant_rate_obs']; caps=.12+.012*t

def residual(fit,held):
    part={k:a[fit] for k,a in tr.items()}; tau,b=m._fit(part,x[fit],complete[fit],maps,dt,t,sy)
    A,B=m._design(tau,h[held],u[held],x[held],dt,t,ov[held],w[held]); delta=r[held]-B@b
    return y[held]-A@b-np.where(complete[held],.25,1.)[:,None]*delta[:,None]*t

scores=[]; pooled=[]
for l in np.unique(lab):
    held=lab==l; fit=~held; e=residual(fit,held); pool=np.empty((fit.sum(),len(t))); lf=lab[fit]
    for j in np.unique(lf):
        zj=lab==j; pool[lf==j]=residual(fit&~zj,zj)
    # Estimate any missing drift using only inner-held-out complete coupons.
    q=complete[fit]; xf=x[fit][q]; zf=z[fit][q]; proj=np.linalg.lstsq(xf,zf,rcond=None)[0]; a=zf-xf@proj
    target=np.median(np.clip(pool[q],-caps,caps)/t,axis=1)
    b=np.linalg.solve(a.T@a+30*np.eye(a.shape[1]),a.T@target)
    pred=(z[held]-x[held]@proj)@b
    for above in (False,True):
        sl=(abs(u[held])>sy)==above
        base=e[sl]; alt=base-(pred[:,None]*t*complete[held,None])[sl];
        metric=lambda a: float(np.sqrt(np.mean(np.minimum(abs(a),caps)**2)))
        scores.append([metric(base),metric(alt)]); print('lab/hold-above',l,above,'baseline/nonlinear clipped RMSE',np.round(scores[-1],6))
    # Compare lab-level residual location only within matched yield strata.
    for above in (False,True):
        sl=complete[held]&((abs(u[held])>sy)==above)
        pooled.append([int(l),above,int(sl.sum()),float(np.median(e[sl,-1]))])
print('worst base/probe',np.max(scores,axis=0),'mean base/probe',np.mean(scores,axis=0))
print('complete-coupon within-lab/yield final-time median residuals',pooled)
