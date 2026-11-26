import importlib.util,pickle
import numpy as np
p='/holdout_task/'
s=importlib.util.spec_from_file_location('candidate',p+'alpha_w03_batch_04_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
record=pickle.load(open(p+'public_alpha_w03_batch_04.pkl','rb'))[0]; d=record['dataset']; gt=record['gt']; tr=d['train']; te=d['test']; n=len(tr['lab_id']); t=d['eval_times']; dt=d['past_dt']; y=d['yield_stress']; x,h,z=m._prep(d)
obs=tr['strain_curves_obs']; rate=tr['instant_rate_obs']; start=tr['strain_0']; lab=tr['lab_id']; missing=np.isnan(tr['coupon_features_raw']).any(1); yielded=(abs(h[:n])>y)|(abs(z[:n]).max(1)>y)
pred=np.empty_like(obs); pr=np.empty(n)
for l in np.unique(lab):
    i=np.flatnonzero((lab!=l)&~missing); j=np.flatnonzero(lab==l)
    tau,coef=m._fit(x[i],h[i],z[i],dt,t,y,obs[i]-start[i,None],rate[i]); c,r=m._design(tau,x[j],h[j],z[j],dt,t,y)
    pred[j]=start[j,None]+c@coef; pr[j]=r@coef; print('LOLO',l,'tau',tau,'plastic',coef[7:])
def rm(a): return np.sqrt(np.mean(a*a))
for name, j in [('all',np.ones(n,dtype=bool)),('complete',~missing),('missing',missing),('subyield',~yielded),('yielded',yielded)]+[(f'lab{l}/yield={q}/missing={a}',(lab==l)&(yielded==q)&(missing==a)) for l in np.unique(lab) for q in [False,True] for a in [False,True]]:
    if j.any():
        e=pred[j]-obs[j]; delta=rate[j]-pr[j]; print(name,j.sum(),'curve/rate/early/late',*[round(v,8) for v in (rm(e),rm(delta),rm(e[:,0]),rm(e[:,-1]))], 'curve with missing-only rate innovation',round(rm(e+.95*missing[j,None]*delta[:,None]*t),8))
out=m.solve_alpha_w03_batch_04(d)
print('public reference keys and shapes',[(k,np.shape(v)) for k,v in gt.items()] if isinstance(gt,dict) else type(gt))
for k,v in out.items():
    if k in gt and np.shape(gt[k])==np.shape(v):
        e=v-gt[k]; print('public reference',k,'RMSE/max',rm(e),np.max(abs(e))); 
    if v.ndim==1 and len(v)<10: print(k,v)
miss=np.isnan(te['coupon_features_raw']).any(1)
if 'predicted_strain' in gt and np.shape(gt['predicted_strain'])==np.shape(out['predicted_strain']):
    err=out['predicted_strain']-gt['predicted_strain']; dr=te['instant_rate_obs']-out['initial_rate_pred'];
    for l in np.unique(te['lab_id']):
        for b in (False,True):
            j=(te['lab_id']==l)&(miss==b)
            if j.any(): print('public curve lab/missing',l,b,j.sum(),rm(err[j]),rm(err[j]+.95*miss[j,None]*dr[j,None]*t))
