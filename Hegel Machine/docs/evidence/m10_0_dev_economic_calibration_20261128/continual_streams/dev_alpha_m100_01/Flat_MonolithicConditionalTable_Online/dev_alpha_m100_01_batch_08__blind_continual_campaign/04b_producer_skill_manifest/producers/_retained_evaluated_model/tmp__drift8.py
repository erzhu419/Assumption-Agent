import sys,pickle,importlib.util,numpy as np
from scipy.optimize import least_squares
root='/tmp/hegel_continual_archive/'
def load(i):
 s=importlib.util.spec_from_file_location('solver',root+f'dev_alpha_m100_01_batch_{i:02d}_solver.py'); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load(7); import cubic8 as cb; m._bridge=cb._bridge; m._loads=cb._loads; item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; tr,te,d=ds['train'],ds['test'],item['campaign_diagnostics']; t,dt=ds['eval_times'],ds['past_dt']; n=len(tr['lab_id']); x=m._features(tr,te); xd=m._features(tr,d)[n:]; b=m._bridge(ds['calibration_load_cells']);  p,h=m._loads(tr,b)
def obs(q):return q['strain_curves_obs']-q['strain_0'][:,None],q['instant_rate_obs']
c,r=obs(tr); T,w=m._fit(p,h,x[:n],dt,t,c,r); print('TAUS',T,'THETA',w,'bridge',b[0],b[1].round(5));
def stats(label,curve,rate,c,r):
 e=curve-c; f=rate-r; print(label,'rmse',np.sqrt(np.mean(e**2)),np.sqrt(np.mean(f**2)),'median',np.median(abs(e)),np.median(abs(f)),'curve by time',np.sqrt(np.mean(e**2,axis=0)).round(4))
A,B=m._design(T,p,h,x[:n],dt,t);stats('train',A@w,B@w,c,r)
p,h=m._loads(d,b); c,r=obs(d); xd2=m._features(tr,te) # retain deployment centering for the verification comparison below
labs=np.r_[tr['lab_id'],te['lab_id']]; refs=np.r_[tr['ref_coupons'],te['ref_coupons']]; xd3=d['coupon_features_raw'].copy()
for lab in np.unique(d['lab_id']):
 panel=refs[labs==lab]; q=d['lab_id']==lab
 if len(panel): xd3[q]=(xd3[q]-np.median(panel.mean(0)))/np.median(panel.std(0))
labcell=np.column_stack((labs,np.r_[tr['stratum'],te['stratum']])); _,idx,ct=np.unique(labcell,axis=0,return_inverse=True,return_counts=True); sel=np.isfinite(np.r_[tr['coupon_features_raw'],te['coupon_features_raw']]); means=(xd2*sel/ct[idx,None]).sum(0)/(sel/ct[idx,None]).sum(0); xd3=np.where(np.isfinite(xd3),xd3,means)
print('feature disagreements',np.max(abs(xd3-xd),axis=0),'diag missing',np.isnan(d['coupon_features_raw']).sum(0)); print('labs/strata',d['lab_id'],d['stratum']); print('deployment max/min',x[n:].min(0),x[n:].max(0))
R=np.column_stack((-np.expm1(-t[:,None]/T),t)); S=np.r_[1/T,1.]; H=h[:,None]*w[:2]@R[:,:2].T; I=h[:,None]*w[:2]@S[:2]; y=np.column_stack((c-H,r-I));
for xdd in [xd,xd3]:
 g=w[2]+xdd@w[3:]; coeff=np.array([.14,2.8,.2]); A=g[:,None,None]*np.vstack((R,S)); E=A@coeff-y
 print('rounded errors c|r',np.column_stack((g,np.sqrt(np.mean(E[:,:-1]**2,1)),E[:,-1])).round(5));
 for mode in ['linear','robust','free-time','drift-affine','nonlinear-drift']:
  out=np.empty_like(y); pars=[]
  for k in range(len(g)+1):
   sel=np.arange(len(g))!=k if k<len(g) else np.ones(len(g),bool)
   def pred(q):
    if mode=='free-time':
     tt=np.exp(q[3]); return A@q[:3]+g[:,None]*q[4]*np.r_[-np.expm1(-t/tt),1/tt]
    if mode=='drift-affine': return A@q[:3]+np.vstack((R,S))@q[3:]
    if mode=='nonlinear-drift': return A@q[:3]+g[:,None]**2*(np.vstack((R,S))@q[3:])
    return A@q
   init=np.r_[coeff,0.,0.] if mode=='free-time' else np.r_[coeff,0.,0.,0.] if mode in ['drift-affine','nonlinear-drift'] else coeff
   fit=least_squares(lambda q:(pred(q)-y)[sel].ravel(),init,loss='linear' if mode=='linear' else 'cauchy',f_scale=.007,max_nfev=120); pars.append(fit.x)
   if k<len(g):out[k]=pred(fit.x)[k]
  e=out-y; print(mode,'params',pars[-1].round(6),'OOF overall',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),'median',np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])),'max-time',max(np.sqrt(np.mean(e[:,:-1]**2,0))))
