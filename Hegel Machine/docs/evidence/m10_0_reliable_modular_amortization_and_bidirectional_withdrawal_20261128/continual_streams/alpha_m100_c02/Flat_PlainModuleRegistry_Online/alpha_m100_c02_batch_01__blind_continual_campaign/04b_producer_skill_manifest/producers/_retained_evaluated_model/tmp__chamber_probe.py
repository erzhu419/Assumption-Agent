import sys,pickle,numpy as n
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive'); import alpha_m100_c02_s00_solver as m
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=p['dataset'];g=p['campaign_diagnostics'];tr,te=d['train'],d['test'];dt,t=d['past_dt'],d['eval_times'];c=d['calibration_load_cells']; P,H=m._loads(c,tr);X=m._features(tr,te);tau,th=m._fit(P,H,X[:len(H)],dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); past,h=m._loads(c,g);x=m._features(tr,{k:n.concatenate((te[k],g[k])) for k in ('lab_id','stratum','coupon_features_raw','ref_coupons')})[-len(h):]; y=n.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; _,r,*_=m._design(n.log(tau),past,h,x,dt,t); amp0=r[:,:2]*tau*th[:2];z=h[:,None]-amp0/th[:2]; drift=x@th[2:];
def basis(u): return n.vstack([n.c_[-n.expm1(-t[:,None]/u),t],n.r_[1/u,1.]])
def fit(F, mask, u=tau):
 b=basis(u); A=n.column_stack([(f[:,None,:]*b[None,:,j,None]).reshape(y.size,-1) for j,f in enumerate(F)]); mask=n.repeat(mask,y.shape[1]); a=n.linalg.lstsq(A[mask],y.ravel()[mask],rcond=None)[0];return (A@a).reshape(y.shape),n.split(a,n.cumsum([f.shape[1] for f in F])[:-1])
one=x[:,:1]; common=n.c_[one,past,h,x[:,1:]]; vars={'full':(common,common,common),'gap drift':(n.c_[one,amp0[:,0],past[:,-1]],n.c_[one,amp0[:,1],drift],n.c_[one,drift]),'z hold last drift':(n.c_[one,z[:,0],h,past[:,-1]],n.c_[one,z[:,1],h,drift],n.c_[one,drift]),'z hold last drift x':(n.c_[one,z[:,0],h,past[:,-1]],n.c_[one,z[:,1],h,x[:,1:]],x),'z hold last drift sq':(n.c_[one,z[:,0],h,past[:,-1],past[:,-1]**2],n.c_[one,z[:,1],h,drift,drift**2],n.c_[one,drift,drift**2])}
if __name__=='__main__':
 for u in (tau,[tau[0],2.73]):
  for k,F in vars.items():
   pred,a=fit(F,n.ones(len(h),bool),u); e=pred-y; q=n.empty_like(y)
   for l in n.unique(g['lab_id']):
    ix=g['lab_id']==l;q[ix]=fit(F,~ix,u)[0][ix]-y[ix]
   print(u,k,'coeff',[v.round(4).tolist() for v in a],'rms',n.sqrt(n.mean(e**2,axis=0))[[0,4,11,12]].round(4),'worst LOLO',n.max([n.sqrt(n.mean(q[g['lab_id']==i]**2,axis=0))[[0,4,11,12]] for i in range(4)],axis=0).round(4))
 print('past hold ranges',past.min(0),past.max(0),h.min(),h.max()); print('drift',drift.round(3)); print('refs means',[(n.mean(s['ref_coupons'],axis=0).round(2),n.isnan(s['coupon_features_raw']).sum(0)) for s in [tr,te,g]])
