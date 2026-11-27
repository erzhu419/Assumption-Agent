import pickle,numpy as np
from scipy.optimize import least_squares
from m98core import _bridge,_loads,_invert,_bridge_basis,_features,_fit,_design,_mad
it=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=it['dataset']; tr,te=d['train'],d['test']; q=it['campaign_diagnostics']; cal=d['calibration_load_cells']; dt,t=d['past_dt'],d['eval_times']; b=_bridge(cal)
print('bridge coefficients',b.round(5))
for f in [tr,te,q]:
 r=f['past_channel_raw'].reshape(-1,4); l=_invert(r,b); print('inverse residual max',np.max(abs(np.einsum('ncp,cp->nc',_bridge_basis(l),b)-r)))
x=_features(tr,te)[:len(tr['lab_id'])]; tau,th=_fit(x,*_loads(tr,b),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('tau/theta',tau,th)
a,r=_design(x,*_loads(tr,b),dt,t,tau); print('train MAD',_mad((a@th).reshape(-1,len(t))+tr['strain_0'][:,None]-tr['strain_curves_obs']),_mad(r@th-tr['instant_rate_obs']))
p,h=_loads(q,b); y=q['strain_curves_obs']-q['strain_0'][:,None]; yr=q['instant_rate_obs']; rho=np.r_[np.cumsum(dt[:0:-1])[::-1],0.]; gap=h[:,None]-p@(-np.expm1(-dt[:,None]/tau)*np.exp(-rho[:,None]/tau));
allblock={k:np.concatenate([tr[k],te[k]]) for k in ('lab_id','stratum','ref_coupons','coupon_features_raw')}
for label,x in [('test+diag',_features(te,q)[-len(h):]),('train+diag',_features(tr,q)[-len(h):]),('all+diag',_features(allblock,q)[-len(h):]),('diag only',_features(q,q)[-len(h):])]:
 dr=th[2]+x@th[3:]; A,B=_design(x,p,h,dt,t,tau); print(label,'baseline',np.sqrt(np.mean((A@th-y.ravel())**2)),np.sqrt(np.mean((B@th-yr)**2)))
 scale=np.r_[np.full(y.size,.01),np.full(len(h),.025)]
 def analyze(name,cols,fixed=False):
  def mat(tau):
   C=-np.expm1(-t[:,None]/tau); D=np.c_[C,t]; E=np.r_[1/tau,1]; return np.vstack((np.concatenate([D[None,:,i:i+1]*v[:,None,:] for i,v in enumerate(cols)],axis=2).reshape(y.size,-1), np.column_stack([e*v for e,v in zip(E,cols)])))
  target=np.r_[y.ravel(),yr]; m=mat(tau); co=np.linalg.lstsq(m/scale[:,None],target/scale,rcond=None)[0];
  if fixed: f=least_squares(lambda c:(m@c-target)/scale,co,loss='soft_l1'); tv=tau; c=f.x
  else: f=least_squares(lambda v:(mat(np.exp(v[:2]))@v[2:]-target)/scale,np.r_[np.log(tau),co],loss='soft_l1',max_nfev=200); tv=np.exp(f.x[:2]); c=f.x[2:]
  e=mat(tv)@c-target; print(name,'tau',tv.round(5),'coef',c.round(5),'rmse',np.sqrt(np.mean(e[:y.size]**2)).round(5),np.sqrt(np.mean(e[y.size:]**2)).round(5),'mad',_mad(e[:y.size]).round(5),_mad(e[y.size:]).round(5))
 hd=np.c_[h,dr]; gd=np.c_[gap,h,dr,np.ones(len(h))]; dd=np.c_[dr,h,np.ones(len(h))]
 for name,cols in [('hold/drift',[hd]*3),('memory/drift',[np.c_[gap[:,0],h,dr],np.c_[gap[:,1],h,dr],dd]),('full stress',[np.c_[h,p,dr]]*3)]:
  analyze(name+' fixed',cols,True); analyze(name+' free',cols)
