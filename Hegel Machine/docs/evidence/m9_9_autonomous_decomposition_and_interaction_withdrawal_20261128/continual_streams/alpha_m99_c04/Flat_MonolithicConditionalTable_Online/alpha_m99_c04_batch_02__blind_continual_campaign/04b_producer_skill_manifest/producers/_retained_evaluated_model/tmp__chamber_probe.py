import pickle,runpy,numpy as np
from scipy.optimize import least_squares
I=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; D=I['dataset']; G=I['campaign_diagnostics']; T=D['train']; E=D['test']; t=D['eval_times']; dt=D['past_dt']; P=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c04_batch_01_solver.py'); s=P['_stress_map'](D['calibration_load_cells']); xt,xe=P['_features'](T,E); tau,theta=P['_fit'](T,xt,s,dt,t)
cat=lambda *groups:{key:np.concatenate([g[key] for g in groups]) for key in ('lab_id','stratum','coupon_features_raw','ref_coupons')}
X={'diag':P['_features'](G,G)[0],'train+diag':P['_features'](T,G)[1],'all':P['_features'](cat(T,E),G)[1]}
# Apply deployment's actual normalization constants to the diagnostic rows.
base=cat(T,E); xd=G['coupon_features_raw'].copy(); xb=base['coupon_features_raw'].copy()
for l in np.unique(base['lab_id']):
 m=base['lab_id']==l; rr=base['ref_coupons'][m]; mu=np.median(rr.mean(0)); sd=np.median(rr.std(0)); xd[G['lab_id']==l]=(xd[G['lab_id']==l]-mu)/sd; xb[m]=(xb[m]-mu)/sd
_,ix,nn=np.unique(np.c_[base['lab_id'],base['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/nn[ix,None]; xd=np.where(np.isfinite(xd),xd,np.nansum(w*xb,axis=0)/np.sum(w*np.isfinite(xb),axis=0)); X['deploy']=xd
h=s[0]+G['hold_channel_raw']@s[1:]; p=s[0]+G['past_channel_raw']@s[1:]; age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]; z=p@(-np.expm1(-dt[:,None]/tau)*np.exp(-age[:,None]/tau)); y=G['strain_curves_obs']-G['strain_0'][:,None]; rate=G['instant_rate_obs']; v=np.c_[y,rate]; n=len(h); groups=G['lab_id']; basis=lambda u:np.vstack([np.c_[-np.expm1(-t[:,None]/u),t],np.r_[1/u,1]]); B=basis(tau)
print('taus, theta:',tau,theta,'missing diagnostic',np.isnan(G['coupon_features_raw']).sum(0)); print('h, p ranges',h.min(),h.max(),p.min(),p.max()); print('basis singular values',np.linalg.svd(B,compute_uv=False));
for use_rate in (False,True):
 b=B if use_rate else B[:-1]; a=np.linalg.lstsq(b,v[:,:len(b)].T,rcond=None)[0].T; print('unrestricted',use_rate,'rmse curves/rates',np.sqrt(np.mean((a@B.T-v)**2,axis=0)).round(4));
 for label,x in X.items():
  r=theta[2]+x@theta[3:];
  if use_rate: print('corr',label,np.corrcoef(np.c_[a,h,z,r,x].T)[:3,3:].round(2))
  for name,F in [('h,r',np.c_[h,r]),('h,z,r',np.c_[h,z,r]),('h,z,r,1',np.c_[h,z,r,np.ones(n)]),('h,z,x',np.c_[h,z,np.ones(n),x]),('h,p,r',np.c_[h,p,r]),('h,z,r,products',np.c_[h,z,r,h*r,z*r[:,None],h*h,z*z,r*r]),('h,z,r,quadratic',np.c_[h,z,r,np.ones(n),h*r,z*r[:,None],h*h,z*z,r*r,h[:,None]*z,z[:,0]*z[:,1]])]:
   res=np.empty_like(v)
   for l in np.unique(groups):
    m=groups!=l; c=np.linalg.lstsq(F[m],a[m],rcond=None)[0]; res[~m]=F[~m]@c@B.T-v[~m]
   c=np.linalg.lstsq(F,a,rcond=None)[0]; er=F@c@B.T-v; sc=lambda q:np.sqrt(np.mean(q*q,axis=0)); score=[np.sqrt(np.mean(res[:,:-1]**2)),sc(res)[-1],max(np.sqrt(np.mean(res[groups==l,:-1]**2)) for l in np.unique(groups))]; print(label,name,np.round(score,4),np.round([np.sqrt(np.mean(er[:,:-1]**2)),sc(er)[-1]],4),c.round(3).tolist() if name in ('h,r','h,z,r') else '')
# Free chamber rates of decay, estimated from observable trajectories, independent of the material fit.
for use_rate in (False,True):
 vv=v if use_rate else y
 def fun(lu):
  bb=basis(np.exp(lu))[:vv.shape[1]]; return (vv-np.linalg.lstsq(bb,vv.T,rcond=None)[0].T@bb.T).ravel()
 o=least_squares(fun,np.log(tau),bounds=(np.log(.05),np.log(25))); print('profiled basis',use_rate,np.exp(o.x),np.sqrt(np.mean(o.fun**2)))
