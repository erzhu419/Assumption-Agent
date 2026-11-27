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
from itertools import product
r=theta[2]+X['diag']@theta[3:]; scales=np.r_[np.full(len(t),.004),.006]; gap=h[:,None]-z
print('test keys',[(k,np.shape(w)) for k,w in E.items()]); print('bridge error',np.std(np.c_[np.ones(len(D['calibration_load_cells']['bridge_raw'])),D['calibration_load_cells']['bridge_raw']]@s-D['calibration_load_cells']['true_channel_loads'].sum(1)))

def fit(m, time, memory, intercept):
 def design(lu):
  u=np.exp(lu) if time else tau
  b=basis(u); f=np.c_[r,np.ones(n)] if intercept else r[:,None]
  if memory=='fixed':
   prior=(gap*theta[:2])@b[:,:2].T
  elif memory=='zero':
   prior=(h[:,None]*theta[:2])@b[:,:2].T
  else:
   prior=(h[:,None]*theta[:2])@b[:,:2].T
   if memory=='scale': pass
   if memory=='cross': pass
  C=(f[:,None,:,None]*b[None,:,None,:]).reshape(n,len(b),-1)
  if memory=='scale': C=np.concatenate([C,-z[:,None,:]*b[None,:,:2]*theta[:2]],axis=2)
  if memory=='cross': C=np.concatenate([C,(-z[:,None,:,None]*b[None,:,None,:2]).reshape(n,len(b),4)],axis=2)
  return C,prior
 def evaluate(lu):
  C,prior=design(lu); cw=C/scales[None,:,None]; a=np.linalg.lstsq(cw[m].reshape(-1,C.shape[-1]),((v-prior)/scales)[m].ravel(),rcond=None)[0]; return (C@a+prior-v),a
 lu=np.log(tau)
 if time: lu=least_squares(lambda lu:(evaluate(lu)[0][m]/scales).ravel(),lu,bounds=(np.log([.1,.8]),np.log([.8,15])),max_nfev=150).x
 er,a=evaluate(lu); return er,np.exp(lu),a
for time,memory,intercept in product([False,True],['fixed','zero','scale','cross'],[False,True]):
 if memory=='zero' and (time or intercept):continue
 er,u,a=fit(np.ones(n,bool),time,memory,intercept); q=np.empty_like(v)
 for lab in np.unique(groups):
  mask=groups!=lab; q[~mask]=fit(mask,time,memory,intercept)[0][~mask]
 rms=lambda q:np.sqrt(np.mean(q*q,axis=0)); worst=max(np.sqrt(np.mean(q[groups==l,:-1]**2)) for l in np.unique(groups)); print(time,memory,intercept,'in',np.round([np.sqrt(np.mean(er[:,:-1]**2)),rms(er)[-1]],4),'CV',np.round([worst,rms(q)[-1]],4),'u',u.round(5),'a',a.round(5))
print('diag normalization medians/stds')
for gg in [T,E,G,base]:
 print([(l,round(np.median(gg['ref_coupons'][gg['lab_id']==l].mean(0)),3),round(np.median(gg['ref_coupons'][gg['lab_id']==l].std(0)),3)) for l in np.unique(gg['lab_id'])])
