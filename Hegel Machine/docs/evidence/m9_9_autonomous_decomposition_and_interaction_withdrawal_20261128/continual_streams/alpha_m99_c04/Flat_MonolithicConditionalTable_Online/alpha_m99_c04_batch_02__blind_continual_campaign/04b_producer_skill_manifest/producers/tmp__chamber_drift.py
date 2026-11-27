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
sc=np.r_[np.full(len(t),.004),.006]; prior=((h[:,None]-z)*theta[:2])@B[:,:2].T; target=(v-prior)/sc
for name, F in [('intrinsic',np.c_[np.ones(n),X['diag']]@theta[2:]),('free features',np.c_[np.ones(n),X['diag']]),('free features + missing flags',np.c_[np.ones(n),X['diag'],np.isnan(G['coupon_features_raw'])[:,[1,3]]])]:
 F=F.reshape(n,-1); C=(F[:,None,:,None]*(B/sc[:,None])[None,:,None,:]).reshape(n,len(B),-1)
 pr=np.zeros_like(v); coeff=[]
 for lab in list(np.unique(groups))+[-1]:
  m=groups!=lab; a=np.linalg.lstsq(C[m].reshape(-1,C.shape[-1]),target[m].ravel(),rcond=None)[0]; coeff.append(a.reshape(-1,3));
  if lab>=0: pr[~m]=(C[~m]@a-target[~m])*sc
 print(name,'full coeff',coeff[-1].round(5).tolist(),'worst curve/rate RMSE',np.max([np.sqrt(np.mean(pr[groups==l,:-1]**2)) for l in np.unique(groups)]),np.max([np.sqrt(np.mean(pr[groups==l,-1]**2)) for l in np.unique(groups)]))
print('intrinsic weight direction',theta[2:]/theta[2])
