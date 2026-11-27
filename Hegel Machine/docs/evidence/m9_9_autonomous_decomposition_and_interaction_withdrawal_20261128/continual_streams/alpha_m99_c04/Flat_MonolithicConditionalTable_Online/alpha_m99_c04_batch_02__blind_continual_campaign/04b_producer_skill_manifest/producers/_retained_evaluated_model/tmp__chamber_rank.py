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
from scipy.optimize import least_squares
raw=G['coupon_features_raw']; miss=np.isnan(raw); xd=X['diag']; sc=np.r_[np.full(len(t),.004),.006]; prior=((h[:,None]-z)*theta[:2])@B[:,:2].T; target=v-prior; r=theta[2]+xd@theta[3:]
for label,sel in [('all',np.ones(n,bool)),('complete',~miss.any(1))]:
 for flags in [False,True] if label=='all' else [False]:
  def fit(m):
   def f(a):
    w=r+miss@np.r_[0.,a[3],0.,a[4]] if flags else r
    return w[:,None]*(B@a[:3])[None,:]-target
   a=least_squares(lambda a:(f(a)[m]/sc).ravel(),[.33,2.5,-.26]+([0.,0.] if flags else []),loss='soft_l1',f_scale=1.5).x
   return a,f(a)
  a,e=fit(sel); q=np.empty_like(v)
  for lab in np.unique(groups):
   m=groups==lab; q[m]=fit(sel&~m)[1][m]
  print(label,flags,'n',sel.sum(),'coeff',a.round(6),'RMSE by lab',[(round(np.sqrt(np.mean(q[groups==l,:-1]**2)),5),round(np.sqrt(np.mean(q[groups==l,-1]**2)),5)) for l in np.unique(groups)])
  latent=target@((B@a[:3])/sc**2)/np.sum(((B@a[:3])/sc)**2)-r; print("latent drift residual",[(key,latent[m].round(4).tolist()) for key,m in [("complete",~miss.any(1)),("missing 1",miss[:,1]),("missing 3",miss[:,3])]])
print('diagnostic imputation vector',[(j,np.unique(xd[miss[:,j],j]).tolist(),float(np.nanmean(np.where(miss[:,j],np.nan,xd[:,j])))) for j in (1,3)])
