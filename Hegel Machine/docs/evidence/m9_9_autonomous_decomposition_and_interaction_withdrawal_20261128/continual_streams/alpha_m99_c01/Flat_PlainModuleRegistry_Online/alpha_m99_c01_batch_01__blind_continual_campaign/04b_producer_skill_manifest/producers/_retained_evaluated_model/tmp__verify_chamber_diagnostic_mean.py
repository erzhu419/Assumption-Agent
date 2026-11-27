import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
item=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; ds=item['dataset']; d=item['campaign_diagnostics']; tr=ds['train']; te=ds['test']; t=ds['eval_times']; dt=ds['past_dt']; w=s._bridge(ds['calibration_load_cells']); fit=s.solve_batch(ds); u=fit['estimated_taus']; b=fit['hereditary_weights']; p,h=s._loads(d,w)
raw=d['coupon_features_raw']; X=raw.copy(); lab=d['lab_id']; ok=np.isfinite(X).all(1); cols=np.nonzero(~np.isfinite(X))[1]
for l in np.unique(lab):
 m=lab==l; R=d['ref_coupons'][m]; X[m]=(X[m]-np.median(R.mean(0)))/np.median(R.std(0))
_,inv,cnt=np.unique(np.c_[lab,d['stratum']],axis=0,return_inverse=True,return_counts=True); wt=1./cnt[inv]; center=np.nansum(X*wt[:,None],axis=0)/(np.isfinite(X).T@wt)
print('imputation alternatives',[(k,np.round(v,4)) for k,v in [('cell',center),('plain',np.nanmean(X,axis=0)),('complete',np.mean(X[ok],axis=0)),('median',np.nanmedian(X,axis=0)),('joint',s._features(tr,te)[0].mean(0))]])
center=np.nanmean(X,axis=0); X=np.where(np.isfinite(X),X,center); r=b[2]+X@b[3:]; y=np.c_[d['strain_curves_obs']-d['strain_0'][:,None],d['instant_rate_obs']]; T=np.r_[-np.expm1(-t[:,None]/u),[1/u]]; q=np.r_[t,1.]; Y=y-h[:,None]*(T@b[:2]); A=r[:,None,None]*np.c_[T,q];
def met(e): return np.r_[np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-2]**2)),np.sqrt(np.mean(e[:,-1]**2))]
for tag,ind in [('all',np.ones(len(d['lab_id']),bool)),('complete',ok)]:
 c=np.linalg.lstsq(A[ind].reshape(-1,3),Y[ind].ravel(),rcond=None)[0]; o=np.empty_like(Y); print('fit',tag,c,'errors',met((A@c-Y)[ind]));
 for l in np.unique(lab):
  m=lab==l; ii=ind&~m; v=np.linalg.lstsq(A[ii].reshape(-1,3),Y[ii].ravel(),rcond=None)[0]; o[m]=A[m]@v-Y[m]
 print('OOF',tag,'worst complete lab',np.max([met(o[(lab==l)&ok]) for l in np.unique(lab)],axis=0),'missing',met(o[~ok]))
 if tag=='complete':
  v=np.c_[T,q]@c; ri=Y@v/(v@v); print('inferred fill',center[cols]+(ri-r)[~ok]/b[3+cols]); c0=np.array([.39,1.83,.036]); print('coeff fixed vs fitted',c0,c)
cur,rate=s._chamber(u,b,X,h,t); e=np.c_[cur,rate]-y; print('artifact diagnostics worst complete lab',np.max([met(e[(lab==l)&ok]) for l in np.unique(lab)],axis=0),'missing',met(e[~ok])); print('u/b',u,b)
# Validate recovery from the measured benchtop data without each lab's responses.
x,xt=s._features(tr,te); pt,ht=s._loads(tr,w); yy=tr['strain_curves_obs']-tr['strain_0'][:,None]; rr=tr['instant_rate_obs']; labtr=tr['lab_id'];
for l in np.unique(labtr):
 m=labtr==l; uf,bf=s._fit(x[~m],pt[~m],ht[~m],dt,t,yy[~m],rr[~m]); a,g=s._design(uf,x[m],pt[m],ht[m],dt,t); print('benchtop lab',l,'tau',uf,'weights max shift',np.max(abs(bf-b)),'median errors',np.median(abs(a@bf-yy[m])),np.median(abs(g@bf-rr[m])))
assert 0.05<u[0]<u[1]<25.
# Initial rate must equal the derivative of the deployed forecast at t=0.
for eps in [1e-6,1e-7]:
 modified={**ds,'eval_times':np.r_[eps,t]}; cv,_=s._chamber(u,b,xt,s._loads(te,w)[1],modified['eval_times']); err=np.max(abs(cv[:,0]/eps-fit['initial_rate_pred'])); print('rate derivative max error',eps,err); assert err<1e-5
