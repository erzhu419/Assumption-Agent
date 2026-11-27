exec(open('/tmp/chamber_probe.py').read().split("print('tau theta'")[0]); from itertools import product
keys=('coupon_features_raw','ref_coupons','lab_id','stratum');base={k:np.concatenate([tr[k],te[k]]) for k in keys}; past,hold,_=m._specimen(g,np.zeros((len(g['lab_id']),4)),bm); n=len(hold); labs=g['lab_id']; obs=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; w=np.r_[np.ones(len(t)),.65]
def norm(panel):
 out=np.empty_like(g['coupon_features_raw']); bp=np.empty_like(panel['coupon_features_raw']);
 for l in np.unique(labs):
  mask=panel['lab_id']==l; ref=panel['ref_coupons'][mask]; loc=np.median(ref.mean(0));scale=np.median(ref.std(0));out[labs==l]=(g['coupon_features_raw'][labs==l]-loc)/scale;bp[mask]=(panel['coupon_features_raw'][mask]-loc)/scale
 return out,bp
def impute(x,panel):
 _,cell,counts=np.unique(np.c_[panel['lab_id'],panel['stratum']],axis=0,return_inverse=True,return_counts=True);ww=1/counts[cell,None]; return np.nansum(ww*x,axis=0)/(ww*np.isfinite(x)).sum(0)
allg={k:np.concatenate([base[k],g[k]]) for k in keys}; vals={}
for name,panel in [('train',tr),('test',te),('base',base),('all',allg),('diag',g)]:
 x,bp=norm(panel)
 for missing,fill in [('self',impute(bp,panel)),('zero',0),('bayes',np.nanmean(bp,axis=0))]:
  xx=np.where(np.isfinite(x),x,fill);vals[name+':'+missing]=th[2]+xx@th[3:]
def K(tt): return np.c_[-np.expm1(-t[:,None]/tt),t].tolist()+[[*(1/tt),1.]]
def metric(e):return [*(round(a,6) for a in (np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)),max(np.sqrt(np.mean(e[labs==l]**2)) for l in np.unique(labs))))]
rows=[]
B=np.array(K(tau))[:,:2]
for name,D in vals.items():
 for typ in ('fixed_theta_tau','shared_scale','free_drift_basis','free_hold_and_drift','free_features','fixed_theta_tau_plus_drift'):
  H=hold[:,None]*th[:2]; X0=H@B.T; Q=D[:,None,None]*B[None,:,:]; Xd=Q@(th[:2]*tau); target=obs-X0
  if typ=='fixed_theta_tau': X=np.zeros((n,len(w),0));target-=Xd
  elif typ=='fixed_theta_tau_plus_drift': X=D[:,None,None]*np.r_[t,1.][None,:,None];target-=Xd
  elif typ=='shared_scale': X=Xd[:,:,None]
  elif typ=='free_drift_basis': X=Q
  elif typ=='free_hold_and_drift': X=np.concatenate((Q,hold[:,None,None]*B[None,:,:]),2)
  else: continue
  def fit(mask):
   xx=(X[mask]*w[None,:,None]).reshape(mask.sum()*len(w),X.shape[-1]); return np.linalg.lstsq(xx,(target[mask]*w).ravel(),rcond=None)[0]
  coef=fit(np.ones(n,dtype=bool)); e=X@coef-target; cv=e.copy()
  for l in np.unique(labs):
   mask=labs==l; cv[mask]=X[mask]@fit(~mask)-target[mask]
  rows.append((metric(cv)[-1],name,typ,metric(e),metric(cv),coef))
for row in sorted(rows,key=lambda r:r[0])[:18]: print(*row)
for row in rows:
 if row[2]=='fixed_theta_tau' and row[1].endswith(':self'): print('fixed',*row)
Q=(B@(th[:2]*tau)); implied=((obs-hold[:,None]*(B@th[:2]))*(w*w*Q)).sum(1)/np.sum((w*Q)**2)
print('diagnostic rows lab/missing/implied-D/base-D/all-D/diag-D/difference');print(np.round(np.c_[labs,np.isnan(g['coupon_features_raw']).sum(1),implied,vals['base:self'],vals['all:self'],vals['diag:self'],implied-vals['all:self']],5))
print('past correlations with best drift projection residual',np.corrcoef(np.c_[implied-vals['all:self'],hold,past],rowvar=False)[0,1:]);
