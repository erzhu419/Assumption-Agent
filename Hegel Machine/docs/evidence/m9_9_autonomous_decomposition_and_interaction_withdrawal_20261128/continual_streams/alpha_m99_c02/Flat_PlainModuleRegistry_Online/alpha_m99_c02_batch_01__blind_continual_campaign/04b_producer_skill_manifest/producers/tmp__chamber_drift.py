import pickle,runpy,numpy as np
from scipy.optimize import least_squares
s=runpy.run_path('/tmp/hegel_continual_archive/alpha_m99_c02_s00_solver.py'); globals().update({k:v for k,v in s.items() if k.startswith('_') and not k.startswith('__')})
i=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]; d=i['dataset']; g=i['campaign_diagnostics']; tr=d['train']; te=d['test']; dt=d['past_dt']; t=d['eval_times']; b=_bridge(d['calibration_load_cells']); xt,xe=_features(tr,te); _,xg=_features({k:np.concatenate((tr[k],te[k])) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')},g); ta,th=_material(tr,b,xt,dt,t); p,h=_loads(g,b); cv,rt=_basis(ta,p,h,xg,dt,t); Y=np.c_[g['strain_curves_obs']-g['strain_0'][:,None],g['instant_rate_obs']]; lab=g['lab_id']; n=len(lab)
def T(q): return np.r_[np.c_[-np.expm1(-t[:,None]/q),t], [np.r_[1/q,1.]]]
def hist(q):
 z=np.zeros((n,len(q)))
 for w,a in zip(dt,p.T): z+=-np.expm1(-w/q)*(a[:,None]-z)
 return z
from itertools import product
M=T(ta); exp=M[:,:2]; base=h[:,None]*(exp@th[:2]); R=Y-base
# Estimate the dominant residual time profile and per-coupon scores directly.
u,s,v=np.linalg.svd(R,full_matrices=False); k=v[0]*np.sign(v[0,-1]); q=R@k; rem=R-q[:,None]*k
print('residual singular values',s[:7]); print('rank-one RMSE',np.sqrt(np.mean(rem**2,axis=0))); print('dominant temporal coefficients',np.linalg.lstsq(M,k,rcond=None)[0]); print('hold theta and fitted hold',th[:2],np.linalg.lstsq(np.c_[h,q],Y,rcond=None)[0][0]@np.linalg.pinv(M.T))
def join(*groups):
 return {k:np.concatenate([a[k] for a in groups]) for k in ('coupon_features_raw','ref_coupons','lab_id','stratum')}
raw=g['coupon_features_raw']; modes={'train+diag':_features(tr,g)[1],'train+test+diag':xg,'diag':_features(g,g)[0],'test+diag':_features(te,g)[1]}
for tag,a in [('from_train_test',join(tr,te)),('from_train',tr)]:
 x=raw.copy(); ref=a['ref_coupons']; labels=a['lab_id']; ax=a['coupon_features_raw'].copy()
 for l in np.unique(lab):
  rp=ref[labels==l]; center=np.median(rp.mean(0)); scale=np.median(rp.std(0)); x[lab==l]=(x[lab==l]-center)/scale; ax[labels==l]=(ax[labels==l]-center)/scale
 _,cells,count=np.unique(np.c_[labels,a['stratum']],axis=0,return_inverse=True,return_counts=True); w=1/count[cells]
 for j in range(x.shape[1]):
  m=np.isfinite(ax[:,j]); x[~np.isfinite(x[:,j]),j]=np.average(ax[m,j],weights=w[m])
 modes[tag]=x
z=hist(ta)
for name,x in modes.items():
 dr=th[2]+x@th[3:]; C=np.c_[h,dr]; coeff=np.linalg.lstsq(C,q,rcond=None)[0]; e=q-C@coeff
 print(name,'score RMSE',np.sqrt(np.mean(e**2)),'complete/incomplete',*[np.sqrt(np.mean(e[m]**2)) for m in (np.all(np.isfinite(raw),1),np.any(~np.isfinite(raw),1))],'coeffs',coeff)
 for tag,F in [('linear',np.c_[dr]),('drift2',np.c_[dr,dr**2]),('drift3',np.c_[dr,dr**2,dr**3]),('interaction',np.c_[dr,dr*h]),('history',np.c_[dr,z]),('free features',np.c_[np.ones(n),x])]:
  E=np.empty_like(R)
  for l in np.unique(lab):
   sel=lab!=l; E[~sel]=R[~sel]-F[~sel]@np.linalg.lstsq(F[sel],R[sel],rcond=None)[0]
  print(tag,np.round(np.max([[np.sqrt(np.mean(E[lab==l,:-1]**2)),np.sqrt(np.mean(E[lab==l,-1]**2))] for l in np.unique(lab)],0),5))
print('effective residual by coupon: lab, raw complete, inferred score, intrinsic drift, hold'); print(np.round(np.c_[lab,np.all(np.isfinite(raw),1),q,th[2]+modes['from_train_test']@th[3:],h],5))
