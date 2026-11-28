exec(open('/tmp/chamber_probe.py').read().split('def fit(F):')[0])
from pathlib import Path
labs=d['lab_id']; raw=d['coupon_features_raw']; xx=np.empty_like(raw)
for j in np.unique(labs):
 m=labs==j; panel=d['ref_coupons'][m]; xx[m]=(raw[m]-np.nanmedian(np.nanmean(panel,axis=0)))/np.nanmedian(np.nanstd(panel,axis=0,ddof=0))
complete=np.isfinite(xx).all(1); drift=th[2]+xx@th[3:]; E=Y-(h[:,None]*th[:2])@T[:,:2].T

def coeff(m):
 A=(drift[m,None,None]*T[None,:,:]).reshape(-1,3); y=E[m].ravel()
 return least_squares(lambda q:A@q-y,np.linalg.lstsq(A,y,rcond=None)[0],loss='soft_l1',f_scale=.004).x
q=coeff(complete); print('chamber couplings',q)
for j in np.unique(labs):
 m=(labs==j)&complete; qj=coeff(complete&~m); e=drift[m,None]*(T@qj)-E[m]; print('lab',j,'complete',sum(m),'coupling',qj,'curve/rate rmse',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2)))
v=T@q; implied=E@v/(v@v); i,k=np.where(~np.isfinite(xx)); required=(implied[i]-th[2]-np.nan_to_num(xx[i])@th[3:])/th[3:][k]; print('lab, missing col, standardized value implied by label',np.c_[labs[i],k,required]); print('diag observed feature means',np.nanmean(xx,axis=0)); print('train absolute residual quantiles',np.quantile(abs(s['_design'](tau,inp,dt,t)[0]@th-(tr['strain_curves_obs']-tr['strain_0'][:,None])),[.5,.9,.95,.99]))
source=Path('/tmp/hegel_continual_archive/alpha_m100_c04_batch_03_solver.py').read_text(); start=source.index('    # The pre-run apparatus trace'); end=source.index('    rise =',start)
source=source[:start]+'''    # The chamber starts at zero branch memory. Verification coupons identify
    # one coupon drift that feeds both branch amplitudes and steady creep.
    drift = covariates @ theta[2:]
    amplitudes = hold[:, None] * theta[:2] + drift[:, None] * np.array(%r)
    linear = %.9g * drift
'''%(q[:2].round(8).tolist(),q[2])+source[end:]
Path('/holdout_task/batch_solver.py').write_text(source)
