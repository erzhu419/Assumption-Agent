exec(open('/tmp/check_rheology01.py').read().split("print('active'")[0])
wt=np.r_[g['theta'],g['vp_theta']]; tau=g['taus']; l=d['test']['lab_id']; miss=np.isnan(d['test']['coupon_features_raw']); b=m._basis(np.log(tau),*args,dt,t,'all')[0]@wt
err=b[n:,:-1]+off[n:,None]-g['te_clean']; er=b[n:,-1]-g['te_true_rate0']
for lab in range(4):
 for missing in (False,True):
  k=(l==lab)&(miss.any(1)==missing)
  if k.any(): print('oracle coefficient audit',lab,missing,k.sum(),'strain/rate max',abs(err[k]).max(),abs(er[k]).max(),'rate mean',er[k].mean())
print('rate vs curve drift residual',np.max(abs(err-er[:,None]*np.log1p(t))))
for j in np.flatnonzero(miss.any(1)):
 print('missing',j,'lab',l[j],'stratum',d['test']['stratum'][j],'columns',np.where(miss[j])[0], 'drift error',er[j])
for mode in ('gated','all'):
 for lab in range(4):
  keep=d['train']['lab_id']!=lab
  tau,w=m._fit((*[v[:n][keep] for v in args],dt,t),yy[keep].ravel(),mode)
  pred=m._basis(np.log(tau),*[v[:n][~keep] for v in args],dt,t,mode)[0]@w
  e=pred-yy[~keep]
  print('LOLO',mode,lab, 'taus',tau.round(4),'curve/rate MAE medians',np.median(abs(e[:,:-1])),np.median(abs(e[:,-1])))
