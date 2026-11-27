exec(open('/tmp/chamber_probe.py').read().split("for name,xc")[0])
from scipy.optimize import least_squares
Y=np.c_[c['instant_rate_obs'], c['strain_curves_obs']-c['strain_0'][:,None]]
S=c['past_channel_raw'].sum(-1); h=c['hold_channel_raw'].sum(-1); age=np.r_[np.cumsum(dt[::-1])[::-1][1:],0.]
H=lambda ts: -np.expm1(-dt/ts)*np.exp(-age/ts)
A=lambda ts: np.r_[1/ts,-np.expm1(-t/ts)]
labs=c['lab_id']; strata=c['stratum']; raw=c['coupon_features_raw']
# Profile exact temporal contrasts without assuming a feature or history law.
for linear in [False,True]:
 def res(p, ret=False):
  f=p[2]*h[:,None]*A(p[0]); B=np.stack([A(p[1]),np.r_[1,t]],axis=0) if linear else A(p[1])[None,:]
  coef=np.linalg.lstsq(B.T,(Y-f).T,rcond=None)[0].T
  return (p,coef) if ret else (f+coef@B-Y).ravel()
 p=least_squares(res,[tau[0],tau[1],theta[0]],bounds=([.2,1,.4],[1,12,1])).x
 p,C=res(p,True); print('fast-reset projection, linear',linear,'parameters',p,'RMSE',np.sqrt(np.mean(res(p)**2)))
 if not linear: continue
 print('linear drift amplitudes',np.round(C[:,1],4)); print('amplitudes minus intrinsic held slow branch',np.round(C[:,0]-theta[1]*h,4))
 for name,xx in [('train+diag',_features(a,c)[1]),('test+diag',_features(b,c)[1]),('diag+diag',_features(c,c)[1])]:
  dr=np.c_[np.ones(len(xx)),xx]@theta[2:]; E=C[:,0]-theta[1]*h
  print(name,'[lab,stratum,drift, slow correction]',np.round(np.c_[labs,strata,dr,E],3).tolist())
  V=np.c_[np.ones(len(xx)),dr]; coef=np.linalg.lstsq(V,E,rcond=None)[0]; e=E-V@coef
  print('linear drift fit',coef,'residual',np.round(e,3),'missing rows',np.where(np.any(np.isnan(raw),axis=1))[0])
  Fs={'h':h,'dr':dr,'dr2':dr**2,'dr3':dr**3,'h*dr':h*dr,'sin_dr':np.sin(dr),'tanh_dr':np.tanh(dr),'intr_mem':S@H(tau[1]),'past_mean':S.mean(axis=1),'past_sq':(S*S).mean(axis=1),'past_abs':abs(S).mean(axis=1),'history_square':(S@H(tau[1]))**2}
  for j in range(S.shape[1]):
   Fs[f'p{j}']=S[:,j]; Fs[f'p{j}^2']=S[:,j]**2; Fs[f'p{j}*dr']=S[:,j]*dr
   for k in range(j): Fs[f'p{j}*p{k}']=S[:,j]*S[:,k]; Fs[f'(p{j}-p{k})^2']=(S[:,j]-S[:,k])**2
  corr=sorted([(abs(np.corrcoef(v,e)[0,1]),k) for k,v in Fs.items()],reverse=True); print('residual correlations',corr[:12])
 print('obs singular values after subtracting fast',np.linalg.svd(Y-p[2]*h[:,None]*A(p[0]),compute_uv=False)[:5])
