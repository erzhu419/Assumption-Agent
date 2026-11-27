import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from m98_probe import *
m=mods['06']; bridge=m._axial_load_map(cal); xtr,xte=m._features(tr,te); _,xd=m._features(tr,diag); tau,theta=m._fit(m._specimen(tr,xtr,bridge),dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); past,hold,F=m._specimen(diag,xd,bridge); drift=F@theta[2:]; B,R=m._branches((past,hold,F),dt,t,tau); K=-np.expm1(-t[None,:]/tau[:,None]); reset=hold[:,None]*theta[:2]; carry=B[:,-1,:]*theta[:2]/K[:,-1]; y,r=targets
print('tau, theta:',tau,theta,'rate scale',np.std(r),'drift range',np.quantile(drift,[0,.5,1])); print('feature calibration shift',np.max(np.abs(xtr-m._features(tr,diag)[0]),axis=0));
for g,A in [('carry',carry),('reset',reset)]:
 for k,u in [('none',0*t),('lin',t),('sat',-np.expm1(-t)),('log',np.log1p(t)),('coupled',np.array([0.34273982,2.13827756])@K)]:
  dr=0 if k=='none' else (np.array([0.34273982,2.13827756])@(1/tau) if k=='coupled' else 1)
  print(g,k,np.round(np.sqrt(np.mean((y-A@K-drift[:,None]*u)**2)),6),np.round(np.sqrt(np.mean((r-A@(1/tau)-dr*drift)**2)),6))
print('rank and singular values',np.linalg.svd(y,compute_uv=False)); print('residual singular values',np.linalg.svd(y-carry@K,compute_uv=False)); print('carry rate error',np.round(r-carry@(1/tau)-drift,5));
for name,u in [('lin',t),('sat',-np.expm1(-t)),('log',np.log1p(t)),('sqrt',2*(np.sqrt(1+t)-1)),('none',0*t)]:
 Q=K if name=='none' else np.vstack([K,u]); W=np.linalg.lstsq(Q.T,y.T,rcond=None)[0].T; print('basis',name,'curve rmse',np.std(y-W@Q),'rate rmse',np.std(r-W@np.r_[1/tau,([] if name=='none' else [1])]),'amp corr',np.round(np.corrcoef(np.column_stack([W,carry,reset,drift]).T)[:W.shape[1],W.shape[1]:],3));
print('full carry residual medians vs times',t,np.median(y-carry@K,axis=0)); print('residual vs carry rows:',np.round(np.column_stack([hold,carry,drift,r,r-carry@(1/tau),(y-carry@K)[:,[0,4,-1]]]),4));
