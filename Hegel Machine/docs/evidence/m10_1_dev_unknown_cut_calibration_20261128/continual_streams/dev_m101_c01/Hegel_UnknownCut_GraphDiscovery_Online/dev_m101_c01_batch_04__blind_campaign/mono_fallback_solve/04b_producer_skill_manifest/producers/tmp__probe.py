import pickle,numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; np.set_printoptions(precision=5,suppress=True)
print('t,tel',t,s['telemetry_vec'])
a=s['panel_a']; A=np.c_[np.ones(len(a['u'])),a['u']]; keep=np.ones(len(A),bool)
for _ in range(4):
 b=np.linalg.lstsq(A[keep],a['v'][keep],rcond=None)[0]; r=np.linalg.norm(A@b-a['v'],axis=1); keep=r<4*np.median(r)
print('calibration',b, 'inliers',sum(keep), 'errors',np.sort(r)[-10:]);
for k in [1,2,3]:
 F=PolynomialFeatures(k).fit_transform(a['u']); print('panel_a poly',k,np.sqrt(np.mean((F[keep]@np.linalg.lstsq(F[keep],a['v'][keep],rcond=None)[0]-a['v'][keep])**2)))
xc=b[0]+x@b[1:];
for mode in ('exp','exp+constant','exp+linear','biexp'):
 def basis(p):
  g=-np.expm1(-t[:,None]*p)
  return {'exp':g,'exp+constant':np.c_[g,np.ones(len(t))],'exp+linear':np.c_[g,t],'biexp':g}[mode]
 def fun(p):
  B=basis(p); return (z.T-B@np.linalg.lstsq(B,z.T,rcond=None)[0]).ravel()
 fit=least_squares(fun,[1.] if mode!='biexp' else [1.,.3],bounds=(.001,10)); print(mode,fit.x,np.sqrt(np.mean(fit.fun**2)))
 if mode=='exp':
  B=basis(fit.x); amp=(np.linalg.lstsq(B,z.T,rcond=None)[0])[0]; print('temporal residual sv',np.linalg.svd(fun(fit.x).reshape(8,-1),compute_uv=False));
for W in [x,xc,np.c_[xc,amp]]:
 for k in [1,2,3]:
  F=PolynomialFeatures(k).fit_transform(W); P=F@np.linalg.lstsq(F[:56],np.c_[y,amp][:56],rcond=None)[0]; print('poly',W.shape[1],k,'cv',np.mean(abs(P[56:]-np.c_[y,amp][56:]),axis=0),'fit',np.std(P[:56]-np.c_[y,amp][:56],axis=0));
print('linear in xc',np.linalg.lstsq(np.c_[np.ones(len(x)),xc],np.c_[y,amp],rcond=None)[0]); print('ranges',np.min(xc,axis=0),np.max(xc,axis=0))
