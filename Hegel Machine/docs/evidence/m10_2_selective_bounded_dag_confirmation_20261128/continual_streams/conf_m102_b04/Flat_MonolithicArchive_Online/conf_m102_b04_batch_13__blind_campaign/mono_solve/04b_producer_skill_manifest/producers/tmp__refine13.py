import numpy as np,pickle,importlib.util
from scipy.optimize import least_squares,minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']
sp=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_b04_batch_09_solver.py'); old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
af=lambda x:np.c_[np.ones(len(x)),x]
rms=lambda e:np.mean(e**2)**.5
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v-u,axis=1)<3;u,v=u[keep],v[keep]; a=u; b=np.roll(u,-2,axis=1)
extra={'none':lambda a,b,k:[], 'generic3':lambda a,b,k:[a*a*b,a*b*b,a**3,b**3], 'self_cubic':lambda a,b,k:[a**3], 'cross_cubic':lambda a,b,k:[b**3], 'self_square':lambda a,b,k:[a*a], 'cross_square':lambda a,b,k:[b*b], 'self_tanh':lambda a,b,k:[np.tanh(k*a)], 'self_rat':lambda a,b,k:[a/(1+k*a*a)], 'self_soft':lambda a,b,k:[a/(1+k*abs(a))], 'sum':lambda a,b,k:[np.sin(k*(a+b))], 'difference':lambda a,b,k:[np.sin(k*(a-b))], 'mixed':lambda a,b,k:[a*np.sin(k*b),b*np.sin(k*a),np.sin(k*a)*np.sin(k*b)],'opposite_frequency':lambda a,b,k:[np.sin(k*b)], 'self_frequency':lambda a,b,k:[np.sin(k*a)]}
for name,ex in extra.items():
 def fit(p,get=False):
  k,j=p; d=np.stack([np.ones_like(a),a,b,a*b,np.sin(k*a),np.sin(k*b),*ex(a,b,j)],-1); w=np.array([ls(d[:,i],v[:,i]) for i in range(4)]); e=np.einsum('nij,ij->ni',d,w)-v
  if get:
   h=np.stack([(np.linalg.qr(d[:,i],mode='reduced')[0]**2).sum(1) for i in range(4)],1);return rms(e),rms(e/(1-h)),w
  return e.ravel()
 o=min([least_squares(fit,[2.4,j],bounds=([1.5,.03],[3.5,8.]),max_nfev=150) for j in (0.5,1.6,3.5)],key=lambda x:x.cost); e,l,w=fit(o.x,True);print(name,o.x,e,l); 
 if l<0.002:print(np.round(w,6))
pu,pv=[s['panel_b'][k] for k in ('u','v')]; f=lambda k:af(np.cos(k*pu)); k=minimize_scalar(lambda k:rms(f(k)@ls(f(k),pv)-pv),bounds=(2.4,2.8),method='bounded').x
z=old._invert(tr['x'],old._calibration(s['panel_a'])); B=af(np.cos(k*z))@ls(f(k),pv); ij=list(combinations_with_replacement(range(4),2)); Q=np.c_[af(B),*[B[:,i]*B[:,j] for i,j in ij]];print('quadratic coefficients',list(zip(['bias',0,1,2,3]+ij,np.round(ls(Q[:52],tr['y'][:52]),6))))
T=tr['traj']; A=af(B); beta=ls(A[:52],T[:52]); H=lambda p:-np.expm1(-t[:,None]*p); rs=least_squares(lambda p:(H(p)@ls(H(p),beta.T)-beta.T).ravel(),[.4,1.8],bounds=(.01,10.)).x; print('rates',rs,'temporal residual raw',rms(H(rs)@ls(H(rs),T[:52].T)-T[:52].T));print('spatial coeffs',ls(H(rs),beta.T).T)
