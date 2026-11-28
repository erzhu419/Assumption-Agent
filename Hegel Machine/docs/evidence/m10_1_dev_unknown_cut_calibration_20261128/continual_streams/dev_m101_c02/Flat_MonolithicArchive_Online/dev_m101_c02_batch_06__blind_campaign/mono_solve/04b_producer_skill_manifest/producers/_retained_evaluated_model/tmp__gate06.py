from model06 import *
from scipy.linalg import qr
from sklearn.model_selection import KFold
from scipy.optimize import minimize_scalar
tr=s['train']; w=calibration(s)(tr['x']); y=tr['y']; T=tr['traj']; t=s['eval_times']; u,v=(s['panel_b'][k] for k in ('u','v')); folds=[(np.arange(56),np.arange(56,72)),*KFold(4,shuffle=True,random_state=4).split(w)]
fs={'rational':(lambda x,p:np.c_[aff(x),x**3/(1+p[0]*x*x)],[.8],([0],[10])), 'sin':(lambda x,p:np.c_[aff(x),np.sin(p[0]*x)],[2.8],([.01],[10])), 'rat+sin':(lambda x,p:np.c_[aff(x),x**3/(1+p[0]*x*x)+p[1]*np.sin(p[2]*x)],[.8,.01,7],([0,-.5,1],[10,.5,25])), 'power':(lambda x,p:np.c_[aff(x),x**3/(1+p[0]*abs(x)**p[1])],[.53,2.42],([0,1],[10,8])), 'archive':(lambda x,p:aff(x**3/(1+p[0]*abs(x)**p[1])+p[2]*x),[.53,2.42,-.09],([0,1,-1],[10,8,1]))}
models={}
for n,(f,p,b) in fs.items():
 def res(p):
  d=f(u,p);return (d@ls(d,v)-v).ravel()
 starts=[p] if n!='rat+sin' else [[.8,.01,k] for k in (3,5,7,10,14,19)]
 rs=min((least_squares(res,a,bounds=b,max_nfev=180) for a in starts),key=lambda o:o.cost); d=f(u,rs.x); c=ls(d,v); z=f(w,rs.x)@c; models[n]=z; print(n,np.round(rs.x,5),np.std(rs.fun),'coef',np.round(c,3) if n in ('rational','rat+sin') else '')
 def score(D):
  arr=[]
  for i,j in folds:
   pred=D[j]@ls(D[i],np.c_[y,T][i]); arr.append(np.mean(abs(pred-np.c_[y,T][j]),axis=0))
  print(np.round([arr[0][0],np.mean(arr[0][1:]),np.max(np.array(arr)[:,0]),np.max(np.mean(np.array(arr)[:,1:],axis=1))],5),end=' ')
 print('ols designs:');
 for D in (np.c_[aff(z),z*w,np.roll(z,1,axis=1)*w],np.c_[aff(z),z*w],np.c_[aff(z),(w[:,:,None]*z[:,None,:]).reshape(len(w),16)],np.c_[aff(z),z*w,np.roll(z,1,axis=1)*w,z[:,0]*z[:,2]+z[:,1]*z[:,3]]):score(D)
 print()
z=models['archive']; D=np.c_[aff(z),z*w,np.roll(z,1,axis=1)*w,z[:,0]*z[:,2]+z[:,1]*z[:,3]]; co=ls(D,np.c_[y,T]); print('coeff y, trajectory vs time:',np.round(co[:,[0,1,3,8]],4)); print('conditional residual sv',np.linalg.svd(T-D@co[:,1:],compute_uv=False)); print('OLS reconstructed time sv',np.linalg.svd(D@co[:,1:],compute_uv=False))
for name,F,init in [('logexp',lambda p:np.c_[np.log1p(p[0]*t),-np.expm1(-p[1]*t)],[1.7,.5]),('loglin',lambda p:np.c_[np.log1p(p[0]*t),t],[1.7]),('logsin',lambda p:np.c_[np.log1p(p[0]*t),np.sin(p[1]*t)],[1.7,1.7]),('logtanh',lambda p:np.c_[np.log1p(p[0]*t),np.tanh(p[1]*t)],[1.7,1.7])]:
 C=D@co[:,1:]; p=least_squares(lambda p:(ls(F(p),C.T).T@F(p).T-C).ravel(),init,bounds=(.01,12)).x
 print(name,p,'temporal projection rms',np.std(ls(F(p),C.T).T@F(p).T-C),'readout',np.round(ls(F(p),co[:,1:].T).T,3))
