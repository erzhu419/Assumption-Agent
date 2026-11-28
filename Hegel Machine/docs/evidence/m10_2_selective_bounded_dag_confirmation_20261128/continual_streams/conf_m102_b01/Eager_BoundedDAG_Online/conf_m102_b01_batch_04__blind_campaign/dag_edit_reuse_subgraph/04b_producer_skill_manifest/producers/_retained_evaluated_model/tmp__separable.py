exec(open('/tmp/fit.py').read().split("print('B linear'")[0])
from scipy.optimize import least_squares, minimize_scalar
np.set_printoptions(precision=5,suppress=True,linewidth=150)
print('times',s['eval_times'],'telemetry',s['telemetry_vec']); print('shapes',[(name,{k:v.shape for k,v in p.items()}) for name,p in zip('ABC',(a,b,c))]);
def lmat(z,v):
 h=np.sum(z*np.linalg.pinv(z).T,axis=1); r=v-z@ls(z,v); return np.mean(abs(r)),np.mean(abs(r/np.maximum(1-h[:,None],1e-6)))
def pw(u,ix):return np.c_[np.ones(len(u)),*[u**j for j in ix]]
for name,u,v in [('B',b['u'],b['v']),('C2',c['u'],c['v2'])]:
 for ix in [[3],[1,3],[3,5],[1,3,5],[1,2,3,4],[1,3,5,7],[1,2,3,4,5,6]]:
  z=pw(u,ix); print(name,ix,lmat(z,v));
  if ix in ([1,3,5],[1,2,3,4]): print(ls(z,v))
 for nm,fn in [('cubic-tanh',lambda p,u:np.tanh(p[0]*u**3)),('cubic-atan',lambda p,u:np.arctan(p[0]*u**3)),('cubic-sin',lambda p,u:np.sin(p[0]*u**3)),('cubic-sinh',lambda p,u:np.sinh(p[0]*u**3)),('cubic+rational',lambda p,u:(u**3+p[1]*u)/(1+p[0]*u*u)),('u+tanh',lambda p,u:u+p[1]*np.tanh(p[0]*u)),('sinh',lambda p,u:np.sinh(p[0]*u)),('exp',lambda p,u:np.expm1(p[0]*u)),('quad+quartic',lambda p,u:u+p[0]*u*u+p[1]*u**4),('quad-rational',lambda p,u:u+p[0]*u*u/(1+p[1]*u*u)),('quad-cos',lambda p,u:u+p[0]*(1-np.cos(p[1]*u)))]:
  def resid(p):
   z=aff(fn(p,u)); return (v-z@ls(z,v)).ravel()
  sol=min((least_squares(resid,pp,bounds=([.0001,-.26] if nm=='quad-rational' else [.0001,-8],[8,8]),max_nfev=80) for pp in ([.22,1.25],[1.25,.22],[.44,-.04])),key=lambda so:so.cost)
  z=aff(fn(sol.x,u)); print(name,nm,sol.x,lmat(z,v));
  if lmat(z,v)[1]<.02: np.savez('/tmp/cal_'+name+'_'+nm,bx=aff(fn(sol.x,x))@ls(z,v),p=sol.x)
Y=np.c_[tr['y'],tr['traj']]; U,S,V=np.linalg.svd(tr['traj'],full_matrices=False); print('corr y,modes',np.corrcoef(np.c_[Y[:,0],U[:,:2]].T));
for j,t in enumerate(s['eval_times']): print('time',t,np.corrcoef(Y[:,0],Y[:,j+1])[0,1],np.mean(Y[:,j+1]),np.std(Y[:,j+1]));
