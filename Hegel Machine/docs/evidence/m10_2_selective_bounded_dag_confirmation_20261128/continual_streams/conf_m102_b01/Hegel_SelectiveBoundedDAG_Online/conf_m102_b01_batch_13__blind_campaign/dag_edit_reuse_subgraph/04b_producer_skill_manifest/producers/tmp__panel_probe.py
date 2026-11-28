import numpy as np,pickle,sys,importlib.util
from scipy.optimize import least_squares
P='/holdout_task/'
s=pickle.load(open(P+'public_batch.pkl','rb'))[0]['study']
sp=importlib.util.spec_from_file_location('sink',P+'_frozen_nodes/node_01.py'); sink=importlib.util.module_from_spec(sp); sp.loader.exec_module(sink)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
aff=lambda a:np.c_[np.ones(len(a)),a]
u,v=[s['panel_b'][k] for k in ['u','v']]; a,b=[s['panel_a'][k] for k in ['u','v']]
co=[]
for j in range(4):
 d=aff(a[:,j]); y=b[:,j]; q=ls(d,y)
 for _ in range(8):
  idx=np.argsort(abs(d@q-y))[:42]; q=ls(d[idx],y[idx])
 er=abs(d@q-y); idx=er<=max(10*np.median(er),.002); co.append(ls(d[idx],y[idx])); print('A',j,co[-1],np.quantile(er,[.5,.8,.9,1]))
c=np.array(co).T;x=(s['train']['x']-c[0])/c[1]; t=s['train']
def score(w):
 n=52
 yp=.35*sink._predict(w,t['y'][:n],True,'asinh')+.65*sink._predict(w,t['y'][:n],True,'atan'); tp=sink._predict(w,t['traj'][:n],False,'inv3')
 return np.mean(abs(yp-t['y'][n:])),np.mean(abs(tp-t['traj'][n:]))
def fit(name, fun, seeds, inverse=False):
 best=None
 for q in seeds:
  def res(q):
   z=aff(fun(u,q)); return (z@ls(z,v)-v).ravel()
  if len(q):
   o=least_squares(res,q,max_nfev=150);q=o.x
  e=np.mean(res(q)**2)**.5
  if best is None or e<best[0]:best=e,q
 e,q=best; zz=aff(fun(u,q)); w=aff(fun(x,q))@ls(zz,v);print(name,'params',np.round(q,6),'panel RMS',e,'cv',score(w))
if __name__=='__main__':
 print('B corr',np.corrcoef(u.T,v.T)[:4,4:]); print('B linear coeff',ls(aff(u),v)); print('B rows',np.round(np.c_[u,v],3))
 fit('linear',lambda x,q:x,[[]]); fit('identity coordinates',lambda x,q:np.c_[x], [[]]);print('unmixed',score(x))
 for name,f in [('sin',np.sin),('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh),('sinh',np.sinh),('inv3',lambda x:np.sinh(np.arcsinh(x)/3))]:
  fit(name,lambda x,q:f(q[0]*x),[[h] for h in [.8,2,4,7]])
  fit('lin+'+name,lambda x,q:x+q[0]*f(q[1]*x),[[h,k] for h in [-1,-.4,.4,1] for k in [1,2.9,5]])
  fit('lin+cbrt+'+name,lambda x,q:x+q[0]*f(q[1]*x)+q[2]*np.cbrt(x),[[h,k,0] for h in [-.5,.5] for k in [1.5,3,5]])
 fit('rational',lambda x,q:x/(1+np.exp(q[0])*x*x),[[-2],[0],[1]])
 fit('linear+cubic+fifth',lambda x,q:x+q[0]*x**3+q[1]*x**5,[[0,0]])
 fit('linear+cuberoot',lambda x,q:x+q[0]*np.cbrt(x),[[0],[1],[-1]])
