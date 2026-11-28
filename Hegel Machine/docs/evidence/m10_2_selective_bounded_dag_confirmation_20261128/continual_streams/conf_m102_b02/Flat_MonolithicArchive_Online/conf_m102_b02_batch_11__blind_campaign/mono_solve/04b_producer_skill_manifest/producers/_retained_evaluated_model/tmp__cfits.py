import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; I=lambda a:np.c_[np.ones(len(a)),a]; ls=lambda a,v:np.linalg.lstsq(a,v,rcond=1e-10)[0]
func={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'rat':lambda a:a/(1+a*a),'soft':lambda a:a/np.sqrt(1+a*a),'atan':np.arctan,'cos':np.cos,'cosh':np.cosh,'exp':lambda a:np.exp(np.clip(a,-30,30))}
for k in ['v1','v2']:
 v=p[k]; print(k)
 for name,f in func.items():
  for linear in [False,True]:
   def D(a,q):return I(np.c_[a,f(a*q)]) if linear else I(f(a*q))
   def E(q):
    a=D(u,q);return (a@ls(a,v)-v).ravel()
   r=min([least_squares(E,[q],bounds=(.02,7),max_nfev=80) for q in [.4,1,2,3,5]],key=lambda r:r.cost); print(name,linear,r.x,np.sqrt(np.mean(E(r.x)**2)))
 if k=='v2':
  a=I(np.c_[u,u*u,*[u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)]]); print('quadratic terms',ls(a,v).round(4))
