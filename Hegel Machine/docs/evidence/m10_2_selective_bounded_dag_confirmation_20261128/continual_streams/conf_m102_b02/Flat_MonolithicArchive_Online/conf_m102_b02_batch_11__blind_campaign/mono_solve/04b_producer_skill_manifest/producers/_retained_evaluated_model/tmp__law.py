import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; I=lambda a:np.c_[np.ones(len(a)),a]; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-11)[0]
for pk,vk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 p=s[pk];u=p['u'];v=p[vk]; print(pk,vk,'linear',np.std(I(u)@ls(I(u),v)-v));
 for name,fun in [('tanh',np.tanh),('sin',np.sin),('erf',erf),('asinh',np.arcsinh),('rat',lambda a:a/(1+a*a)),('cos',np.cos),('soft',lambda a:a/np.sqrt(1+a*a))]:
  def fit(q):
   X=I(u+q[0]*fun(q[1]*u)); C=ls(X,v); return C,(X@C-v).ravel()
  best=min((least_squares(lambda q:fit(q)[1],[a,k],bounds=([-8,.1],[8,7]),max_nfev=100) for a in [-.4,.4] for k in [.6,1.3,2.15,3,5]),key=lambda r:r.cost);print(name,best.x,np.std(best.fun))
 def d(a): return I(np.c_[a,np.tanh(1.5*a)])
 c=ls(d(u),v); print('tanh coef',c.round(3));print('ratio', (c[5:]/c[1:5]).round(2))
