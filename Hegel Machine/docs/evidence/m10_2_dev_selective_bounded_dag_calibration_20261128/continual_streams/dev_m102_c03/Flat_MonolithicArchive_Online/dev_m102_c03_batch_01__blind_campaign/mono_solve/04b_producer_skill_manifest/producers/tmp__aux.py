import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v2'];
for name,f in [('tanh',lambda u,k:u*np.tanh(k*u)),('sqrtr',lambda u,k:u*u/np.sqrt(1+k*u*u)),('sin',lambda u,k:u*np.sin(k*u)),('atan',lambda u,k:u*np.arctan(k*u)),('logcosh',lambda u,k:np.log(np.cosh(k*u))),('soft',lambda u,k:np.sqrt(1+k*u*u)-1),('log',lambda u,k:np.log1p(k*u*u)),('rat',lambda u,k:u*u/(1+k*u*u))]:
 def F(p):return np.c_[np.ones(len(u)),u+p[0]*f(u,p[1])]
 def r(p):
  g=F(p);return (g@np.linalg.lstsq(g,v,rcond=None)[0]-v).ravel()
 o=least_squares(r,[.55,.7],bounds=([.01,.03],[10,4]),gtol=1e-12);print(name,o.x,np.std(o.fun))
