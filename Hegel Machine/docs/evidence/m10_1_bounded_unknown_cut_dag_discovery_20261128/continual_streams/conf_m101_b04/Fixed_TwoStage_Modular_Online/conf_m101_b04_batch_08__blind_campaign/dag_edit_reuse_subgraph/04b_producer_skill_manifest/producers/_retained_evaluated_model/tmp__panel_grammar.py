from sink_probe import *
from scipy.optimize import minimize_scalar
p=s['panel_c']; x=p['u']
for v,fn in [('v1',np.tanh),('v2',np.cos)]:
 def obj(freq,lin):
  A=np.c_[np.ones(len(x)),x,fn(freq*x)] if lin else np.c_[np.ones(len(x)),fn(freq*x)]
  C=np.linalg.lstsq(A,p[v],rcond=None)[0];return np.std(A@C-p[v]), C
 for lin in (True,False):
  r=minimize_scalar(lambda f:obj(f,lin)[0],bounds=(0.5,4),method='bounded'); print(v,lin,r.x,r.fun,np.round(obj(r.x,lin)[1],3))
print('eval times',s['eval_times'])
