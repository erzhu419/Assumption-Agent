import numpy as np,pickle
from scipy.optimize import minimize_scalar,least_squares
from cal20 import D,fit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.linalg.norm(u-v,axis=1)<3; u=u[keep];v=v[keep]
forms={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'soft':lambda x:x/np.sqrt(1+x*x),'rational':lambda x:x/(1+x*x),'sinh':np.sinh,'asinh':np.arcsinh,'erf':__import__('scipy.special',fromlist=['erf']).erf}
for name,fn in forms.items():
 def des(x,q):return D(np.c_[x,fn(q*x)])
 def e(q):
  F=des(u,q); return np.mean((F@fit(F,v)-v)**2)
 q=min([minimize_scalar(e,bounds=(a,a+.5),method='bounded') for a in np.arange(.1,5,.5)],key=lambda r:r.fun)
 print(name,q.x,q.fun**.5,fit(des(u,q.x),v).round(4).tolist())
print('panel C shapes',[(k,np.shape(v)) for k,v in s['panel_c'].items()]);print('time',s['eval_times']); print('tele',s['telemetry_vec'])
