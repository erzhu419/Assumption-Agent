import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; U=s['panel_a']['u']; V=s['panel_a']['v']
forms={'linear':(lambda x,k:x,[0.]),'cubic':(lambda x,k:x+k*x**3,[-.1,0,.2]),'quintic':(lambda x,k:x+k*x**5,[-.1,0,.2]),'tanh':(lambda x,k:np.tanh(k*x)/k,[.2,1.,2.]),'atan':(lambda x,k:np.arctan(k*x)/k,[.2,1.,2.]),'asinh':(lambda x,k:np.arcsinh(k*x)/k,[.2,1.,2.]),'sine':(lambda x,k:np.sin(k*x)/k,[.2,1.,2.]),'sinh':(lambda x,k:np.sinh(k*x)/k,[.2,1.,2.]),'rational':(lambda x,k:x/(1+k*x*x),[.01,.3,1.]),'sqrt':(lambda x,k:x/np.sqrt(1+k*x*x),[.01,.3,1.]),'log':(lambda x,k:np.sign(x)*np.log1p(k*np.abs(x))/k,[.2,1.,2.])}
for j,(u,v) in enumerate(zip(U.T,V.T)):
 print('CHANNEL',j, '\n',np.round(np.c_[u,v][np.argsort(u)],3).T)
 for name,(f,starts) in forms.items():
  opts=[]
  for k in starts:
   for mixed in ([False] if name=='linear' else [False,True]):
    def res(p): return p[0]+p[1]*f(u,p[2])+(p[3]*u if mixed else 0)-v
    lo=[-np.inf,-10,1e-5 if name not in ['linear','cubic','quintic'] else -2]+([-10] if mixed else []); hi=[np.inf,10,5]+([10] if mixed else [])
    sol=least_squares(res,[0,1,k]+([0] if mixed else []),bounds=(lo,hi),loss='cauchy',f_scale=.02,max_nfev=500)
    opts.append((sol.cost,mixed,sol.x,np.sort(abs(sol.fun))))
  cost,mixed,p,e=min(opts,key=lambda a:a[0]); print(name,mixed,np.round(p,5),'med,p85,p90,max',np.round(np.quantile(e,[.5,.85,.9,1]),5))
