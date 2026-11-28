import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=(s['panel_a'][k] for k in ('u','v')); mask=np.max(abs(v),axis=1)<3;u,v=u[mask],v[mask];n=len(u)
ls=lambda f,y:np.linalg.lstsq(f,y,rcond=None)[0]
np.set_printoptions(precision=4,suppress=True)
def report(name,f):
 c=ls(f,v); print(name,'fit',np.sqrt(np.mean((v-f@c)**2)),'cv',np.sqrt(np.mean(np.concatenate([(v[i:i+5]-f[i:i+5]@ls(np.concatenate((f[:i],f[i+5:])),np.concatenate((v[:i],v[i+5:])))).ravel() for i in range(0,n,5)])**2))); return c
terms=[t for d in range(4) for t in combinations_with_replacement(range(4),d)];f=np.column_stack([np.prod(u[:,t],1) for t in terms]); c=report('dense cubic',f)
for t,w in zip(terms,c):print(t,w)
for j in range(4):
 a=u[:,j]; b=u[:,(j+2)%4]; y=v[:,j]; print('channel',j)
 for name,g in [('sin',lambda x,k:np.sin(k*x)),('tanh',lambda x,k:np.tanh(k*x)),('rat',lambda x,k:x/(1+k*x*x)),('cos',lambda x,k:np.cos(k*x)),('atan',lambda x,k:np.arctan(k*x)),('exp',lambda x,k:x*np.exp(-k*x*x)),('soft',lambda x,k:x/np.sqrt(1+k*x*x))]:
  for label, xx in [('other',b),('own',a),('sum',a+b),('diff',a-b)]:
   def cost(k,ret=False):
    f=np.c_[np.ones(n),a,b,g(xx,k)];c=ls(f,y);return c if ret else np.sqrt(np.mean((f@c-y)**2))
   grid=np.linspace(.02,8,250); k=grid[np.argmin([cost(k) for k in grid])]; fit=minimize_scalar(cost,bounds=(max(.0001,k-.04),k+.04),method='bounded'); print(name,label,round(fit.fun,6),round(fit.x,5),cost(fit.x,True))
