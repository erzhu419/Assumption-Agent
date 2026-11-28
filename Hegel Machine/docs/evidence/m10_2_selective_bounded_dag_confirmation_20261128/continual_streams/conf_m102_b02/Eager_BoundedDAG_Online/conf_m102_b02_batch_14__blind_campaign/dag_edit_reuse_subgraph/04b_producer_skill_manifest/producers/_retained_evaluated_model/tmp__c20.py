import sys,pickle,numpy as np
from scipy.optimize import minimize_scalar
sys.path.insert(0,'/holdout_task');import node_00 as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; D=m._design; fit=m._fit
for k in ('v1','v2'):
 v=s['panel_c'][k];print(k,'rows',np.c_[u,v][:5]);
 for inp,x in [('u',u),('v1',s['panel_c']['v1']),('v2',s['panel_c']['v2'])]:
  if inp==k:continue
  for name,f in [('sin',np.sin),('tanh',np.tanh),('cos',np.cos),('erf',m.erf),('rat',lambda x:x/(1+x*x)),('soft',lambda x:x/np.sqrt(1+x*x))]:
   for j in range(3):
    def phi(q):
     g=f(q*x); return D(g if j==0 else np.c_[x,g] if j==1 else np.c_[x,g,x*np.roll(x,-1,axis=1)])
    def err(q):
     A=phi(q);return np.mean((A@fit(A,v)-v)**2)
    grid=np.linspace(.2,5,25);i=np.argmin([err(q) for q in grid]);q=minimize_scalar(err,bounds=(grid[max(0,i-1)],grid[min(24,i+1)]),method='bounded').x
    if err(q)**.5<.014:print(inp,name,j,round(q,6),round(err(q)**.5,6),fit(phi(q),v).round(3).tolist() if j<=1 else '')
