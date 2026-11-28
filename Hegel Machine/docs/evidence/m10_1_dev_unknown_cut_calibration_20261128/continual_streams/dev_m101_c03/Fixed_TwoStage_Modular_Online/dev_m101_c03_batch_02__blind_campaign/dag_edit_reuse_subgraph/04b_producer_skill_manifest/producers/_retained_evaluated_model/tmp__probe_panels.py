import pickle, numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
for panel in ('panel_a','panel_b','panel_c'):
 p=s[panel]; u=p['u']; one=np.ones((len(u),1))
 for key,v in p.items():
  if key=='u': continue
  print(panel,key)
  def assess(*terms):
   X=np.column_stack((one,*terms)); P=np.linalg.pinv(X); C=P@v; E=v-X@C
   loo=E/np.maximum(1e-8,1-(X*P.T).sum(1))[:,None]
   return np.mean(E**2)**.5,np.mean(loo**2)**.5,C,E
  if panel=='panel_a':
   _,_,C,E=assess(u); print(C, 'sorted row residuals',np.sort(np.linalg.norm(E,axis=1))); continue
  for name,terms in [('linear',[u]),('square',[u*u]),('linear+square',[u,u*u]),('linear+cube',[u,u**3]),('square+fourth',[u*u,u**4]),('square+fourth+sixth',[u*u,u**4,u**6]),('linear+sin cubic',[u,np.sin(u**3)])]:
   print(name,np.round(assess(*terms)[:2],6))
  for name,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('gauss',lambda x:np.exp(-x*x)),('rational',lambda x:1/(1+x*x)),('abs',np.abs)]:
   for lin in (False,True):
    fun=lambda k: assess(*(([u] if lin else [])+[f(k*u)]))[0]
    grid=np.arange(.2,12.1,.1); j=np.argmin([fun(k) for k in grid]); k=minimize_scalar(fun,bounds=(max(.02,grid[j]-.12),grid[j]+.12),method='bounded').x
    res=assess(*(([u] if lin else [])+[f(k*u)])); print(name,lin,round(k,5),np.round(res[:2],6))
    if res[0]<.009: print(res[2])
  if panel=='panel_c' and key=='v2':
   for name,z in [('B input',s['panel_b']['u']),('B output',s['panel_b']['v']),('v1',p['v1'])]:
    print(name,'linear/squared', np.round(assess(z)[:2],5),np.round(assess(z*z)[:2],5))
