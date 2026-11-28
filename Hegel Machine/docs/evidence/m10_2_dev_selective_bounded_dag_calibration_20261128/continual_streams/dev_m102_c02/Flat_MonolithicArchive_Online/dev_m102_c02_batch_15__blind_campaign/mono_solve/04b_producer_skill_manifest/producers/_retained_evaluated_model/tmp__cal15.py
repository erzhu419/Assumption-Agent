import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A=s['panel_a']; U,V=A['u'],A['v']; ok=np.linalg.norm(V,axis=1)<5; u,v=U[ok],V[ok]; print('range',u.min(),u.max()); print('archive calibration forms:')
def fit(h,y):return np.linalg.lstsq(h,y,rcond=1e-10)[0]
def prof(h):
 return np.column_stack([h[:,j]@fit(h[:,j],v[:,j])-v[:,j] for j in range(4)])
def stack(*args):return np.stack((np.ones_like(u),*args),axis=-1)
def cv(h):
 es=[]
 for j in range(4):
  a=h[:,j]; e=a@fit(a,v[:,j])-v[:,j]; lev=(a*np.linalg.pinv(a).T).sum(1); es.append(e/np.maximum(1-lev,.001))
 return np.std(es)
for k in (1,2,3):
 w=np.roll(u,k,axis=1)
 for nm,f,p,lo,hi in [('sin',lambda x,p:x*0+np.sin(p[0]*x),[2.4],[.5],[8]),('cos',lambda x,p:np.cos(p[0]*x),[2.6],[.5],[8]),('rat',lambda x,p:x/(1+p[0]*abs(x)**p[1]),[.6,1.2],[.001,.3],[5,5]),('atan',lambda x,p:np.arctan(p[0]*x),[2.4],[.3],[8])]:
  for mode in range(4):
   def h(p):
    if mode==0:return stack(u,u**3,w,f(w,p))
    if mode==1:return stack(u,u**3,w,u*w,f(w,p),u*f(w,p))
    if mode==2:return stack(u,w,u*w,f(u,p),f(w,p))
    return stack(u,w,f(u,p),f(w,p),u*f(w,p),w*f(u,p))
   o=least_squares(lambda p:prof(h(p)).ravel(),p,bounds=(lo,hi),max_nfev=50)
   print(k,nm,mode,round(np.std(o.fun),6),round(cv(h(o.x)),6),o.x)
for deg in range(1,6):
 from itertools import combinations_with_replacement
 zz=[np.roll(u,k,axis=1) for k in range(4)]; h=stack(*(np.prod([zz[i] for i in inds],axis=0) for d in range(1,deg+1) for inds in combinations_with_replacement(range(4),d)))
 print('poly',deg,np.std(prof(h)),cv(h))
