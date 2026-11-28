import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
u,v=(s['panel_a'][k] for k in ('u','v')); keep=np.linalg.norm(v-u,axis=1)<3;u,v=u[keep],v[keep];r=np.roll(u,2,axis=1)
def A(p,ret=False):
 ds=[np.c_[np.ones(len(u)),u[:,j],np.sin(p[0]*r[:,j]),np.sin(p[1]*u[:,j])] for j in range(4)]; ws=[np.linalg.lstsq(d,v[:,j],rcond=None)[0] for j,d in enumerate(ds)];err=np.array([d@w-v[:,j] for j,(d,w) in enumerate(zip(ds,ws))]);return (np.std(err),ws) if ret else err.ravel()
for g in ([2,1],[2,2.8],[3,3],[4,1]):
 p=least_squares(A,g,bounds=([.3,.2],[7,7])).x;print('A',p,A(p,True))
u,v=(s['panel_c'][k] for k in ('u','v1')); n=len(u)
forms={'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'tanh':lambda x,k:np.tanh(k*x),'erf':lambda x,k:erf(k*x),'softsign':lambda x,k:x/(1+k*abs(x)),'rat':lambda x,k:x/(1+k*x*x),'abs':lambda x,k:abs(x),'oddquad':lambda x,k:x*abs(x)/(1+k*abs(x)),'gauss':lambda x,k:np.exp(-k*x*x),'xcos':lambda x,k:x*np.cos(k*x),'hinge':lambda x,k:np.maximum(x-k,0),'clip':lambda x,k:np.clip(x,-k,k),'exp':lambda x,k:np.expm1(k*x),'cubic':lambda x,k:x**3}
rows=[]
for name,f in forms.items():
 for mode in ('free','shared','post'):
  def fun(p,ret=False):
   k=np.exp(p[0]); w=np.c_[np.ones(n),u]
   if mode=='post':
    a=p[1:21].reshape(5,4); h=w@a; res=h+p[21]*f(h,k)-v
   else:
    z=f(u,k); d=np.c_[w,z] if mode=='free' else np.c_[np.ones(n),z+p[1]*u];res=d@np.linalg.lstsq(d,v,rcond=None)[0]-v
   return res.ravel()
  for k in (.3,1,2.5,4,7):
   p=[np.log(k)]+([0] if mode=='shared' else list(np.linalg.lstsq(np.c_[np.ones(n),u],v,rcond=None)[0].ravel())+[.2] if mode=='post' else []);a=least_squares(fun,p,bounds=([-4]+[-30]*(len(p)-1),[3]+[30]*(len(p)-1)),max_nfev=90);rows.append((np.std(a.fun),name,mode,a.x[0],a.x[1] if len(a.x)>1 else None))
print('C sorted');print(*sorted(rows)[:30],sep='\n')
