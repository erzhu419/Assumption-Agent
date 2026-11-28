import numpy as np,pickle,itertools as it
from scipy.special import erf
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True)
u,v=(s['panel_a'][k] for k in ('u','v'))
terms=[('lin',[u]),('sqdiag',[u,u**2]),('cubdiag',[u,u**2,u**3]),('odd',[u,u**3]),('quad',[u,np.array([u[:,i]*u[:,j] for i,j in it.combinations_with_replacement(range(4),2)]).T]),('quad+cubdiag',[u,np.array([u[:,i]*u[:,j] for i,j in it.combinations_with_replacement(range(4),2)]).T,u**3]),('odd5',[u,u**3,u**5])]
for name,arr in terms:
 d=np.column_stack([np.ones(len(u)),*arr]); rng=np.random.default_rng(72); best=None
 for i in range(500):
  idx=rng.choice(len(u),len(d.T),False); w=np.linalg.lstsq(d[idx],v[idx],rcond=None)[0]; r=np.linalg.norm(d@w-v,axis=1); score=np.mean(np.sort(r)[:36]**2)
  if best is None or score<best[0]:best=(score,w)
 w=best[1]
 for i in range(8):
  r=np.linalg.norm(d@w-v,axis=1); inds=np.argsort(r)[:39]; w=np.linalg.lstsq(d[inds],v[inds],rcond=None)[0]
 r=np.linalg.norm(d@w-v,axis=1); print('A',name,np.sort(r)[[0,10,20,30,38,39,40,41,42,43,44,45,46,47]],'bad',np.where(r>2)[0]);print(w if name in ('lin','quad+cubdiag') else '')

def lib(x,k):
 z=k*x
 return [np.sin(z),np.cos(z),np.tanh(z),erf(z),x/(1+z*z),x*abs(x)/(1+k*abs(x)),np.sin(z*z),np.cos(z*z),z/(1+abs(z)),np.arctan(z),np.exp(-z*z),x*np.sin(z),x*np.cos(z),z*np.exp(-z*z)]
for panel,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[panel]['u'],s[panel][key]; out=[]
 def basis(u,j,k,supp):
  z=lib(u,k)[j]; fs=[z];
  if supp&1:fs.append(u)
  if supp&2:fs.append(u**3)
  return fs
 for j in range(14):
  for supp in range(4):
   def profile(p,details=False):
    k=np.exp(p[0]); b=basis(u,j,k,supp); z=b[0]+sum(a*w for a,w in zip(b[1:],p[1:])); d=np.c_[np.ones(len(u)),z]; c=np.linalg.lstsq(d,v,rcond=None)[0];r=(d@c-v).ravel();return (np.sqrt(np.mean(r*r)),p,c) if details else r
   for k in [.6,1.5,2.7,4.3,6,8,11]:
    p=least_squares(profile,[np.log(k)]+[0]*int(supp).bit_count(),bounds=([-3]+[-40]*int(supp).bit_count(),[3]+[40]*int(supp).bit_count()),max_nfev=85).x; out.append((profile(p,True)[0],j,supp,p))
 print(panel,key,[(round(a,7),b,c,d) for a,b,c,d in sorted(out,key=lambda t:t[0])[:18]])
