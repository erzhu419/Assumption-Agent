import numpy as np,pickle
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def af(x): return np.c_[np.ones(len(x)),x]
def ls(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(u,d): return np.column_stack([np.ones(len(u))]+[np.prod(u[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(4),k)])
u,v=[s['panel_a'][k] for k in ('u','v')]; keep=np.linalg.norm(v-u,axis=1)<3; u,v=u[keep],v[keep]
for d in (1,2,3,4):
 a=poly(u,d); print('cal poly',d,np.sqrt(np.mean((a@ls(a,v)-v)**2)))
for off in (1,2,3):
 for mode in ('cubic','sine','tanh','rational'):
  def fit(f,ret=False):
   sn={'cubic':u**3,'sine':np.sin(f*u),'tanh':np.tanh(f*u),'rational':u/(1+f*u*u)}[mode]; yz=np.roll(u,-off,1); sz=np.roll(sn,-off,1)
   ds=[np.c_[np.ones(len(u)),u[:,j],yz[:,j],u[:,j]*yz[:,j],sn[:,j],sz[:,j]] for j in range(4)]; w=np.array([ls(a,v[:,j]) for j,a in enumerate(ds)]); err=np.sqrt(np.mean(np.column_stack([a@w[j]-v[:,j] for j,a in enumerate(ds)])**2)); return (err,w) if ret else err
  f=minimize_scalar(fit,bounds=(0.1,5),method='bounded').x; e,w=fit(f,True); print('cal',off,mode,round(f,5),round(e,7));
  if e<0.001: print(np.round(w,6))
for pk,vk in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[pk]['u'],s[pk][vk]; print(pk,vk)
 for name in ['lin','quadratic','cubic+cross','tanh','sin','cos','softsign','rational','mixed','res-sin','res-tanh']:
  def des(p):
   f=p*u; mp={'lin':af(u),'quadratic':poly(u,2),'cubic+cross':np.c_[poly(u,2),u**3],'tanh':af(np.tanh(f)),'sin':af(np.sin(f)),'cos':af(np.cos(f)),'softsign':af(f/(1+abs(f))),'rational':af(u/(1+p*u*u)),'mixed':np.c_[af(np.sin(f)),np.cos(f)],'res-sin':np.c_[af(u),np.sin(f)],'res-tanh':np.c_[af(u),np.tanh(f)]}; return mp[name]
  def obj(f):
   a=des(f); return np.sqrt(np.mean((a@ls(a,v)-v)**2))
  o=minimize_scalar(obj,bounds=(.1,5),method='bounded'); print(name,round(o.x,5),round(o.fun,7),end='; ')
 print()
z=s['train']['traj']; y=s['train']['y']; print('svd traj',np.linalg.svd(z,compute_uv=False)); print('corr endpoints and y',np.corrcoef(np.c_[z[:,[0,-1]],y].T));
