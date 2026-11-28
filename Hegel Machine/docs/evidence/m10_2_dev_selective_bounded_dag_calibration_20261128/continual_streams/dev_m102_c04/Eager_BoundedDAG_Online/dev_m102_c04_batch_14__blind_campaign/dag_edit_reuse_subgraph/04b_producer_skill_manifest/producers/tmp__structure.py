import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda u:np.c_[np.ones(len(u)),u]
u,v=(s['panel_a'][k] for k in ('u','v'))
print('A pairs',np.c_[u,v]); print('A uv diff norm',np.sort(np.linalg.norm(u-v,axis=1)))
# Use random minimal affine fits to identify the clean reference regime.
rng=np.random.default_rng(0); a=aff(u); best=(np.inf,None)
for z in range(2000):
 i=rng.choice(len(u),5,False); w=ls(a[i],v[i]); e=np.linalg.norm(a@w-v,axis=1); m=np.median(e)
 if m<best[0]:best=(m,w)
e=np.linalg.norm(a@best[1]-v,axis=1); mask=e<3*np.median(e); w=ls(a[mask],v[mask]); print('A robust affine',w,'resid',np.sort(np.linalg.norm(a@w-v,axis=1)))
for f in [lambda u,c:u/(1+c*abs(u)),lambda u,c:u/(1+c*u*u),lambda u,c:np.tanh(c*u),lambda u,c:u**3,lambda u,c:u*u/(1+c*u*u),lambda u,c:u*abs(u)]:
 def design(u,c):return np.c_[aff(u),f(u,c)]
 def err(c):
  d=design(u[mask],c[0]); return (d@ls(d,v[mask])-v[mask]).ravel()
 opt=least_squares(err,[.5],bounds=(.005,6)); print('A nonlinear',opt.x,np.linalg.norm(opt.fun)/np.sqrt(mask.sum()))
u=s['panel_c']['u']; a=aff(u)
for key in ('v1','v2'):
 v=s['panel_c'][key]; print('C',key)
 for name,f in [('soft',lambda x,c:x/(1+c*abs(x))),('tanh',lambda x,c:np.tanh(c*x)),('rat',lambda x,c:x/(1+c*x*x)),('asinh',lambda x,c:np.arcsinh(c*x)),('sin',lambda x,c:np.sin(c*x)),('exp2',lambda x,c:np.exp(-c*x*x)),('sat',lambda x,c:x*x/(1+c*x*x)),('tanh2',lambda x,c:np.tanh(c*x*x))]:
  for bias in (False,True):
   d=lambda c:np.c_[a if bias else a[:,:1],f(u,c)]
   def er(c):
    b=d(c);return (b@ls(b,v)-v).ravel()
   o=least_squares(er,[.5],bounds=(.005,6),ftol=1e-11,gtol=1e-11); print(name,bias,o.x,np.linalg.norm(o.fun)/np.sqrt(len(u)))
print('summary',s['train']['summary'],'x mean std y mean std traj end',s['train']['x'].mean(0),s['train']['x'].std(0),s['train']['y'].mean(),s['train']['y'].std(),s['train']['traj'][:,-1].mean(),s['train']['traj'][:,-1].std())
