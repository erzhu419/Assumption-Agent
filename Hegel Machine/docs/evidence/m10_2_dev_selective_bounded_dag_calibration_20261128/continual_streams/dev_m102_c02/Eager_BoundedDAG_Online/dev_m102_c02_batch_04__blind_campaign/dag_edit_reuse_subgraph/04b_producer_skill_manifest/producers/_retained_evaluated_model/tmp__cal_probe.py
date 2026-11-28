import pickle,numpy as np,itertools
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
u,v=[s['panel_a'][z] for z in ('u','v')]; n,d=u.shape
A=lambda z:np.column_stack([np.ones(len(z)),z])
X=A(u); rng=np.random.default_rng(485)
best=(np.inf,None)
for _ in range(3000):
 ix=rng.choice(n,5,replace=False); b=np.linalg.lstsq(X[ix],v[ix],rcond=None)[0]; r=np.linalg.norm(v-X@b,axis=1); score=np.median(r)
 if score<best[0]: best=(score,b)
b=best[1]
for k in range(4):
 r=np.linalg.norm(v-X@b,axis=1); ix=np.argsort(r)[:36]; b=np.linalg.lstsq(X[ix],v[ix],rcond=None)[0]
r=np.linalg.norm(v-X@b,axis=1); ix=np.argsort(r)[:36]; print('A affine',b,'sorted residual',np.sort(r));
D={'1':np.ones(n)}
for j in range(d):
 t=u[:,j]; D[f'x{j}']=t
 for name,w in [('q',t*t),('cube',t**3),('tanh',np.tanh(1.25*t)),('sin',np.sin(1.25*t)),('rat',t/(1+1.25*t*t))]:D[f'{name}{j}']=w
for j,k in itertools.combinations(range(d),2):
 a,b=u[:,j],u[:,k]; D[f'prod{j}{k}']=a*b
 for name,w in [('sin+',np.sin(a+b)),('sin-',np.sin(a-b)),('tanh+',np.tanh(a+b)),('tanh-',np.tanh(a-b)),('r',a/(1+b*b)),('R',b/(1+a*a))]: D[f'{name}{j}{k}']=w
names=list(D); T=np.column_stack(list(D.values()))[ix];
for j in range(d):
 inds=[names.index(k) for k in ['1','x0','x1','x2','x3']]; y=v[ix,j]; print('A target',j,end=' ')
 for _ in range(5):
  Z=T[:,inds]; Q=np.linalg.qr(Z)[0]; W=T-Q@(Q.T@T); r=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]; c=np.abs(r@W)/np.maximum(np.linalg.norm(W,axis=0),1e-7); c[inds]=0; i=np.argmax(c); inds.append(i); Z=T[:,inds]; b=np.linalg.lstsq(Z,y,rcond=None)[0]; print(names[i],round(b[-1],5),round(np.sqrt(np.mean((y-Z@b)**2)),6),end='; ')
 print()
u,v=[s['panel_b'][z] for z in ('u','v')];
for name,fn in [('tanh',lambda a:np.tanh(a*u)),('sin',lambda a:np.sin(a*u)),('atan',lambda a:np.arctan(a*u)),('rat',lambda a:u/(1+a*u*u)),('sat',lambda a:u/np.sqrt(1+a*u*u)),('cubic',lambda a:u-a*u**3),('exp',lambda a:u*np.exp(-a*u*u))]:
 for add in [False,True]:
  def loss(a,summary=False):
   Z=A(fn(a)); Z=np.c_[Z,u] if add else Z; b=np.linalg.lstsq(Z,v,rcond=None)[0]; r=v-Z@b
   if summary:return np.sqrt(np.mean(r*r)),np.sqrt(np.mean((r/(1-np.sum(np.linalg.qr(Z)[0]**2,axis=1))[:,None])**2))
   return np.mean(r*r)
  res=minimize_scalar(loss,bounds=(0.02,5),method='bounded'); print('B',name,add,'scale',res.x,'rms/loo',loss(res.x,True),'at1.25',loss(1.25,True))
