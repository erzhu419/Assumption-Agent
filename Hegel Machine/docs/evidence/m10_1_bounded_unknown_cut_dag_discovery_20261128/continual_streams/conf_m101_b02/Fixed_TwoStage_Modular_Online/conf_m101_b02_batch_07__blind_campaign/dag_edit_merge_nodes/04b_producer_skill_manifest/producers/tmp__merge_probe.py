import pickle,itertools as it,numpy as np
from scipy.optimize import least_squares, minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
A=s['panel_a']; B=s['panel_b']; C=s['panel_c']; tr=s['train']; t=s['eval_times']; x=tr['x']; y=tr['y']; T=tr['traj']; n=len(y)
def fit(d,y): return np.linalg.lstsq(d,y,rcond=1e-10)[0]
def aff(z): return np.column_stack((np.ones(len(z)),z))
def quad(z): return np.column_stack([z[:,i]*z[:,j] for i,j in it.combinations_with_replacement(range(z.shape[1]),2)])
def diag(z): return np.column_stack((np.ones(len(z)),z,z*z))
def cross(z): return np.column_stack((np.ones(len(z)),z[:,0]*z[:,1],z[:,0]*z[:,3],z[:,1]*z[:,2],z[:,2]*z[:,3]))
a=np.vstack([fit(aff(A['u'][:,j:j+1]),A['v'][:,j]) for j in range(4)]).T
for _ in range(12):
 r=A['v']-a[0]-A['u']*a[1]; w=1/np.sqrt(1+np.sum(r*r,axis=1)/.02**2); a=np.vstack([fit(aff(A['u'][:,j:j+1])*w[:,None],A['v'][:,j]*w) for j in range(4)]).T
b=B['u']; v=B['v']; bm={}
for kind in ['sin','tan','rat','phase','mix']:
 def bd(z,p):
  f=p[:4]; sh=p[4:] if len(p)>4 else 0
  h={'sin':lambda:np.sin(z*f+sh),'tan':lambda:np.tanh(z*f+sh),'rat':lambda:(z+sh)/(1+f*(z+sh)**2),'phase':lambda:np.sin(z*f+sh),'mix':lambda:np.sin(z*f)}[kind]()
  return np.column_stack((aff(z),h,np.cos(z*f))) if kind=='mix' else np.column_stack((aff(z),h))
 p=np.r_[np.full(4,{'sin':2.8,'tan':1.4,'rat':.7,'phase':2.8,'mix':2.8}[kind]),np.zeros(4) if kind=='phase' else []]; lim=np.r_[np.full(4,.15),np.full(len(p)-4,-np.pi)]; hi=np.r_[np.full(4,6),np.full(len(p)-4,np.pi)]
 opt=least_squares(lambda p:(bd(b,p)@fit(bd(b,p),v)-v).ravel(),p,bounds=(lim,hi)); d=bd(b,opt.x); w=fit(d,v); q=np.linalg.qr(d)[0]; r=d@w-v; print('B',kind,np.round(opt.x,4),'rms',np.std(r),'conditional LOO',np.mean(abs(r/(1-(q*q).sum(1)[:,None])))); bm[kind]=(bd(x,opt.x)@w,bd((x-a[0])/a[1],opt.x)@w,bd(x*a[1]+a[0],opt.x)@w)
 print('B weights',np.round(w,3)) if kind in ('sin','phase') else None
cx=C['u']; c1=fit(cross(cx),C['v1']); c2=fit(diag(cx),C['v2']);
for k in ['component cubic','rational diagonal','tanh diagonal','rational input']:
 def cv(p,z):
  b=diag(z)@p[:9]; g=p[9:];
  if k=='component cubic': return b+(z**3)@g
  if k=='rational diagonal': return b/(1+g[0]*b*b)
  if k=='tanh diagonal': return np.tanh(g[0]*b)/g[0]
  return b/(1+z*z@g)
 ps=[]
 for j in range(4):
  m=4 if k in ('component cubic','rational input') else 1
  o=least_squares(lambda p:cv(p,cx)-C['v2'][:,j],np.r_[c2[:,j],np.full(m,.1)],max_nfev=300); ps.append(o.x)
 print('C2',k,np.sqrt(np.mean((np.column_stack([cv(p,cx) for p in ps])-C['v2'])**2)))
# Fit an exponential per trajectory and inspect the recoverable amplitude-rate coordinates.
def curve(k):return -np.expm1(-k*t)
def pars(z):
 def loss(k):
  f=curve(k); r=z-f*(z@f)/(f@f); return r@r
 k=minimize_scalar(loss,bounds=(.04,1.6),method='bounded').x; f=curve(k); return z@f/(f@f),k,loss(k)
p=np.array([pars(z) for z in T]); print('exp fit residual',np.sqrt(np.mean(p[:,2])/len(t)),'amp percentiles',np.quantile(p[:,0],[0,.1,.5,.9,1]),'rate quantiles',np.quantile(p[:,1],[0,.1,.5,.9,1])); print('rate signed amplitude weight corr',np.corrcoef(np.column_stack((x,y,p[:,:2])).T)[-1]);
# Compare panel modules and interaction closures on disjoint held-out blocks.
folds=[np.arange(52,n)]+[np.arange(i,min(i+18,n)) for i in range(0,n,18)]
def score(d,o):
 d=d/(np.sqrt(np.mean(d*d,axis=0))+1e-10); errs=[]
 for va in folds:
  ii=np.setdiff1d(np.arange(n),va); D=d[ii]; yy=o[ii]; u,g,h=np.linalg.svd(D,full_matrices=False); errs.append([np.mean(abs((d[va]@((h.T*(g/(g*g+alpha)))@(u.T@yy))-o[va])[:,sl])) for alpha in [] for sl in []] if False else [])
  for alpha in [.00003,.001,.01,.1,.5,2]:
   r=abs(d[va]@((h.T*(g/(g*g+alpha)))@(u.T@yy))-o[va]);errs[-1].append((r[:,0].mean(),r[:,1:].mean()))
 return np.array(errs)
F=[]
for coord,z in enumerate([x,(x-a[0])/a[1],x*a[1]+a[0]]):
 c=cross(z)@c1; h=diag(z)@c2
 for bk in bm:
  b=bm[bk][coord]; groups={'x':z,'b':b,'c':c,'h':h};
  for count in range(1,5):
   for block in it.combinations(groups,count):
    u=np.column_stack([groups[k] for k in block]); raw=aff(u); q=quad(u)
    for key,d in [('lin',raw),('q',np.column_stack((raw,q))),('diag',np.column_stack((raw,u*u))),('sdiag',np.column_stack((raw,u*u,np.sin(2*u)))),('xinter',np.column_stack((raw,np.einsum('ni,nj->nij',z,u).reshape(n,-1))),)]:
     r=score(d,np.column_stack((y,T)))
     for j in range(r.shape[1]): F.append((r[:,j,:].max(),r[0,j,:].max(),coord,bk,''.join(block),key,j,np.round(r[:,j,:],4).tolist()))
for name,G in [('joint',F),('public',sorted(F,key=lambda q:q[1]))]:
 print(name); seen=set(); m=0
 for row in (sorted(G) if name=='joint' else G):
  tag=row[2:6]
  if tag in seen:continue
  seen.add(tag);print(row);m+=1
  if m==16:break
