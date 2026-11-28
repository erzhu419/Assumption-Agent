import numpy as np,pickle,itertools as it
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(x,y): return np.linalg.lstsq(x,y,rcond=1e-11)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
A=s['panel_a']; a=np.array([ls(aff(A['u'][:,i:i+1]),A['v'][:,i]) for i in range(4)]); err=np.linalg.norm(A['v']-A['u']*a[:,1]-a[:,0],axis=1);m=np.argsort(err)[:45]; a=np.array([ls(aff(A['u'][m,i:i+1]),A['v'][m,i]) for i in range(4)])
B=s['panel_b'];b=ls(aff(B['u']),B['v']); U,_,V=np.linalg.svd(b[1:]);b[1:]=.88*U@V
C=s['panel_c'];u=C['u'];v=C['v1'];w=C['v2'];
func={'lin':lambda x:x,'sin':np.sin,'tanh':np.tanh,'atan':np.arctan,'rat':lambda x:x/(1+x*x),'erf':__import__('scipy').special.erf,'softsign':lambda x:x/np.sqrt(1+x*x)}
def make(z,ns,p):return np.column_stack([np.ones(len(z))]+[func[n](z*k) for n,k in zip(ns,p)])
def err(p,ns,idx):
 f=make(u[idx],ns,p);return (f@ls(f,v[idx])-v[idx]).ravel()
models=[]
for k in [1,2,3]:
 for ns in it.combinations(func,k):
  if k==3 and not ('lin' in ns and 'sin' in ns): continue
  pp=least_squares(err,[1.6 if n!='lin' else 1 for n in ns],args=(ns,np.arange(24)),bounds=(.08,8),max_nfev=100).x
  er=np.std(err(pp,ns,np.arange(24))); models.append((er,ns,pp))
for er,ns,p in sorted(models,key=lambda x:x[0])[:12]:
 val=[]
 for idx in np.array_split(np.arange(24),4):
  ii=np.setdiff1d(np.arange(24),idx); q=least_squares(err,p,args=(ns,ii),bounds=(.08,8),max_nfev=60).x; val.extend((make(u[idx],ns,q)@ls(make(u[ii],ns,q),v[ii])-v[idx]).ravel())
 print('v1',ns,np.round(p,5),er,np.std(val))
ns=('lin','sin','tanh');p=min([z for z in models if z[1]==ns],key=lambda z:z[0])[2]; va=ls(make(u,ns,p),v)
f2=lambda z,k:np.c_[aff(z),np.sqrt(1+(k*z)**2)-1];k=least_squares(lambda p:(f2(u,p[0])@ls(f2(u,p[0]),w)-w).ravel(),[1],bounds=(.2,3)).x[0];wa=ls(f2(u,k),w)
print('v2',k,'coef',wa.T);print('v1 coef',va.T)
X=s['train']['x'];Y=np.c_[s['train']['y'],s['train']['traj']];Ainv=lambda x:(x-a[:,0])/a[:,1]; ops={'B':lambda x:aff(x)@b,'I':lambda x:(x-b[0])@np.linalg.inv(b[1:]),'C':lambda x:make(x,ns,p)@va,'D':lambda x:f2(x,k)@wa}
G={'x':X,'a':Ainv(X)}
for base,x in list(G.items()):
 for h,g in ops.items():
  G[base+h]=g(x)
  for j,fun in ops.items():
   if j+h not in ('BI','IB'):G[base+h+j]=fun(g(x))
   if h not in 'CD' or j not in 'CD':continue
   for l,ff in list(ops.items())[:2]:G[base+h+j+l]=ff(fun(g(x)))
folds=np.array_split(np.random.default_rng(202).permutation(len(X)),6)
def cv(F,reg=None):
 out=[]
 for hi in folds:
  tr=np.setdiff1d(np.arange(len(F)),hi)
  pred=F[hi]@ls(F[tr],Y[tr]) if reg is None else F[np.ix_(hi,tr)]@np.linalg.solve(F[np.ix_(tr,tr)]+reg*np.eye(len(tr)),Y[tr]-Y[tr].mean(0))+Y[tr].mean(0)
  out.append(np.mean(abs(pred-Y[hi]),axis=0)[[0,8]])
 return np.asarray(out)
res=[]
def record(name,er):res.append((er.mean(0).sum(),name,er.mean(0),er.max(0)))
for name,z in G.items():
 if not np.isfinite(z).all():continue
 for mode in ('linear','diagonal','quadratic','cubic','rbf','matern'):
  if mode in ('linear','diagonal','quadratic','cubic'):
   fs=aff(z)
   if mode=='diagonal': fs=np.c_[fs,z*z,np.tanh(z),np.sin(2*z)]
   if mode in ('quadratic','cubic'):
    fs=np.column_stack([fs]+[z[:,idx].prod(1) for d in range(2,4 if mode=='cubic' else 3) for idx in it.combinations_with_replacement(range(4),d)])
   record((name,mode),cv(fs))
  else:
   zd=(z-z.mean(0))/np.maximum(.1,z.std(0)); d=cdist(zd,zd)
   for l in (1.,2.,4.,8.):
    R=d/l; K=np.exp(-R**2/2) if mode=='rbf' else (1+R+R**2/3)*np.exp(-R)
    for reg in (.00001,.001,.015):record((name,mode,l,reg),cv(K,reg))
for z in sorted(res,key=lambda q:q[0])[:25]:print(z)
for i in [2,3]:
 print('target',i,[z for z in sorted(res,key=lambda q:q[i][0])[:3]],[z for z in sorted(res,key=lambda q:q[i][1])[:3]])
