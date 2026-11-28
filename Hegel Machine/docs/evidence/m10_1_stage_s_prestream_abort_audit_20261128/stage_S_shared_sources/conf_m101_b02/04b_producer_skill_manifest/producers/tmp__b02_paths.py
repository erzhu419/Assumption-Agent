import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def aff(x): return np.c_[np.ones(len(x)),x]
def fit(x,y): return lstsq(x,y)[0]
def op(u,v,fn,q0,linear):
 def base(x,q): return np.c_[aff(x) if linear else np.ones(len(x)),fn(x*q)]
 def res(q):
  a=base(u,q);return (a@fit(a,v)-v).ravel()
 q=least_squares(res,np.array(q0),bounds=(.1,8),max_nfev=150).x
 c=fit(base(u,q),v);return lambda x:base(x,q)@c,np.mean(abs(res(q))),q
p=s['panel_c'];u=p['u'];v=p['v1'];g=lambda z:np.logaddexp(z,-z)-np.log(2)
for n,fn in [('atan',np.arctan),('erf',__import__('scipy').special.erf),('root',lambda z:z/np.sqrt(1+z*z)),('tanh',np.tanh),('asinh',np.arcsinh),('sin',np.sin)]:
 for lin in (False,True):
  _,e,q=op(u,v,fn,[2]*4,lin); print(n,lin,round(e,6),np.round(q,3))
C1,_,_=op(u,v,np.arctan,[2]*4,True)
C2,_,_=op(u,p['v2'],g,[1.1]*4,True)
a=s['panel_a'];A=[]
for j in range(4):
 u=a['u'][:,j];v=a['v'][:,j];i,k=np.triu_indices(len(u),1);m=np.median((v[i]-v[k])/(u[i]-u[k]));e=v-m*u;I=abs(e-np.median(e))<.01;A.append(fit(aff(u[I,None]),v[I]))
A=np.array(A);p=s['panel_b'];B=fit(aff(p['u']),p['v']);b=lambda x:aff(x)@B;bi=lambda x:(x-B[0])@np.linalg.inv(B[1:]);
x=s['train']['x']; y=s['train']['y'];T=s['train']['traj'];t=s['eval_times'];F=-np.expm1(-t[:,None]*[.4121,1.8165]);Y=np.c_[y,fit(F,T.T).T];Z={'x':x,'a':x*A[:,1]+A[:,0],'ai':(x-A[:,0])/A[:,1]}
for k,z in list(Z.items()): Z[k+'b']=b(z); Z[k+'bi']=bi(z)
for k,z in list(Z.items()):
 Z[k+'c']=C1(z);Z[k+'d']=C2(z);Z[k+'cb']=b(C1(z));Z[k+'db']=b(C2(z)); Z[k+'+d']=np.c_[z,C2(z)];Z[k+'+c']=np.c_[z,C1(z)]; Z[k+'cd']=np.c_[C1(z),C2(z)]
np.savez('/tmp/b02_paths.npz',**Z,Y=Y,F=F,T=T)
folds=[np.arange(56,72),*np.array_split(np.random.default_rng(34).permutation(72),4)]
for col in (0,1,2):
 out=[]
 for name,z in Z.items():
  if z.shape[1]!=4: continue
  features=[z];names=[f'x{i}' for i in range(4)]
  for j in (2,3):
   for ix in itertools.combinations_with_replacement(range(4),j): features.append(np.prod(z[:,ix],axis=1)[:,None]);names.append(str(ix))
  for i,j in itertools.combinations_with_replacement(range(4),2):
   for c in ([0] if i==j else [-1.,-.5,.5,1.]):
    w=z[:,i]+c*z[:,j];
    for h in (.6,1.,1.6,2.4):
     for k,f in [('t',np.tanh),('s',np.sin),('c',np.cos),('a',np.arctan),('r',lambda q:q/(1+q*q)),('g',g)]:
      features.append(f(h*w)[:,None]);names.append(f'{k}{h}({i},{c:g},{j})')
  for i,j in itertools.product(range(4),repeat=2):
   features.append((z[:,i]/(1+z[:,j]**2))[:,None]); names.append(f'quot{i},{j}')
  P=np.column_stack(features);ee=[]
  for J in folds:
   I=np.setdiff1d(np.arange(72),J);tr=P[I];mu=tr.mean(0);tr=tr-mu;scale=np.linalg.norm(tr,axis=0).clip(1e-10);tr/=scale;val=(P[J]-mu)/scale;yy=Y[I,col];ym=yy.mean(); yy=yy-ym;M=[];err=[];r=yy.copy();norm=1
   for step in range(14):
    sc=abs(tr.T@r)/np.maximum(norm,1e-4);sc[M]=-1;w=np.argmax(sc);M.append(w);D=tr[:,M];c=fit(D,yy);r=yy-D@c;norm=np.sqrt(np.maximum(0,1-np.sum((np.linalg.qr(D)[0].T@tr)**2,axis=0)));err.append(np.mean(abs(val[:,M]@c+ym-Y[J,col])))
   ee.append(err)
   if J[0]==56: terms=[names[k] for k in M]
  e=np.array(ee);j=e.mean(0).argmin();out.append((e.mean(0)[j],e.max(0)[j],name,j+1,terms[:j+1]))
 print('sparse',col,sorted(out)[:5])
