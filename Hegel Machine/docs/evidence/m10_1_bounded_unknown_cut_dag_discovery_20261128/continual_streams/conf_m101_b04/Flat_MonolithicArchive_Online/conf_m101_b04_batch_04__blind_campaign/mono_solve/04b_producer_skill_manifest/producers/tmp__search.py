import numpy as np,pickle
from itertools import combinations_with_replacement as cr,product
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor,cho_solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=np.r_[s['train']['x'],s['test']['x']]; t=s['eval_times'];y=s['train']['y'];tr=s['train']['traj'];fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in cr(range(x.shape[1]),k)])
def learn(u,v,fn,start,lo,hi):
 def obj(p):
  d=fn(u,p);return (d@fit(d,v)-v).ravel()
 p=least_squares(obj,start,bounds=(lo,hi)).x;co=fit(fn(u,p),v);return lambda z:fn(z,p)@co
u,v=s['panel_a'].values();keep=np.arange(len(u))
for _ in range(5):keep=np.argsort(np.linalg.norm(poly(u,1)@fit(poly(u[keep],1),v[keep])-v,axis=1))[:45]
co=fit(poly(u[keep],3),v[keep]); A=lambda z:poly(z,3)@co
ix=list(i for k in range(4) for i in cr(range(4),k));print('A terms',[(i,np.round(c,4).tolist()) for i,c in zip(ix,co) if np.max(abs(c))>.001]);
u,v=s['panel_b'].values();bc=fit(poly(u,1),v); B=lambda z:poly(z,1)@bc
u,v1,v2=s['panel_c'].values(); C=learn(u,v1,lambda z,p:np.c_[poly(z,1),np.tanh(z*p)], [2]*4,.2,6); D=learn(u,v2,lambda z,p:np.c_[poly(z,1),np.cos(z*p)],[1.3]*4,.2,6)
az=A(x);xr=az; zdict={'x':x,'a':az,'c':C(x),'d':D(x),'ca':C(az),'da':D(az),'cb':C(B(x)),'db':D(B(x)),'cba':C(B(az)),'dba':D(B(az))};print('ranges', [(k,np.max(abs(v),axis=0).round(2)) for k,v in zdict.items()]);
def basis(i):
 def b(r):return -np.expm1(-t[:,None]*r)
 def obj(r):return (b(r)@fit(b(r),tr[i].T)-tr[i].T).ravel()
 return np.linalg.qr(b(least_squares(obj,[1.8,.41],bounds=([.7,.05],[4,.65])).x))[0]
fds=[(np.arange(52),np.arange(52,72))];rng=np.random.default_rng(931)
for _ in range(4):
 ii=rng.permutation(72);fds.append((ii[:52],ii[52:]))
def maes(p,k,reg,folds=fds):
 out=[]
 for i,j in folds:
  b=bases[tuple(i)];Y=np.c_[y,tr@b]; q=p[i];yt=Y[i]
  if k is None:pred=p[j]@fit(q,yt)
  else:
   ki=k[np.ix_(i,i)]+reg*np.eye(len(i)); fact=cho_factor(ki); kk=k[np.ix_(j,i)];a=cho_solve(fact,np.c_[q,yt]);m=q.shape[1];
   beta=fit(q.T@a[:,:m],q.T@a[:,m:]);pred=p[j]@beta+kk@(a[:,m:]-a[:,:m]@beta)
  out.append([np.mean(abs(pred[:,0]-y[j])),np.mean(abs(pred[:,1:]@b.T-tr[j]))]);
 return np.array(out)
bases={tuple(i):basis(i) for i,j in fds}
if __name__=='__main__':
 ds={k:cdist(z/z[:72].std(0),z/z[:72].std(0),'sqeuclidean') for k,z in zdict.items()};P={k+str(d):poly(z,d) for k,z in zdict.items() for d in (1,2,3)};P['const']=np.ones((len(x),1));
 for k,z in zdict.items():
  if k in ('x','a'):continue
  for base in ('x','a'):
   P[base+'2+'+k]=np.c_[P[base+'2'],z]; P[base+'2+'+k+'2']=np.c_[P[base+'2'],P[k+'2'][:,1:]]
 P['x2+a']=np.c_[P['x2'],az]; P['a2+x']=np.c_[P['a2'],x];
 for k in ('x','a'):
  z=zdict[k];P[k+'23sep']=np.c_[P[k+'2'],z**3]; P[k+'2r']=np.c_[P[k+'2'],1/(1+z*z),z/(1+z*z),np.tanh(1.5*z)]
 K={'-':None}
 for d in ('x','a','ca','da','cba','dba','x+a','a+ca','a+da'):
  dist=sum(ds[p] for p in d.split('+'))/len(d.split('+'))
  for g in (.008,.025,.065,.16,.4):K[d,g]=np.exp(-g*dist)
 rows=[]
 for (pn,p),(kn,k),reg in product(P.items(),K.items(),(.0003,.005,.03)):
  if k is None and reg!=.005:continue
  e=maes(p,k,reg);rows.append((e.mean(0).max(), e.max(),pn,kn,reg,e.round(5).tolist()))
 for f in [lambda r:r[0],lambda r:r[1],lambda r:max(r[-1][0])]:
  print('TOP');print(*sorted(rows,key=f)[:9],sep='\n')
