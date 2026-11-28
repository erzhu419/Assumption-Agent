import pickle,itertools,numpy as np
from scipy.linalg import lstsq
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; t=s['eval_times']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
def affine(u,v):return lstsq(np.c_[np.ones(len(u)),u],v)[0]
def panel():
 p=s['panel_a'];u=p['u'];v=p['v']; A=[]
 for j in range(4):
  uu=u[:,j];vv=v[:,j];i,k=np.triu_indices(len(u),1); slope=np.median(((vv[i]-vv[k])/(uu[i]-uu[k])));r=vv-slope*uu;ok=abs(r-np.median(r))<0.01;A.append(affine(uu[ok,None],vv[ok]))
 p=s['panel_b'];B=affine(p['u'],p['v']);return np.array(A),B
A,B=panel();print('A,B',A,B)
def coords(x):
 z=x*A[:,1]+A[:,0];zi=(x-A[:,0])/A[:,1];return {'x':x,'a':z,'ai':zi,'b':x@B[1:]+B[0],'ab':z@B[1:]+B[0],'aib':zi@B[1:]+B[0]}
def poly(x,d):
 ix=[j for k in range(d+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),k)];return np.column_stack([np.prod(x[:,j],axis=1) for j in ix]),ix
p=s['panel_c'];P,ix=poly(p['u'],2)
for k in ['v1','v2']:
 c=lstsq(P,p[k])[0];print(k,'quadratic coeff',list(zip(ix,c)),'rms',np.sqrt(np.mean((P@c-p[k])**2,axis=0)))
# Time models: use projection to identify which common time functions survive truncation.
for n,f,guess in [('two exp',lambda q:np.c_[np.expm1(-q[0]*t),np.expm1(-q[1]*t)],[.6,2]),('exp+linear',lambda q:np.c_[np.expm1(-q[0]*t),t],[1]),('log+exp',lambda q:np.c_[np.log1p(q[0]*t),np.expm1(-q[1]*t)],[1,2]),('log+time-exp',lambda q:np.c_[np.log1p(q[0]*t),t*np.exp(-q[1]*t)],[1,2]),('exp+time-exp',lambda q:np.c_[np.expm1(-q[0]*t),t*np.exp(-q[1]*t)],[1,2]),('log+tanh',lambda q:np.c_[np.log1p(q[0]*t),np.tanh(q[1]*t)],[1,2]),('log+atan',lambda q:np.c_[np.log1p(q[0]*t),np.arctan(q[1]*t)],[1,2]),('log+sin',lambda q:np.c_[np.log1p(q[0]*t),np.sin(q[1]*t)],[1,1])]:
 def fun(q):
  F=f(q);return (F@lstsq(F,T.T)[0]-T.T).ravel()
 r=least_squares(fun,guess,bounds=(1e-4,15));print(n,r.x,np.sqrt(np.mean(r.fun**2)))
# Power and univariate saturation scan, all feature selection on first 56.
U,S,V=np.linalg.svd(T[:56],full_matrices=False);Z=np.c_[y,T@V[:2].T]
for key,z in coords(x).items():
 F,ix=poly(z,3);names=[str(k) for k in ix];Fs=[F];
 for size in (1,2):
  for i,j in ([(i,i) for i in range(4)] if size==1 else itertools.combinations(range(4),2)):
   for sign in ([0] if size==1 else [-1,1]):
    q=z[:,i]+sign*z[:,j]
    for n,g in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('atan',np.arctan),('abs',np.abs),('rational',lambda q:q/(1+q*q))]:
     Fs.append(g(q)[:,None]);names.append(f'{n}({i},{sign},{j})')
 F=np.column_stack(Fs);F=F[:,1:];names=names[1:];F-=F[:56].mean(0);F/=np.maximum(np.linalg.norm(F[:56],axis=0),1e-9)
 for j in range(3):
  Y=Z[:,j];Y=Y-Y[:56].mean();r=Y[:56].copy();J=[];out=[]
  for k in range(12):
   sc=abs(F[:56].T@r);sc[J]=-1;a=np.argmax(sc);J.append(a);c=lstsq(F[:56,J],Y[:56])[0];r=Y[:56]-F[:56,J]@c
   if k in (2,4,7,11):out.append(round(np.mean(abs(F[56:,J]@c-Y[56:])),5))
  print(key,j,out,[names[i] for i in J[:8]])
