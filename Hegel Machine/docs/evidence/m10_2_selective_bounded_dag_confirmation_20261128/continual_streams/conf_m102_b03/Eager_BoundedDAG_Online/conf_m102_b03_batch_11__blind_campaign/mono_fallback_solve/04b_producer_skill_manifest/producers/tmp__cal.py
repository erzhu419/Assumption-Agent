from compose import *
from itertools import combinations_with_replacement as cwr
u=U[:,1:];g=np.sort(ids); v=V[g]; R=V-U@M
np.set_printoptions(precision=6,suppress=True)
print('u ranges',u.min(0),u.max(0),'analytic check:');print(np.c_[u[g],R[g]][:25])
F={}
for i in range(4):
 for j in range(i+1,4):
  for k in (-1,1): F[f'x{i}{k:+d}x{j}']=u[:,i]+k*u[:,j]
for i in range(4):F[f'x{i}']=u[:,i]
D={}
for k,w in F.items():
 for nm,f in [('sq',lambda x:x*x),('cb',lambda x:x**3),('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('soft',lambda x:x/(1+abs(x))),('rat',lambda x:1/(1+x*x)),('signedsq',lambda x:x*abs(x)),('clip',lambda x:np.clip(x,-1,1))]:D[k+nm]=f(w)
for d in (2,3):
 for k in cwr(range(4),d):D[str(k)]=np.prod(u[:,k],axis=1)
keys=list(D); d=np.array(list(D.values())).T; d=d-U@ls(U[g],d[g]); d/=np.maximum(1e-12,np.linalg.norm(d[g],axis=0));
for j in range(4):
 f=lambda I:np.c_[U,d[:,I]]
 I=[];r=R[g,j];print('output',j,'initial',np.std(r),'top',[(keys[k],round((d[g].T@r)[k],5)) for k in np.argsort(-abs(d[g].T@r))[:12]])
 for it in range(6):
  I.append(np.argmax(abs(d[g].T@r))); p=ls(f(I)[g],v[:,j]);r=v[:,j]-f(I)[g]@p; print(keys[I[-1]],np.std(r),p)
for d in (2,3):
 f=one(np.c_[u,*[np.prod(u[:,k],axis=1) for l in range(2,d+1) for k in cwr(range(4),l)] ]); c=ls(f[g],v);print('full degree',d,np.std(f[g]@c-v,axis=0)); print(c[np.linalg.norm(c,axis=1)>.04])
