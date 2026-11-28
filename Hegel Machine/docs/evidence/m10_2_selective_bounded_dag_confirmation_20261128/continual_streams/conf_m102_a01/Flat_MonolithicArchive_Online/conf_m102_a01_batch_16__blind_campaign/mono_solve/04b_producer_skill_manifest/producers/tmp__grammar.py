exec(open('/tmp/probe.py').read().split("for p in ['panel_a'")[0])
from itertools import combinations_with_replacement
from scipy.linalg import cho_solve,cho_factor
from scipy.optimize import minimize
A=lambda a: np.c_[np.ones(len(a)),a]
def soft(a,j):return a/(1+j*abs(a))
def rad(a,j):return np.sqrt(1+j*a*a)-1
p=s['panel_c']; u=p['u']; pc=[]
for key,fun in [('v1',soft),('v2',rad)]:
 def fit(j):
  b=np.c_[A(u),fun(u,j)]; c=ls(b,p[key]); return b@c-p[key],c
 j=minimize_scalar(lambda j:np.sum(fit(j)[0]**2),bounds=(.01,3),method='bounded').x
 print(key,j,fit(j)[1]); pc.append((fun,j,fit(j)[1]))
p=s['panel_b']; B=ls(A(p['u']),p['v']); print('A decontam',ls(A(x),z)); print('B sv',np.linalg.svd(B[1:])[1]);
W={'x':x,'z':z,'h':h,'bz':A(z)@B,'bx':A(x)@B}
for k,w in list(W.items()):
 for i,(f,j,c) in enumerate(pc):W[f'{k}c{i}']=np.c_[A(w),f(w,j)]@c
_,_,V=np.linalg.svd(t[:n],full_matrices=False); a=t@V[0]; Z=np.c_[y,a];
for k,w in W.items():
 groups={'quad':np.stack([w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(4),2)],1),'cubic':np.stack([w[:,i]*w[:,j]*w[:,l] for i,j,l in combinations_with_replacement(range(4),3)],1)}
 for j in [.22,1.25]:
  tag=str(j); ii,jj=np.triu_indices(4,1); d=np.c_[w,w[:,ii]+w[:,jj],w[:,ii]-w[:,jj]]
  for f,g in [('soft',soft(d,j)),('rad',rad(d,j)),('rat',d/(1+j*d*d)),('tanh',np.tanh(j*d)),('sin',np.sin(j*d))]:
   groups[f+tag]=g; groups['x'+f+tag]=(g[:,:,None]*w[:,None,:]).reshape(len(w),-1)
  groups['div'+tag]=(groups['quad'][:,:,None]/(1+j*d[:,None,:]**2)).reshape(len(w),-1)
 base=A(x); Q=np.linalg.qr(base[:n])[0]; H0=Q@Q.T; resid=Z[:n]-H0@Z[:n]; scores=[]
 for tag,F in groups.items():
  F=F-base@ls(base[:n],F[:n]); F=F/np.maximum(np.linalg.norm(F[:n],axis=0),1e-10); G=F[:n]@F[:n].T
  for reg in [.0001,.003,.03,.3,3.]:
   fac=cho_factor(G+reg*np.eye(n)); hi=cho_solve(fac,G); al=cho_solve(fac,resid); rr=(reg*al)/(1-np.diag(H0+hi))[:,None]
   pred=base[n:]@ls(base[:n],Z[:n])+F[n:]@F[:n].T@al
   scores.append((np.mean(abs(rr),0),np.mean(abs(pred-Z[n:]),0),tag,reg))
 print(k,'scalar',[(e[2],e[3],round(e[0][0],4),round(e[1][0],4)) for e in sorted(scores,key=lambda e:e[0][0])[:4]],'traj amp',[(e[2],e[3],round(e[0][1],4),round(e[1][1],4)) for e in sorted(scores,key=lambda e:e[0][1])[:4]])
