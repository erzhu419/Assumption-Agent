import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def rms(x): return np.mean(x*x)**.5
p=s['panel_a']; u,v=p['u'],p['v']; cal=[]
for x,y in zip(u.T,v.T):
 i,j=np.triu_indices(len(x),1); m=np.median((y[i]-y[j])/(x[i]-x[j])); r=abs(y-m*x-np.median(y-m*x)); k=r<6*np.median(r); cal.append(ls(aff(x[k,None]),y[k]))
o,g=np.array(cal).T
z=(s['train']['x']-o)/g
p=s['panel_c']; u=p['u']; t=s['eval_times']; Y=s['train']['traj']; y=s['train']['y']
def branch(fn,v,q,bounds):
 def err(q):
  f=aff(fn(u,*q)); return (f@ls(f,v)-v).ravel()
 fit=least_squares(err,q,bounds=bounds,gtol=1e-13,ftol=1e-13,xtol=1e-13); F=aff(fn(u,*fit.x)); M=ls(F,v); print('branch',fit.x,rms(fit.fun),'matrix',M,'singular',np.linalg.svd(M[1:],compute_uv=False)); return aff(fn(z,*fit.x))@M
A=branch(lambda u,a,k:a*u+np.tanh(k*u),p['v1'],[.5,2.2],([0,.4],[3,5])); B=branch(lambda u,k:np.cos(k*u),p['v2'],[2.4],([1.9],[2.9])); p=s['panel_b']; M=ls(aff(p['u']),p['v']); C=aff(z)@M; print('panel B matrix singular',np.linalg.svd(M[1:],compute_uv=False));
def T(q): return -np.expm1(-t[:,None]*q)
def rr(q):
 F=T(q); return (F@ls(F,Y.T)-Y.T).ravel()
q=least_squares(rr,[.42,1.8]).x; U=ls(T(q),Y.T).T; print('rates',q,'amplitude sd',U.std(0),'y joint relation',ls(aff(U),y),rms(y-aff(U)@ls(aff(U),y)))
F={}
for name,V in [('z',z),('A',A),('B',B),('C',C)]:
 for i in range(4): F[name+str(i)]=V[:,i]
for na,V,nb,W in [('A',A,'B',B),('C',C,'B',B),('z',z,'B',B),('z',z,'A',A),('A',A,'A',A),('B',B,'B',B),('C',C,'C',C),('z',z,'z',z)]:
 for i in range(4):
  for j in range(i if na==nb else 0,4): F[f'{na}{i}{nb}{j}']=V[:,i]*W[:,j]
 if na!=nb: F[f'{na}.{nb}']=np.sum(V*W,axis=1)
for i in range(4):
 for j in range(i+1,4): F[f'z{i}{j}']=z[:,i]*z[:,j]
for groups in [['z'],['A'],['B'],['A','B'],['A','B','A.B'],['A','B','A.B','A0B1'],['A','B','AB'],['z','A','B','AB']]:
 keys=[k for k in F if k in groups or (len(k)==2 and k[0] in groups) or ('AB'in groups and len(k)==4 and k[0]=='A' and k[2]=='B')]; X=aff(np.array([F[k] for k in keys]).T); Q=np.c_[y,Y]; fit=ls(X[:56],Q[:56]); D=X@fit-Q; print('groups',groups,'fit/val',*(np.round([np.mean(abs(D[sl,0])),np.mean(abs(D[sl,1:]))],5) for sl in [slice(56),slice(56,None)]))
X=np.array(list(F.values())).T; keys=np.array(list(F)); idx=np.arange(56); W=np.c_[y,U]
for j,name in enumerate(['y','slow','fast']):
 w=W[:,j]; active=[]
 for step in range(15):
  V=aff(X[:,active]); e=w-V@ls(V[idx],w[idx]); xf=X-V@ls(V[idx],X[idx]); score=abs(xf[idx].T@e[idx])/(np.linalg.norm(xf[idx],axis=0)+1e-10); score[active]=-1; k=np.argmax(score); active.append(k); V=aff(X[:,active]); coef=ls(V[idx],w[idx]); e=V@coef-w; print(name,len(active),keys[k],round(coef[-1],4),round(rms(e[:56]),5),round(rms(e[56:]),5))
 print(list(zip(keys[active],np.round(coef[1:],4))))
