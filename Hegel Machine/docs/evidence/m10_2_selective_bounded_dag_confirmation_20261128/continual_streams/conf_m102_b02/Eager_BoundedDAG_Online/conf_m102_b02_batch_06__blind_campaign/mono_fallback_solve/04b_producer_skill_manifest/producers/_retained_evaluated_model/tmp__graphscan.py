import numpy as np,pickle
from scipy.linalg import qr
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from itertools import combinations
from sparse_heads import old,D,fit

def graph(s,x):
 z=old.coords(s,x); p=s['panel_b']; u=p['u']; h=lambda x:D(np.sign(x)*np.maximum(abs(x)-.35,0)**1.5); wb=fit(h(u),p['v']); B=lambda x:h(x)@wb; b=B(z)
 p=s['panel_c']; u=p['u']; h1=lambda x:D(x*np.roll(x,-1,axis=1)); h2=lambda x:D(np.c_[x,np.cos(1.2726*x)]); w1=fit(h1(u),p['v1']); w2=fit(h2(u),p['v2']); C=lambda x:h1(x)@w1; E=lambda x:h2(x)@w2
 G={'z':z,'b':b,'threshold':np.sign(z)*np.maximum(abs(z)-.35,0)**1.5}; S={'z':z,'b':b};
 for k in [-2,-1.25,-1,-.7,-.35,-.3,-.22,.22,.3,.35,.7,1,1.25,2]: S['b%+gz'%k]=b+k*z;S['z%+gb'%k]=z+k*b
 for v in [.22,.35,1.,1.25]: S['gate%g'%v]=z/(1+v*b*b);S['gateB%g'%v]=b/(1+v*z*z)
 for j,q in list(S.items()):
  G[j]=q
  G['C:'+j]=C(q);G['E:'+j]=E(q);G['B:'+j]=B(q)
 for j,q in list(G.items()):
  if j in S or ':' in j:
   for a in [.7,1.25,2.5]:
    G[f't{a}:{j}']=np.tanh(a*q);G[f's{a}:{j}']=np.sin(a*q);G[f'a{a}:{j}']=np.arcsinh(a*q);G[f'r{a}:{j}']=q/(1+a*q*q)
 return G

def run(s,ix,iv):
 tr=s['train']; G=graph(s,tr['x']); y=tr['y']; T=tr['traj']; V=np.linalg.svd(T[ix],full_matrices=False)[2][0]; Y=np.c_[y,(T@V)*V[-1]]; A=Y[ix]; J=Y[iv]; gs=[]; cache=[]
 for key,val in G.items():
  f=D(val); q,r=qr(f[ix],mode='economic');
  if min(abs(r.diagonal()))<1e-8:continue
  pr=q@(q.T@A); lev=(q*q).sum(1)[:,None]; score=(abs(A-pr)/(1-lev)).mean(0); gs.append((score,key)); cache.append((key,val,q))
 for j in range(2):
  rank=sorted(gs,key=lambda q:q[0][j]); print('single',j,[(round(e[j],4),k) for e,k in rank[:12]],flush=True)
  W=[]; keep={k for _,k in rank[:30]}|{'z','b','threshold'}
  for nm,X,Q in cache:
   if nm not in keep:continue
   for nn,H,_ in cache:
    Z=H[ix]-Q@(Q.T@H[ix]); n=np.linalg.norm(Z,axis=0);good=n>1e-8
    if not any(good):continue
    U,S,VH=np.linalg.svd(Z[:,good]/n[good],full_matrices=False); L=U[:,S>1e-7]; P=np.c_[Q,L]; e=A[:,j]-P@(P.T@A[:,j]); score=np.mean(abs(e)/np.maximum(1-(P*P).sum(1),1e-6)); W.append((score,nm,nn))
  for sc,nm,nn in sorted(W)[:14]:
   f=D(np.c_[G[nm],G[nn]]); pred=f[iv]@fit(f[ix],A[:,j]); print('pair',j,round(sc,5),round(np.mean(abs(pred-J[:,j])),5),nm,nn)
if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];run(s,np.arange(52),np.arange(52,72))
