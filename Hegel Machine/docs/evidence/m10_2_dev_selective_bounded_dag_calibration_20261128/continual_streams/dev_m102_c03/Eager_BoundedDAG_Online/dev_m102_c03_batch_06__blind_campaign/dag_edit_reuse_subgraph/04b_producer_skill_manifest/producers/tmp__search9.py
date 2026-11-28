import sys,pickle,numpy as np,itertools as it
sys.path.insert(0,'/holdout_task'); from node_01 import _coordinates,_fit
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; Y=np.c_[tr['y'],tr['traj']]; n=len(Y); z,b,c,d,g=_coordinates(s,tr['x'])
print('times',t)
q=lambda r:-np.expm1(-t[:,None]*np.asarray(r));
for k in range(2,6):
 def ef(r):
  a=q(r);return (a@_fit(a,Y[:,1:].T)-Y[:,1:].T).ravel()
 o=least_squares(ef,np.geomspace(.18,5,k),bounds=(.01,30),max_nfev=500);print('rates',o.x,np.std(o.fun));
 if k==3: R=o.x; Q=q(R)
A=np.c_[Y[:,0],_fit(Q,Y[:,1:].T).T];
T=[(p+str(j),a) for p,m in zip('zbcdg',(z,b,c,d,g)) for j,a in enumerate(m.T)];
D=T.copy();
for (a,u),(b0,v) in it.combinations_with_replacement(T,2):
 D.append((a+'*'+b0,u*v))
 for k in [.5,1,2]:
  D.extend([(a+f'/({k}'+b0+'²)',u/(1+k*v*v)),(b0+f'/({k}'+a+'²)',v/(1+k*u*u))]);
 if a!=b0:
  for k in [-2,-1,-.5,.5,1,2]:
   for m in [.5,1,1.5,2,2.4,3,4]:
    w=m*(u+k*v)
    D.extend([(f'{f.__name__}({m}({a}+{k}{b0}))',f(w)) for f in [np.sin,np.cos,np.tanh]])
for a,u in T:
 for m in [.5,1,1.5,2,2.4,3,4,5,6,7,8]:
  D.extend([(f'{f.__name__}({m}{a})',f(m*u)) for f in [np.sin,np.cos,np.tanh]])
for sub in [D[:20+210*7],D]:
 F=np.column_stack([v for _,v in sub]); G=F-F.mean(0);G/=np.maximum(np.linalg.norm(G,axis=0),1e-8)
 for j,y in enumerate(np.c_[A,Y[:,-1]].T):
  h=y-y.mean(); ix=[]; e=h
  for k in range(14):
   ix.append(int(np.argmax(abs(G.T@e))));
   for rep in range(4):
    changed=False
    for i in range(len(ix)):
     jj=ix[:i]+ix[i+1:]; g0=G[:,jj]; basis=np.linalg.qr(g0)[0]; er=h-basis@(basis.T@h); den=np.maximum(1-np.sum((basis.T@G)**2,axis=0),1e-9); sc=abs(G.T@er)/np.sqrt(den);sc[jj]=-1; k0=int(sc.argmax());changed |= k0!=ix[i];ix[i]=k0
    if not changed:break
   e=h-G[:,ix]@_fit(G[:,ix],h)
   if k in [2,5,8,13]: print(len(sub),j,k+1,np.std(e),[sub[i][0] for i in ix],flush=True)
