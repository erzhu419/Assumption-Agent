from h import *
from itertools import combinations,product
W=Y[:n]; names=[]; cols=[]
def put(name,F):
 F=np.asarray(F).reshape(len(q),-1); cols.extend(F.T); names.extend([(name,i) for i in range(F.shape[1])]);return F
coords={'q':q,'Bq':z,'Ba':B(x),'Bi':zi,'Bplus':q+z,'Bminus':q-z,'Aforward':x*wd[1]+wd[0]}
res=[]
def test(name,f):
 if f.shape[1]>=38:return
 m=metrics(f);res.append((name,m))
for k,p in coords.items():
 C1,D1=C(p),D(p); f=A(np.c_[p,C1,D1]); test((k,'C,D'),f);test((k,'C,D,bilinear'),np.c_[f,(C1[:,:,None]*D1[:,None,:]).reshape(-1,16)]);test((k,'C,D,dot'),np.c_[f,np.sum(C1*D1,axis=1)]);
 for L in [p,C1,D1,B(C1),B(D1),C1+D1,C1-D1,p+C1,p+D1,C(B(p)),D(B(p)),D(C1),C(D1),C1/(1+.22*D1**2),D1/(1+.22*C1**2)]:
  put(k,L)
 put(k+' poly',poly(p,3)[:,1:]);
 for a in (.22,1.25,1.6,2.,2.4,3.,4.,5.):
  t2=np.column_stack([p[:,i]+sgn*p[:,j] for i,j in combinations(range(4),2) for sgn in (-1,1)]); F=np.c_[p,t2];
  for lab,F0 in [('cos',np.cos(a*F)),('sin',np.sin(a*F)),('tanh',np.tanh(a*F)),('odd',F/(1+a*F**2)),('even',F**2/(1+a*F**2)),('abs',abs(F)),('cosprod',np.cos(a*Q(p)[:,1:])),('sinprod',np.sin(a*Q(p)[:,1:]))]:
   put((k,a,lab),F0);test((k,a,lab),np.c_[poly(p,2),F0]);
 for a in (.22,1.25):
  for lab,g in [('norm',np.sum(p*p,1)),('L1',np.sum(abs(p),1)),('max',np.max(abs(p),1))]:test((k,a,lab),np.c_[poly(p,2),poly(p,2)/(1+a*g[:,None])]);
 for name,r in [('C',C1),('D',D1)]:
  put((k,name,'diag'),p*r);put((k,name,'dot'),(p*r).sum(1));put((k,name,'outer'),(p[:,:,None]*r[:,None,:]).reshape(-1,16));
 put((k,'cd','dot'),(C1*D1).sum(1));put((k,'cd','outer'),(C1[:,:,None]*D1[:,None,:]).reshape(-1,16));
print('best compact');print(*[(k,np.round(m,4)) for k,m in sorted(res,key=lambda kv:kv[1][1].sum())[:12]],sep='\n');
F=np.asarray(cols).T; mu=F[:n].mean(0);sd=F[:n].std(0);F=(F-mu)/np.maximum(sd,1e-10); U,S,V=np.linalg.svd(F[:n],full_matrices=False); YC=W-W.mean(0)
for a in np.geomspace(.01,1000,10):
 M=F@(V.T*(S/(S*S+a)))@U.T@YC+W.mean(0);e=abs(M-Y); print('ridge',a,e[52:,0].mean(),e[52:,1:].mean())
for base in [np.ones((len(q),1)),poly(q,2)]:
 for target in [Y[:,0:1],Y[:,1:]@np.linalg.svd(T[:n],full_matrices=False)[2][:3].T,Y]:
  dd=[]; XX=base.copy();
  for it in range(20):
   U=np.linalg.svd(XX[:n],full_matrices=False)[0]; E=target[:n]-U@(U.T@target[:n]); FF=F[:n]-U@(U.T@F[:n]); norms=np.linalg.norm(FF,axis=0); j=np.argmax(np.linalg.norm(FF.T@E,axis=1)/np.maximum(norms,1e-9)); XX=np.c_[XX,F[:,j]];dd.append(names[j]);
   if it in (0,1,2,4,9,14,19):
    M=XX@ls(XX[:n],target[:n]);print('omp',base.shape[1],target.shape[1],it,names[j],np.mean(abs(M-target)[:n]),np.mean(abs(M-target)[n:]));
print('x y traj samples');print(np.c_[q,y,T[:,0],T[:,-1]][:20]);print('B',b)
