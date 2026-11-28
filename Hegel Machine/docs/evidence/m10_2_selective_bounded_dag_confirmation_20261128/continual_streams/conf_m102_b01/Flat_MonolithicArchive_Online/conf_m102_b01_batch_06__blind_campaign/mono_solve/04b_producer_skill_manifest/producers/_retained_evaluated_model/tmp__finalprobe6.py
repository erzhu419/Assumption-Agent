import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from refine6 import *
import importlib.util
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m0=importlib.util.module_from_spec(sp);sp.loader.exec_module(m0)
N=len(Y); y=Y[:,0]; st,old=m0._designs(s,x); g=Z.sum(1); core=A(np.c_[w,q,Z]); base=A(np.c_[q,Z]);
splits=[(np.arange(52),np.arange(52,N))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(N) for i in range(20)]]
for v in [np.abs(st@L(st,Y[:,-1])),np.max(abs(z),axis=1),np.max(abs(w),axis=1)]:
 p=np.argsort(v);splits.append((p[:52],p[52:]))
def errors(d,alpha=0):
 ans=[]
 for tr,te in splits:
  T=d[tr]; E=d[te]; yy=y[tr]; sd=np.maximum(T[:,1:].std(0),1e-10); mu=T[:,1:].mean(0);T=(T[:,1:]-mu)/sd;E=(E[:,1:]-mu)/sd
  coef=L(np.r_[T,np.sqrt(alpha*len(tr))*np.eye(T.shape[1])],np.r_[yy-yy.mean(),np.zeros(T.shape[1])]);ans.append(np.mean(abs(yy.mean()+E@coef-y[te])))
 return np.array(ans)
def result(e):return np.round([e[0],e[1:21].mean(),e[1:21].max(),*e[21:]],5)
Ds={};
for bn,d in [('state',st),('base',base),('core',core),('old',old)]:
 for an,a in [('z',z),('w',w),('q',q),('r',r),('R',R),('g',g[:,None]),('Z',Z)]:
  for k in [.22,.5,1.25]:
   for n,b in [(str(j),a[:,j]) for j in range(a.shape[1])]+[('l2',np.linalg.norm(a,axis=1))]:
    h=1/(1+k*b*b)
    for mode,D in [('f',d*h[:,None]),('p',np.c_[d,q[:,0]*h]),('s',np.c_[d,q*h[:,None]]),('t',np.c_[d,q[:,0]*np.tanh(k*b)])]: Ds[bn,an,n,k,mode]=D
 for k in [.22,1.25]:
  for an,a in [('w',w),('q',q),('r',r),('R',R),('z',z)]:
   for mode,D in [('joint',np.c_[d,q/(1+k*a*a)]),('tanh',np.c_[d,np.tanh(k*a)]),('rot',np.c_[d,q/(1+k*np.roll(a,1,axis=1)**2)]),('cubic',np.c_[d,q*a*a])]: Ds[bn,an,'all',k,mode]=D
Ds['old',]=old; Ds['minimal',]=np.c_[base,q[:,0]*w[:,2]**2]
vals=[]
for name,d in Ds.items():
 for al in [0,.001,.01]:
  e=errors(d,al); vals.append((e[0]+e[1]+.65*max(e[2:]),name,al,e))
for _,n,al,e in sorted(vals,key=lambda h:h[0])[:18]: print(n,al,result(e))
for name in [('old',),('minimal',),('old','z','1',.22,'f')]:
 for al in [0,.001,.01]:print('reference',name,al,result(errors(Ds[name],al)))
