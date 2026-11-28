from merged_probe import *
from scipy.special import erf
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import KFold
from itertools import combinations
U=S['panel_b']['u']; V=S['panel_b']['v'];
def bench(fun,initial,lb,ub):
 def res(t):
  ff=np.c_[ones(U),fun(U,t)]; return (ff@ls(ff,V)-V).ravel()
 z=least_squares(res,initial,bounds=(lb,ub),max_nfev=150); return np.std(z.fun), z.x
best=[]
for name,g in {'tanh':lambda x,k:np.tanh(k*x),'rat':lambda x,k:x/(1+k*x*x),'exp':lambda x,k:x*np.exp(-k*x*x),'sin':lambda x,k:np.sin(k*x),'erf':lambda x,k:erf(k*x),'sig':lambda x,k:x/(1+k*abs(x)),'sqrt':lambda x,k:x/np.sqrt(1+k*x*x),'asinh':lambda x,k:np.arcsinh(k*x),'atan':lambda x,k:np.arctan(k*x),'abs':lambda x,k:x*abs(x)**k,'cub':lambda x,k:x**3}.items():
 for nm,h in {'lin':lambda x:x,'cub':lambda x:x**3,'abs':lambda x:x*abs(x)}.items():
  for nm2,q in {'none':lambda x:0,'square':lambda x:x*x,'cub':lambda x:x**3,'abs':lambda x:x*abs(x)}.items():
   if nm2==nm: continue
   f=lambda x,t,g=g,h=h,q=q:h(x)+t[0]*g(x,t[1])+t[2]*q(x)
   out=min([bench(f,[a,k,0],[-10,.12,-10],[10,10,10]) for a in [-.75,.75] for k in [1.,2.5]],key=lambda r:r[0]); best.append((*out,name,nm,nm2,f))
print('B BEST',[(r[0],r[1].round(5),r[2:5]) for r in sorted(best,key=lambda r:r[0])[:10]])
e,p,*_,fn=min(best,key=lambda r:r[0]); M=ls(np.c_[ones(U),fn(U,p)],V); b1=lambda x:np.c_[ones(x),fn(x,p)]@M
b0=b_fit(S['panel_b']); c,d=c_fit(S['panel_c']); x=S['train']['x']; a=a_fit(S['panel_a']); u=a_inv(x,a)[0]; Y=np.c_[S['train']['y'],S['train']['traj']]; N=len(x)
splits=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=17).split(x)); I=np.eye(N)
def cv(F,k=1,reg=.001):
 E=[]
 for i,j in splits:
  f=(F-F[i].mean(0))/np.maximum(1e-7,F[i].std(0)); G=f@f[i].T/F.shape[1]; K=(1+G)**k
  e=np.mean(abs(K[j]@np.linalg.solve(K[i]+reg*np.eye(len(i)),Y[i])-Y[j]),axis=0); E.append([e[0],e[1:].mean(),e[-1]])
 E=np.array(E); return np.round(np.r_[E[0,:2],E.max(0)],5)
for bn,b in [('old',b0),('new',b1)]:
 P,Q,R=b(u),c(u),d(u); print(bn,'ranges',[(np.round(F.std(0),3),np.round(F.min(0),2),np.round(F.max(0),2)) for F in [u,P,Q,R]])
 B={'u':u,'b':P,'c':Q,'d':R,'x':x}; F=B.copy()
 for tag,z in B.items():
  for key,fun in [('b',b),('c',c),('d',d)]:
   if tag!='u': F[key+tag]=fun(z)
 G=np.column_stack(list(F.values())); names=list(F)
 opts=[]
 for l in range(1,4):
  for sub in combinations(names,l):
   z=np.column_stack([F[s] for s in sub]); score=cv(z,1,1e-6); opts.append((max(score[2:4]),sub,score))
 print(bn,'LINEAR BEST',sorted(opts,key=lambda r:r[0])[:11]);
 for g in [(k,) for k in names]+[('u','b','c','d','bd','cb','db','cd','dc','bc'),('b','cb','db'),('u','b','cb','db'),('b','c','d','cb','db'),('b','c','d','cb','db','bd','bc')]:
  z=np.column_stack([F[s] for s in g]); vals=[(cv(z,k,r),k,r) for k in [1,2,3] for r in [1e-5,.0003,.003]]; print(bn,g,min(vals,key=lambda v:max(v[0][2:4])))
 # Sparse supervised readout selected inside each fold.
 for lam in [.0003,.001,.003,.006,.012]:
  E=[]; az=[]
  for i,j in splits:
   f=(G-G[i].mean(0))/np.maximum(1e-7,G[i].std(0)); m=MultiTaskLasso(alpha=lam,max_iter=5000,tol=1e-4).fit(f[i],Y[i][:,[0,4,8]]); ids=np.linalg.norm(m.coef_,axis=0)>1e-7; A=np.c_[ones(f),f[:,ids]]; err=np.mean(abs(A[j]@ls(A[i],Y[i])-Y[j]),axis=0); E.append([err[0],err[1:].mean()]); az.append(ids.sum())
  print(bn,'sparse',lam,np.round(E[0],4),np.round(np.max(E,axis=0),4),az)
