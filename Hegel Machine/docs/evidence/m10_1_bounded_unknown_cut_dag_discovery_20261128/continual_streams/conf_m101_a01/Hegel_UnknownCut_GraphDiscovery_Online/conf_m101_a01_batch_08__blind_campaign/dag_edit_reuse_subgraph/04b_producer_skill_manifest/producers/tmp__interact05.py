exec(open('/tmp/discover05.py').read().split('P=[]')[0])
from itertools import combinations_with_replacement, product
from scipy.optimize import minimize_scalar
P=[]
for V,mode in [(v1,'tanh'),(v2,'cos')]:
 w,f=calib(u,V,mode,True); A=basis(u,mode,w,True); c=np.linalg.lstsq(A,V,rcond=None)[0];print(mode,w,np.round(c,3)); P.append(f(q))
a,b=P
splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [np.random.default_rng(i).permutation(72) for i in range(12)]]
def cv(A,name,Y=y, al=(0,.0001,.001,.01,.05,.2,1.)):
 A=A[:,1:] if np.std(A[:,0])<1e-10 else A
 L=[]
 for i,j in splits:
  mean=A[i].mean(0);sd=A[i].std(0);sd[sd<1e-10]=1
  X=(A[i]-mean)/sd;Z=(A[j]-mean)/sd
  U,S,V=np.linalg.svd(X,full_matrices=False); M=(U.T@(Y[i]-Y[i].mean(0))).reshape(len(S),-1);Yj=Y[j].reshape(len(j),-1)
  L.append([np.abs((Z@V.T)@((np.where(S>1e-8,S/(S*S+max(l,1e-16)),0))[:,None]*M)+Y[i].mean(0)-Yj).mean() for l in al])
 L=np.array(L);k=L.mean(0).argmin();print(name,A.shape[1],al[k],np.round([L[0,k],L[:,k].mean(),L[:,k].max()],5))
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
Q=poly(q);h=np.cos(q*1.3)-1;g=np.tanh(q*2.3);D=np.c_[q,h,g,a,b]
for nm,z in [('cos',h),('tanh',g),('a',a),('b',b),('ab',np.c_[a,b]),('trig',np.c_[h,g])]:
 for nn,A in [('main',np.c_[Q,z]),('crossq',np.c_[Q,z,cross(q,z)]),('crossab',np.c_[Q,z,cross(a,z)]),('crossb',np.c_[Q,z,cross(b,z)]),('crossall',np.c_[Q,z,cross(np.c_[q,a,b],z)])]:cv(A,nm+nn)
for k in [3,4,5]:cv(poly(q,k),'poly'+str(k))
for z,nm in [(q,'q'),(np.c_[q,b],'qb'),(np.c_[q,a,b],'qab')]:
 for l in [0.5,1,2,4]:
  K=np.exp(-((z[:,None]-z[None,:])**2).sum(2)/(2*l*l));L=[]
  for i,j in splits:
   Ki=K[np.ix_(i,i)];Kj=K[np.ix_(j,i)];L.append([np.mean(np.abs(Kj@np.linalg.solve(Ki+v*np.eye(len(i)),y[i]-y[i].mean())+y[i].mean()-y[j])) for v in [0.0001,.001,.01,.1]])
  L=np.array(L);k=L.mean(0).argmin();print('rbf',nm,l,k,np.round([L[0,k],L[:,k].mean(),L[:,k].max()],5))
for nm,Z in [('q',q),('a',a),('b',b)]:
 A=poly(Z);c=np.linalg.lstsq(A,np.c_[y,T[:,2],T[:,-1]],rcond=None)[0];print(nm,'quad coeff',np.round(c,3).T)
