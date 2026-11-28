exec(open('/tmp/invert.py').read().split('for d in range(1,5):')[0])
from scipy.optimize import minimize_scalar
from sklearn.linear_model import orthogonal_mp, Ridge
from sklearn.model_selection import KFold
np.set_printoptions(precision=4,suppress=True)
def sd(u,d):return np.column_stack([np.ones(len(u))]+[u**k for k in range(1,d+1)])
def check(A,V):
 B=np.linalg.lstsq(A,V,rcond=None)[0]; Q=np.linalg.qr(A)[0]; return np.mean(abs((V-A@B)/np.maximum(.002,1-(Q*Q).sum(1))[:,None])),B
F={'z':z};
for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u,v=s[k]['u'],s[k][j]; out=[]
 def candidate(n,fn):
  A=fn(u); e,b=check(A,v);out.append((e,n,fn(z)@b,b));
 for d in range(1,6):candidate('poly'+str(d),lambda x,d=d:sd(x,d))
 for g in [.5,.75,1,1.25,1.5,1.75,2,2.5,3,3.5,4,4.5,5,5.5,6]:
  for n,f in [('fourier',lambda x:np.c_[np.ones(len(x)),np.sin(x),np.cos(x)]),('tanh',lambda x:np.c_[np.ones(len(x)),np.tanh(x),np.tanh(x)**2]),('sincos+x',lambda x:np.c_[np.ones(len(x)),x,np.sin(x),np.cos(x)])]:candidate(n+str(g),lambda x,g=g,f=f:f(g*x))
 for e,n,w,b in sorted(out,key=lambda a:a[0])[:5]: print(k,j,n,e,b if n.startswith('poly') else '')
 e,n,w,b=min(out,key=lambda a:a[0]);F[k+j]=w
 # Check for an entrywise output link after affine mixing.
 for n,T in [('tanh',np.arctanh),('sin',np.arcsin)]:
  if abs(v).max()<1:
   print(k,j,n,check(sd(u,1),T(v))[0],check(sd(u,2),T(v))[0])
print('target range',Y.min(0),Y.max(0),'times',s['eval_times'])
from itertools import combinations
splits=list(KFold(4,shuffle=True,random_state=23).split(z))+[(np.arange(N),np.arange(N,len(z)))]
def trial(name,A,alphas=(0,),omp=False):
 result=[]
 for alpha in alphas:
  em=[]
  for tr,va in splits:
   m=A[tr,1:].mean(0);sc=np.maximum(A[tr,1:].std(0),1e-8); D=(A[:,1:]-m)/sc;ym=Y[tr].mean(0)
   if omp:
    # use three high-SNR targets, with trailing trajectory channels projected onto their training temporal modes
    _,_,vt=np.linalg.svd(Y[tr,1:]-ym[1:],full_matrices=False); V=np.c_[Y[tr,0]-ym[0],(Y[tr,1:]-ym[1:])@vt[:2].T]; B=orthogonal_mp(D[tr],V,n_nonzero_coefs=min(D.shape[1],len(tr)-2,int(alpha))); P=D[va]@B; P=np.c_[P[:,0],P[:,1:]@vt[:2]]+ym
   else:
    B=(np.linalg.lstsq(D[tr],Y[tr]-ym,rcond=1e-9)[0] if alpha==0 else np.linalg.solve(D[tr]@D[tr].T+alpha*np.eye(len(tr)),Y[tr]-ym)); P=D[va]@B+ym if alpha==0 else (D[va]@D[tr].T)@B+ym
   E=abs(P-Y[va]);em.append([E[:,0].mean(),E[:,1:].mean()]);
  result.append((name,alpha,np.round(np.mean(em[:-1],0),5),np.round(em[-1],5)))
 return result
res=[]
for k in range(1,5):
 for keys in combinations(F,k):
  W=np.column_stack([F[k] for k in keys]); A=np.column_stack([np.prod(W[:,j],1) for d in range(3) for j in cwr(range(W.shape[1]),d)]);res+=trial(str(keys),A,(0,.001,.03,.3,3));res+=trial(str(keys)+' omp',A,(5,10,15,20),True)
for c in (0,1):
 for row in sorted(res,key=lambda a:a[2][c])[:12]: print('score',c,row)
np.savez('/tmp/features.npz',**F)
