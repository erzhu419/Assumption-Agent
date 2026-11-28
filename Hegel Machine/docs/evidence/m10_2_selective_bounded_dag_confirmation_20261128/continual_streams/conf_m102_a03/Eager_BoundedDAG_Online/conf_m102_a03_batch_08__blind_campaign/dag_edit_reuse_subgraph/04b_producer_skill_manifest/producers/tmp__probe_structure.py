exec(open('/tmp/probe_panels.py').read().split('for k,panel')[0])
from scipy.optimize import least_squares
B=s['panel_b']; C=s['panel_c']; Ut,St,Vt=np.linalg.svd(T,full_matrices=False); Y=np.c_[y,Ut[:,0]*St[0]/2]
for name,x,v in [('b',B['u'],B['v']),('c1',C['u'],C['v1']),('c2',C['u'],C['v2']),('train',u,Y)]:
 print(name,'summary');
 for m in range(2,8):
  F=lambda q:np.c_[np.ones(len(q)),*[q**j for j in range(1,m+1)]]; X=F(x); H=X@np.linalg.pinv(X); R=v-H@v; e=np.mean(abs(R),0); loo=np.mean(abs(R/np.maximum(1e-8,1-H.diagonal())[:,None]),0); print(m,np.round(e,5),np.round(loo,5));
  if name=='train': print('test',scores(X,y),scores(X,T))
  if name in ('b','c1') and m==3:
   A=reg(X,v);print('linear/quad/cubic',A[1:5], A[5:9], A[9:]);print('alignment quad and cubic',np.sum(A[1:5]*A[5:9],1)/np.sum(A[1:5]**2,1),np.sum(A[1:5]*A[9:],1)/np.sum(A[1:5]**2,1))
for name,x,v in [('b',B['u'],B['v']),('c1',C['u'],C['v1']),('train',u,Y)]:
 print('rhs model',name); sf=[]
 for fn in [lambda x,k:np.tanh(k*x),lambda x,k:np.sin(k*x),lambda x,k:x/(1+k*x*x),lambda x,k:x+k*x**3,lambda x,k:x/np.sqrt(1+k*x*x)]:
  for k in [.15,.25,.4,.6,.8,1.,1.25,1.6,2.,2.5,3.]:
   w=fn(x,k)
   for mode,X in enumerate([np.c_[np.ones(len(x)),w,x*x],np.c_[np.ones(len(x)),x,w,w*w],np.c_[np.ones(len(x)),x*x,w,w*w]]):
    H=X@np.linalg.pinv(X); R=(v-H@v); loo=np.mean(abs(R/np.maximum(1e-8,1-H.diagonal())[:,None]),0); sf.append((loo.mean(),str(fn.__code__.co_firstlineno),k,mode,loo,np.mean(abs(R))))
 print(*sorted(sf,key=lambda v:v[0])[:4],sep='\n')
# Compare regularized additive and interaction bases on labeled folds.
for tag,F in [('p',lambda x,j:x**j),('c',lambda x,j:np.cos(j*np.arccos(np.clip(x/1.28,-1,1))))]:
 for degree in (3,5,7,9):
  for intr in (1,2,3):
   X=np.c_[np.ones(n),*[F(u,j) for j in range(1,degree+1)],poly(u,intr)[:,5:]]; Q=np.std(X,axis=0);Q[0]=1.; X=X/Q
   for l in (0,.003,.03,.3,1.):
    folds=[(tr,te)]+[(np.setdiff1d(I,h),h) for h in np.array_split(np.random.default_rng(133).permutation(n),6)]; E=[]
    for a,b in folds:
     xx=X[a]; yy=np.c_[y,T][a]; A=np.linalg.lstsq(np.r_[xx,np.sqrt(l)*np.eye(X.shape[1])],np.r_[yy,np.zeros((X.shape[1],yy.shape[1]))],rcond=None)[0]; R=abs(X[b]@A-np.c_[y,T][b]); E.append([R[:,0].mean(),R[:,1:].mean()]);
    if max(E[0])<.062: print('additive',tag,degree,intr,l,np.round(E[0],5),np.round(np.max(E[1:],0),5));
