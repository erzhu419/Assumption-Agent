exec(open('/tmp/explore.py').read().split('funcs=')[0])
from scipy.linalg import solve
from scipy.spatial.distance import cdist
from scipy.special import expit
c=design(quad(z))@fit(design(quad(u)),s['panel_c']['v1']); d=design(z,np.sqrt(1+.93*z*z))@fit(design(u,np.sqrt(1+.93*u*u)),v)
z,b,c,d=[x[:n] for x in (z,b,c,d)]; groups=dict(z=z,b=b,c=c,d=d); F=np.c_[z,b,c,d]; folds=list(KFold(5,shuffle=True,random_state=13).split(y))+[(np.arange(n-20),np.arange(n-20,n))]
t=t[None,:]; Q=traj; T=t.ravel();
def prof(q):
 def fun(k):
  g=T/(1+k*T); a=q@g/(g@g); return np.mean((q-a*g)**2),a
 k=minimize_scalar(lambda k:fun(k)[0],bounds=(0,2),method='bounded').x; return fun(k)[1],k
h,k=np.array([prof(q) for q in Q]).T
print('amplitude/rate quantiles:',np.quantile(np.c_[h,k],[0,.1,.5,.9,1],axis=0));print('strong rows',np.round(np.c_[y,h,k,z,b,c,d][np.argsort(abs(h))[-12:]],3))
print('rate fit on high signal rows (train weighted RMSE)');
for nm,M in {'1':np.zeros((n,0)),**groups,'z2':quad(z),'b2':quad(b),'d2':quad(d),'absz':abs(z),'absb':abs(b),'all':F,'zfull':np.c_[z,quad(z)],'bfull':np.c_[b,quad(b)],'cd':np.c_[c,d]}.items():
 X=design(M); w=np.minimum(abs(h),2); p=fit(X*w[:,None],k*w); print(nm,np.sqrt(np.mean((w*(X@p-k))**2)),np.round(p,3))
Y=np.c_[y,Q,h,k]
feat={'z':z,'zb':np.c_[z,b],'all':F,'zq':np.c_[z,quad(z)],'zqd':np.c_[z,quad(z),d],'zbq':np.c_[z,b,quad(z)],'zbcd':F,'zbcdep':np.c_[F,quad(z),quad(b)]}
for nm,M in feat.items():
 res=[]
 for length in (.55,.85,1.3,2.,3.5,6.):
  for reg in (1e-5,.001,.015,.1):
   metrics=[]
   for i,j in folds:
    V=(M-M[i].mean(0))/(.1+M[i].std(0)); D=cdist(V,V,'sqeuclidean')/M.shape[1]; K=np.exp(-D/(2*length**2)); X=design(V); G=K+1+V@V.T/M.shape[1]; P=G[np.ix_(j,i)]@solve(G[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i],assume_a='pos'); R=P[:,-2,None]*t/(1+np.maximum(P[:,-1,None],0)*t); metrics.append([np.mean(abs(P[:,0]-y[j])),np.mean(abs(P[:,1:9]-Q[j])),np.mean(abs(R-Q[j]))])
   a=np.array(metrics);res.append((a.mean(0)[0]+a.mean(0)[1],length,reg,a.mean(0),a.max(0),a[-1]))
 print('kernel',nm,*sorted(res,key=lambda r:r[0])[:2],sep='\n')
# Sparse search over calibrated coordinate terms and two-/three-factor mechanisms.
D={'1':np.ones(n)}
for nm,M in groups.items():
 for j,x in enumerate(M.T):
  for suff,w in [('',x),('sq',x*x),('cu',x**3),('abs',abs(x)),('tanh',np.tanh(x)),('sin',np.sin(x)),('rat',x/(1+x*x)),('sqrt',np.sqrt(1+x*x))]:D[f'{nm}{j}{suff}']=w
for (a,x),(bb,xx) in itertools.combinations(list(D.items()),2):
 if len(a)==len(bb)==2:D[a+'*'+bb]=x*xx
for js in itertools.combinations_with_replacement(range(4),3):D['z'+''.join(map(str,js))]=np.prod(z[:,js],axis=1)
keys=list(D); M=np.column_stack(list(D.values()));
for name,target,w in [('y',y,np.ones(n)),('drive',h,np.ones(n)),('rate',k,np.minimum(abs(h),2))]:
 base=np.c_[np.ones(n),z,b,c,d];A=base*w[:,None];R=M*w[:,None];l=[]
 for _ in range(7):
  p=fit(A,target*w);e=target*w-A@p;S=R-A@fit(A,R);score=np.abs(e@S)/(np.linalg.norm(S,axis=0)+1e-9);j=np.argmax(score);l.append((keys[j],round(float(score[j]),4)));A=np.c_[A,R[:,j]]
 print('residual terms',name,l)
