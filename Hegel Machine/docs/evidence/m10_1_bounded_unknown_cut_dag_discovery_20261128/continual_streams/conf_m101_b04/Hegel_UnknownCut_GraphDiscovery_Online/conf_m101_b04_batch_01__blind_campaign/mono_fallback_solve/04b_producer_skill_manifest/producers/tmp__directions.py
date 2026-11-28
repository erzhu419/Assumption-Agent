exec(open('/tmp/explore.py').read().split('for name,z')[0])
from scipy.optimize import least_squares
from scipy.linalg import eigh
p=s['panel_c']; u=p['u']; a=np.c_[u,np.ones(len(u))]; q=mono(u,[0,1,2]); coef=np.linalg.lstsq(q,p['v2'],rcond=None)[0]; print('C2 coefficients',coef)
Cs=[]
for v in p['v1'].T:
 w=np.linalg.lstsq(a,v,rcond=None)[0]
 for kind in ['tanh','tanh+linear','sin','sin+linear']:
  act=np.tanh if 'tanh' in kind else np.sin
  def res(h):
   f=act(a@h); d=np.c_[a,f] if '+' in kind else np.c_[np.ones(len(a)),f]; return d@np.linalg.lstsq(d,v,rcond=None)[0]-v
  opt=min([least_squares(res,k*w,max_nfev=350) for k in [.5,1.3,2.3]],key=lambda m:m.cost); print('C1',kind,np.std(opt.fun),opt.x)
  if kind=='tanh': Cs.append(opt.x)
C=np.array(Cs).T
rng=np.random.default_rng(32); splits=[(np.arange(52),np.arange(52,72))]+[(p[:52],p[52:]) for p in [rng.permutation(72) for _ in range(8)]]
Z={'x':x,'A':x@A[:4]+A[4], 'Ai':(x-A[4])@np.linalg.inv(A[:4])}; Z.update({k+n:z@b[:4]+b[4] for k,z in list(Z.items()) for n,b in [('B',B),('Bi',np.vstack([np.linalg.inv(B[:4]),-B[4]@np.linalg.inv(B[:4])])),('C',C)]})
def fit(P,alpha):
 P=np.c_[np.ones(len(x)),P]; err=[]
 for a,b in splits:
  d=P[a].std(0); d[d<1e-8]=1; pa=P[a]/d; pb=P[b]/d; co=np.linalg.solve(pa.T@pa+alpha*np.diag(np.r_[0,np.ones(P.shape[1]-1)]),pa.T@T[a]); err.append(abs(pb@co-T[b]).mean(0))
 e=np.array(err);return e[0],e.mean(0),e.max(0)
for n,z in Z.items():
 q=mono(z,[1,2]); results=[]
 for k in [.5,1,1.6,2.3]:
  for name,f in [('tanh',np.tanh(k*z)),('sin',np.sin(k*z)),('rat',z/(1+(k*z)**2)),('rsq',1/(1+(k*z)**2)),('soft',z/np.sqrt(1+(k*z)**2))]:
   for cross in [False,True]:
    ff=np.c_[f, (f[:,:,None]*z[:,None,:]).reshape(len(z),-1)] if cross else f
    for al in [.001,.03,.3]: results.append((name,k,cross,al,fit(np.c_[q,ff],al)))
 for j in [0,1]:
  l=sorted(results,key=lambda r:r[4][1][j])[:2]; print(n,j,l)
for j in range(3):
 p=mono(x,[0,1,2,3]); co=np.linalg.lstsq(p,T[:,j],rcond=None)[0]; print('target',j,'linear',co[1:5],'quadratic',co[5:15],'cubic',co[15:]); print('linear residual rms',np.std(T[:,j]-p[:,:5]@np.linalg.lstsq(p[:,:5],T[:,j],rcond=None)[0]))
