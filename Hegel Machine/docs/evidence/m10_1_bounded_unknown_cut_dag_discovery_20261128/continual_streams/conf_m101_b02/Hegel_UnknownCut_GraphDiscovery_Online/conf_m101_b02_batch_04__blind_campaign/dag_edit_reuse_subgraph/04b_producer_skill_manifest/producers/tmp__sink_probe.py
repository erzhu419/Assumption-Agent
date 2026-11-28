exec(open('/tmp/explore.py').read().split('funcs=')[0])
pairs=quad(z); c=design(pairs)@fit(design(quad(u)),s['panel_c']['v1']); d=design(z,np.sqrt(1+.93*z*z))@fit(design(u,np.sqrt(1+.93*u*u)),v)
Z=design(z,b,c,d)[:n]; Z=(Z[:,1:]-Z[:,1:].mean(0))/Z[:,1:].std(0); X=design(Z)
folds=list(KFold(5,shuffle=True,random_state=13).split(y))+[(np.arange(n-20),np.arange(n-20,n))]
def profile(law):
 rec=[]
 for row in traj:
  def err(k):
   p=law(k); a=row@p/(p@p); return np.mean((row-a*p)**2),a
  opt=minimize_scalar(lambda k:err(k)[0],bounds=(.0001,3),method='bounded'); e,a=err(opt.x); rec.append((a,opt.x,e**.5))
 return np.array(rec)
laws={'exp':lambda k:-np.expm1(-k*t)/k,'log':lambda k:np.log1p(k*t)/k,'rat':lambda k:t/(1+k*t),'sqrt':lambda k:2*t/(1+np.sqrt(1+k*t)),'tanh':lambda k:np.tanh(k*t)/k}
for nm,law in laws.items():
 r=profile(law); print('law',nm,np.quantile(r,[.1,.5,.9],axis=0).round(4),'signed mean residual by time',np.mean(np.sign(r[:,0,None])*(traj-r[:,0,None]*np.array([law(k) for k in r[:,1]])),axis=0).round(4))
# Use the last time point to anchor amplitude while learning a common smooth rate.
from scipy.special import expit
for k,xx in {'base':X,'poly':design(z,b,c,d,quad(z))[:n],'element':design(z,b,c,d,*[fn(q) for q in (b,c,d) for fn in (np.tanh,np.sin,np.cos)])[:n]}.items():
 xx=xx.copy(); xx[:,1:]=(xx[:,1:]-xx[:,1:].mean(0))/xx[:,1:].std(0)
 for ridge in (.01,.2,1.):
  a=[]
  for i,j in folds:
   p=np.linalg.solve(xx[i].T@xx[i]+ridge*np.diag(np.r_[0,np.ones(xx.shape[1]-1)]),xx[i].T@np.c_[y[i],traj[i]]); a.append(np.mean(abs(xx[j]@p-np.c_[y[j],traj[j]]),axis=0))
  a=np.array(a); print('ridge',k,ridge,'tail',a[-1,0],a[-1,1:].mean(),'worst',a[:,0].max(),a[:,1:].mean(1).max())
# Compare partially linear single-index saturating models with shared calibration features.
groups={'z':np.arange(1,5),'b':np.arange(5,9),'c':np.arange(9,13),'d':np.arange(13,17),'zb':np.arange(1,9),'zbc':np.arange(1,13),'all':np.arange(1,17)}
results=[]
for target in (y,traj[:,-1]):
 for nm,cols in groups.items():
  for fn in ('tanh','sin','asinh'):
   g={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh}[fn]; a=[]; tr=[]; params=[]
   for i,j in folds:
    V=X[:,np.r_[0,cols]]; ridge=.04; ri=np.sqrt(ridge)*np.diag(np.r_[0,np.ones(X.shape[1]-1)]); pinv=np.linalg.pinv(np.r_[X[i],ri]); yi=np.r_[target[i],np.zeros(X.shape[1])]
    def prof(q):
     v=g(V[i]@q[:-1])*q[-1]; p=pinv@(yi-np.r_[v,np.zeros(X.shape[1])]); return p, np.r_[v+X[i]@p-target[i],ri@p, .014*q]
    w=fit(V[i],target[i]); w=w/(np.std(V[i]@w)+1e-8)
    opt=min((least_squares(lambda q:prof(q)[1],np.r_[a0*w,sign*np.std(target[i])],max_nfev=130) for a0 in (.65,1.4) for sign in (1,-1)),key=lambda r:r.cost)
    q=opt.x;p,r=prof(q); a.append(np.mean(abs(X[j]@p+g(V[j]@q[:-1])*q[-1]-target[j])));tr.append(np.mean(abs(r[:len(i)]))); params.append(np.round(q,2))
   results.append((np.mean(a),np.max(a),a[-1],np.mean(tr),nm,fn,params[-1].tolist()))
 print('single index target', 'y' if target is y else 'endpoint'); print(*sorted(results)[0:10],sep='\n');results=[]
