exec(open('/tmp/search.py').read().split('for basis in')[0])
from scipy.optimize import least_squares
P=np.c_[a(z),z[:,0]*z[:,2],z[:,1]*z[:,3]]
shapes={'tanh':np.tanh,'rational':lambda q:q/(1+q*q),'sin':np.sin,'asinh':np.arcsinh,'atan':np.arctan}
for name,g in shapes.items():
 for loc in ('outside','inside','split'):
  for tag,X in [('pool',base),('pair',P),('full',Q)]:
   errs=[]; costs=[]
   for k,(i,j) in enumerate(splits):
    n=X.shape[1]; b=fit(X[i],y[i]); l=5
    def pred(p,X):
     q=X@p[:n]; offset,ampl,freq=p[n:n+3];
     if loc=='outside': h=X[:,:l]@p[:l]
     elif loc=='inside': h=q
     else: h=q+X[:,l:]@p[n+3:]
     return q+ampl*g(freq*h+offset)
    init=[np.r_[b,0,amp,fr,np.zeros(n-l if loc=='split' else 0)] for fr in (.65,1.3,1.7,2.5) for amp in (.2,.45)]
    bounds=([-np.inf]*(n+2)+[.12]+[-np.inf]*(n-l if loc=='split' else 0),[np.inf]*(n+2)+[4.0]+[np.inf]*(n-l if loc=='split' else 0))
    opts=[least_squares(lambda p:pred(p,X[i])-y[i],p0,max_nfev=180,bounds=bounds) for p0 in init]; opt=min(opts,key=lambda o:o.cost); errs.append(np.mean(abs(pred(opt.x,X[j])-y[j]))); costs.append(np.mean(abs(opt.fun))); 
    if k==0: first=opt.x
   print(name,loc,tag,'val',np.round(errs,5),'train',np.round(costs[0],5),'params',first.round(4),flush=True)
