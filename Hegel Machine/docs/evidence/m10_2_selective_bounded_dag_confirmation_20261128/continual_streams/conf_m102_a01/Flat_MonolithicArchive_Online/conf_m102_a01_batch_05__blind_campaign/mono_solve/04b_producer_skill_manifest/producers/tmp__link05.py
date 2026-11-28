exec(open('/tmp/probe05.py').read().split('for k,w in coords.items():')[0])
from scipy.optimize import least_squares
w=z; Q=np.column_stack([w[:,i]*w[:,j] for i,j in combinations_with_replacement(range(4),2)]); F=np.c_[A(w),Q]; time=s['eval_times']; g=1-np.exp(-.8044*time); amp=t@g/(g@g); L=A(w)
for label,y0 in [('y',y),('amp',amp)]:
 print(label, 'quadratic',np.linalg.lstsq(F,y0,rcond=None)[0].round(3));
 for name,fun in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('rational',lambda q:q/(1+q*q)),('asinh',np.arcsinh)]:
  for add in (False,True):
   ev=[]
   for ix in splits[:5]:
    i,j=ix[:52],ix[52:]; f=F[i]; l=L[i]; q=np.linalg.lstsq(f,y0[i],rcond=None)[0]; best=None
    for a in (.25, .65, 1.2):
     def calc(p,jj):return p[-1]*fun(F[jj]@p[:F.shape[1]])+(L[jj]@p[F.shape[1]:-1] if add else 0.)
     p=np.r_[q*a,np.zeros(L.shape[1]) if add else [],1/a]; res=least_squares(lambda p:calc(p,i)-y0[i],p,max_nfev=170)
     if best is None or res.cost<best.cost:best=res
    ev.append([np.mean(abs(calc(best.x,i)-y0[i])),np.mean(abs(calc(best.x,j)-y0[j]))]);
   print(name,add,np.round(ev[0],4),'avg',np.mean(ev,0).round(4),'max',np.max(ev,0).round(4))
# Cross-check measurement residuals for dominant temporal mode against its derivative and smooth curvature.
r=t-amp[:,None]*g; print('resid time means',r.mean(0).round(5),'std',r.std(0).round(5)); print('resid corr amp',np.corrcoef(np.c_[amp,r].T)[0].round(3)); print('amp corr y',np.corrcoef(amp,y)[0,1]);
