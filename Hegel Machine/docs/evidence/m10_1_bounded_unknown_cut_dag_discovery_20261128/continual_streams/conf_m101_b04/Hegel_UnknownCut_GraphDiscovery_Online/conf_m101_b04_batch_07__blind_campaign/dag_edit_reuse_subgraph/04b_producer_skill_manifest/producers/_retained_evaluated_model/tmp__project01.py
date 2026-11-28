exec(open('/tmp/operator01.py').read().split('for base in')[0])
from scipy.linalg import eigh
folds=[np.arange(52,72),*np.array_split(np.arange(len(x)),4),np.argsort(np.max(np.abs(x),axis=1))[-16:]]
qs=list(itertools.combinations_with_replacement(range(4),2)); pairs=list(itertools.combinations(range(4),2))
A=np.c_[ones,b]; q=poly(b,2); beta=np.linalg.lstsq(q,y,rcond=None)[0]; H=np.zeros((4,4))
for (j,k),r in zip(qs,beta[5:]): H[j,k]=H[k,j]=r if j==k else r/2
print('quadratic eigensystem',eigh(H)); print('traj singular values',np.linalg.svd(z,compute_uv=False))
forms={'sine': lambda h,k:np.sin(h+k), 'cosine':lambda h,k:np.cos(h+k), 'tanh':lambda h,k:np.tanh(h+k), 'square':lambda h,k:h*h, 'saturation': lambda h,k:(h+k)**2/(1+(h+k)**2), 'softplus':lambda h,k:np.logaddexp(0,h+k), 'atan':lambda h,k:np.arctan(h+k)}
R=np.random.default_rng(17); starts=np.r_[np.eye(4),-np.eye(4),R.normal(size=(8,4))]; summary=[]
for coords,C in [('b',b),('bx',np.c_[b,x])]:
 for name,f in forms.items():
  held=[]; fit_err=[]; wins=[]
  for te in folds:
   tr=np.setdiff1d(np.arange(len(y)),te); D=A[tr]; Q=np.linalg.qr(D)[0]; r=y[tr]-Q@(Q.T@y[tr]); ct=C[tr]
   def col(p): return f(ct@p[:-1],p[-1]);
   def residual(p):
    v=col(p); v=v-Q@(Q.T@v); return r-v*(v@r)/max(v@v,1e-16)
   best=None
   for a in starts:
    start=np.r_[a,np.zeros(C.shape[1]-4),0.3]; opt=least_squares(residual,start,bounds=(-12,12),max_nfev=80); score=opt.fun@opt.fun
    if best is None or score<best[0]: best=(score,opt.x)
   p0=best[1]; H=np.c_[A,f(C@p0[:-1],p0[-1])]; coef=np.linalg.lstsq(H[tr],y[tr],rcond=None)[0]; held.append(np.mean(abs(H[te]@coef-y[te]))); fit_err.append(np.mean(abs(H[tr]@coef-y[tr]))); wins.append(np.r_[p0,coef[-1]])
  print(coords,name,np.round(held,5), np.round(fit_err,5),np.round(wins[0],3),flush=True)
