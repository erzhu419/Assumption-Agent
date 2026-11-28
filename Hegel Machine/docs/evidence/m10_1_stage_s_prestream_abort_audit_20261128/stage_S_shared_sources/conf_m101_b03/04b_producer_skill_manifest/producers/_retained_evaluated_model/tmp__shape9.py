exec(open('/tmp/readout7.py').read().split('folds=')[0])
from scipy.linalg import orthogonal_procrustes
for key in ('v1','v2'):
 V=C[key]; func=[lambda h,k:np.tanh(k*h),lambda h,k:np.arctan(k*h),lambda h,k:h/np.hypot(1,k*h),lambda h,k:np.sin(k*h),lambda h,k:np.hypot(1,k*h)-1,lambda h,k:np.log1p((k*h)**2)]
 for j,fun in enumerate(func):
  for linear in (False,True):
   def design(k): return np.c_[aff(u) if linear else np.ones(len(u)),fun(u,k)]
   def err(k):
    H=design(k[0]);return (H@ls(H,V)-V).ravel()
   opt=least_squares(err,[1.1],bounds=(.1,6)); coef=ls(design(opt.x[0]),V); sv=np.linalg.svd(coef[-4:],compute_uv=False)
   print(key,j,linear,opt.x.round(4),round(np.std(opt.fun),7),sv.round(4))
print('B',b.round(4));
for name,H in G.items():
 c0,sd=H.mean(0),H.std(0);H=(H-c0)/sd
 for m in ('sq','tanh','cos','rat','sin'):
  fun={'sq':lambda h:h*h,'tanh':np.tanh,'cos':np.cos,'rat':lambda h:1/(1+h*h),'sin':np.sin}[m]
  def fit(Y,j):
   def design(p):return np.c_[aff(H),*(fun(H@p[k:k+4]+p[k+4]) for k in range(0,len(p),5))]
   def res(p):
    d=design(p);return d@ls(d,Y)-Y
   rng=np.random.default_rng(34)
   return min((np.std(least_squares(res,rng.normal(size=5*j)*.65,max_nfev=75).fun) for _ in range(5)))
  if name in ('D','DB','a'): print(name,m,[round(fit(Y[:,j],2),5) for j in range(3)])
