import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from branch6 import *
from scipy.special import expit
for name,F in {'sqrt':lambda u,a,b:u+a*(np.sqrt(1+b*u*u)-1),'cosh':lambda u,a,b:u+a*np.log(np.cosh(b*u)),'atan':lambda u,a,b:u+a*u*np.arctan(b*u),'rat':lambda u,a,b:u+a*u*u/(1+b*u*u),'abssoft':lambda u,a,b:u+a*np.log1p(b*u*u)}.items():
 pp=least_squares(residual,[1.,1.],args=(F,)).x; print(name,pp,np.std(residual(pp,F)))
F=lambda u,a,b:u+a*(np.sqrt(1+b*u*u)-1); pp=least_squares(residual,[1.,1.],args=(F,)).x
r=A(F(w,*pp))@L(A(F(cu,*pp)),V)
# Profile low-dimensional multiplicative gates. Fit only to the first 52 rows.
U,S,VT=np.linalg.svd(Y[:52,1:],full_matrices=False); target=np.c_[Y[:,0],Y[:,1:]@VT[0]]; target/=target[:52].std(0)
for kn,k in [('r',r),('w',w),('q',q),('zr',np.c_[z,r]),('qr',np.c_[q,r]),('wr',np.c_[w,r])]:
 for gn,gf in [('sig',lambda a:2*expit(a)),('rat',lambda a:1/(1+a*a)),('tanh',lambda a:1+np.tanh(a)),('relu',lambda a:np.maximum(0,1+a)),('exp',lambda a:np.exp(np.clip(a,-12,12)))]:
  for mode in ('mod','offset','mix'):
   def des(p):
    g=gf(A(k)@p); h=q*g[:,None]; d=A(h)
    if mode=='offset':d=np.c_[d,q]
    if mode=='mix':d=np.c_[d,r*(1-g[:,None])]
    return d
   def res(p):
    d=des(p)[:52];return np.r_[((d@L(d,target[:52])-target[:52])).ravel(),.003*p]
   rng=np.random.default_rng(33); sols=[least_squares(res,p,max_nfev=100) for p in [np.zeros(k.shape[1]+1),rng.normal(0,.5,k.shape[1]+1),rng.normal(0,1,k.shape[1]+1)]]; sol=min(sols,key=lambda f:f.cost)
   a=score(des(sol.x));
   if min(a[2:])<.041 or kn in ('r','w') and mode=='mod': print(kn,gn,mode,a,sol.x)
