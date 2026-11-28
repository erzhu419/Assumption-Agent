from probe import *
from scipy.special import erfi
p=s['panel_b']; u,v=p['u'],p['v']; base=aff(u)
ff=dict(funcs, gaussodd=lambda x,k:x*np.exp(-k*x*x),xrat2=lambda x,k:x/(1+k*x*x)**2,logistic=lambda x,k:np.tanh(k*x),erfi=lambda x,k:erfi(k*x),sinh=lambda x,k:np.sinh(k*x))
for name,g in ff.items():
 def d(p,x=u):return np.c_[aff(x),g(x+p[4:],np.exp(p[:4]))]
 def residual(p):
  a=d(p);return (a@fit(a,v)-v).ravel()
 cap=1 if name in ('erfi','sinh') else 2
 best=min((least_squares(residual,np.r_[np.full(4,k),np.zeros(4)],bounds=(np.r_[np.full(4,-4),np.full(4,-.8)],np.r_[np.full(4,cap),np.full(4,.8)]),max_nfev=125) for k in [-1,0,.7]),key=lambda f:f.cost)
 print(name,round(np.std(best.fun),7), np.round(np.exp(best.x[:4]),4),np.round(best.x[4:],4))
 if np.std(best.fun)<.001: print(np.round(fit(d(best.x),v),4))
