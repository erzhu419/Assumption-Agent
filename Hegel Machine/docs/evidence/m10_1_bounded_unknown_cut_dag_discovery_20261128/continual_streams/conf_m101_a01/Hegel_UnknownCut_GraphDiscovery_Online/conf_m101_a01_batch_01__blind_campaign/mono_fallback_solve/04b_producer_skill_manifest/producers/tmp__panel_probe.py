from graph_probe import *
from scipy.optimize import minimize_scalar, least_squares
p=s['panel_c']; u=p['u']; m=len(u)
np.set_printoptions(precision=5,suppress=True)
def fun(q,f,fr):
 if f=='sin': return np.sin(q*fr)
 if f=='cos': return np.cos(q*fr)
 if f=='tanh': return np.tanh(q*fr)
 if f=='exp': return np.exp(q*fr)
 if f=='rat':return 1/(1+(q*fr)**2)
 if f=='sinphase':return np.c_[np.sin(q*fr),np.cos(q*fr)]
 if f=='poly':return np.c_[q*q,q**3]
def fit(k,ix):
 v=p[k]; results=[]
 for f in ['sin','cos','tanh','exp','rat','sinphase','poly']:
  for base in [0,1,2]:
   if base==2 and f in ['poly','sinphase']:continue
   def des(q,fr):return np.c_[poly(q,base),fun(q,f,fr)]
   def residual(fr):
    d=des(u[ix],fr); return (v[ix]-d@np.linalg.lstsq(d,v[ix],rcond=1e-9)[0]).ravel()
   raw=[]
   for fr0 in [.6,1.3,2.1,3.5]:
    opt=least_squares(residual,[fr0],bounds=(.15,6.),max_nfev=70); raw.append(opt)
   c=min(raw,key=lambda r:r.cost); fr=c.x
   for mode in [0,1]:
    if mode:
     opt=least_squares(residual,np.repeat(c.x,4),bounds=(.15,6.),max_nfev=90);fr=opt.x
    d=des(u[ix],fr); coef=np.linalg.lstsq(d,v[ix],rcond=1e-9)[0]; other=np.setdiff1d(np.arange(m),ix); error=np.mean(abs(des(u[other],fr)@coef-v[other])) if len(other) else 0
    results.append((np.mean(abs(residual(fr))),error,f,base,mode,fr))
 return sorted(results,key=lambda r:r[0] if len(ix)==m else r[1])
if __name__=='__main__':
 for k in ['v1','v2']:
  for ix in [np.arange(m),np.random.default_rng(3).permutation(m)[:19]]:
   print(k,len(ix),fit(k,ix)[:10]);
