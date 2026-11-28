from dagprobe import *
from scipy.optimize import minimize
x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; n=len(x); d=blocks(x,s); Q=poly(x,2); t=s['eval_times']; inds=np.arange(n); splits=[(inds[:52],inds[52:])]+[(np.setdiff1d(inds,j),j) for j in np.array_split(np.random.default_rng(13).permutation(n),4)]
for i,j in splits[:1]:
 def rr(r):
  F=-np.expm1(-t[:,None]*r);return (T[i].T-F@ls(F,T[i].T)).ravel()
 fit=least_squares(rr,[1.4,.25],bounds=([.6,.01],[5,.8]));print('rates',fit.x,'noise',np.std(fit.fun)); print('summary',s['train']['summary']);print('Q y',ls(Q[i],y[i]));print('Q traj',ls(Q[i],T[i,-1]));print('r y precision',np.quantile(abs(y[i]-Q[i]@ls(Q[i],y[i])),[0,.25,.5,.75,.9,1]))
rows=[]
for key,z in d.items():
 for kind in ['rat','multiply','add','divide','tanh','sin','abs','exp']:
  for deg in ([1,2] if kind in ['rat','multiply','divide'] else [2]):
   for k in ([1,2] if kind=='add' else [1]):
    def design(zz,p):
     if kind=='add': return np.c_[Q, np.tanh(zz@p.reshape(5,k))]
     a=zz@p; h=1+a*a
     if kind=='rat':return poly(x,deg)/h[:,None]
     if kind=='divide':return np.c_[Q,poly(x,deg)/h[:,None]]
     if kind=='multiply':return np.c_[Q,poly(x,deg)*np.tanh(a[:,None])]
     f={'tanh':np.tanh,'sin':np.sin,'abs':np.abs,'exp':lambda a:np.exp(np.clip(a,-8,5))}[kind]
     return np.c_[Q,f(a)]
    zz=aff(z); p0=np.r_[np.c_[np.zeros(4),np.eye(4)],np.random.default_rng(15).normal(0,.65,(4,5))]; pp=[]
    for i,j in splits:
     def r(p):
      F=design(zz,p)[i]; return y[i]-F@ls(F,y[i])
     fits=[least_squares(r,np.tile(p,k),bounds=(-4,4),max_nfev=70) for p in p0]; ft=min(fits,key=lambda p:p.cost); F=design(zz,ft.x); ep=y[j]-F[j]@ls(F[i],y[i]); pp.append((abs(ep).mean(),np.std(ft.fun),ft.x.round(2)))
    rows.append((key,kind,deg,k,np.array([p[0] for p in pp]),pp[0][1],pp[0][2]))
for nm,fun in [('gate',lambda r:r[4][0]),('worst',lambda r:max(r[4])),('fit',lambda r:r[5])]:
 print(nm,flush=True)
 for r in sorted(rows,key=fun)[:12]:print(*r,flush=True)
