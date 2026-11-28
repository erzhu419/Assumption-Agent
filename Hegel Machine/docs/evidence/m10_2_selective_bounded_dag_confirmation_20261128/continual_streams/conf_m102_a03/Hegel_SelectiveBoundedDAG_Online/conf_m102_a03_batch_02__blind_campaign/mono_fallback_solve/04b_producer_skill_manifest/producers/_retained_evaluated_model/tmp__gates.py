exec(open('/tmp/probe.py').read().split("print('times'")[0])
from scipy.optimize import least_squares
from scipy.special import expit
z=(x-a[0])/a[1]; L=A(z); rng=np.random.default_rng(741); I=np.arange(52); J=np.arange(52,72)
for mode in ('gate','rat','sin','relu','gateq','ratq'):
 def features(h):
  g=L@h[:5]; base=L
  if mode.startswith('gate'): f=expit(g)
  elif mode.startswith('rat'): f=1/(1+g*g)
  elif mode=='sin': f=np.sin(g)
  else: f=np.maximum(g,0)
  if mode.endswith('q'): base=P(z,2)
  return np.c_[base,L*f[:,None]]
 def err(h):
  f=features(h)[I]; return f@fit(f,y[I])-y[I]
 starts=[np.r_[v,rng.normal(size=4)*k] for k in (.5,1.5,3) for v in rng.normal(size=9)]
 sols=[least_squares(err,h,bounds=(-14,14),max_nfev=110) for h in starts]; best=min(sols,key=lambda r:r.cost); f=features(best.x); beta=fit(f[I],y[I]); e=f@beta-y; print(mode, np.mean(abs(e[I])),np.mean(abs(e[J])),best.x,beta,flush=True)
r=y-L@fit(L[I],y[I]); print('residual extreme rows:'); k=np.argsort(abs(r))[-14:]; print(np.c_[z[k],y[k],r[k],t[k,-1]])
