exec(open('/tmp/probe.py').read().split("print('raw',score(x))")[0])
from scipy.optimize import least_squares
from itertools import combinations
from scipy.linalg import solve

def inspect(p,k):
 u,v=s[p]['u'],s[p][k]; out=[]
 funcs={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'odd':lambda q:q/(1+q*q),'sat':lambda q:q/np.sqrt(1+q*q),'softplus':lambda q:q/(1+abs(q)), 'even':lambda q:1/(1+q*q),'cos':np.cos}
 for name,f in funcs.items():
  for extras in (0,1,2):
   def D(w,a): return np.c_[np.ones(len(w)),w,f(w*a),*[w**i for i in ((3,) if extras==1 else (2,) if extras==2 else ())]]
   def obj(a):
    d=D(u,a); return (d@np.linalg.lstsq(d,v,rcond=None)[0]-v).ravel()
   o=min((least_squares(obj,np.full(4,g),bounds=(.06,9),max_nfev=65) for g in (.5,1.25,2.5,3.5)),key=lambda q:q.cost)
   d=D(u,o.x); c=np.linalg.lstsq(d,v,rcond=None)[0]; e=obj(o.x).reshape(v.shape); lea=np.diag(d@np.linalg.pinv(d)); rms=np.std(e); loo=np.std(e/(1-lea)[:,None]); out.append((loo,rms,name,extras,o.x,c))
 for loo,rms,name,e,a,c in sorted(out,key=lambda q:q[0])[:4]:
  print('PANEL',p,k,name,e,'rms,loo',rms,loo,'params',a,'coeffs',c.round(3).tolist()); f=funcs[name];
  print('scores',[(q,score(z).round(3).tolist()) for q,z in [('linear',x@c[1:5]),('nonlinear',f(a*x)@c[5:9]),('axis',f(a*x)),('full',np.c_[np.ones(len(x)),x,f(a*x),*[x**i for i in ((3,) if e==1 else (2,) if e==2 else ())]]@c)]])
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]: inspect(p,k)
print('B n u ranges',s['panel_b']['u'].min(0),s['panel_b']['u'].max(0))
# Exact-feature probes in latent space: joint odd/even axes, rotated pairwise forms, and trigonometric products.
from scipy.linalg import qr
for name,axes in [('raw',x),('b_axes', x@np.linalg.lstsq(np.c_[np.ones(24),s['panel_b']['u']],s['panel_b']['v'],rcond=None)[0][1:])]:
 for f,grid in [('tanh',(.5,1.25,2.5,3.5)),('asinh',(.5,1.25,2.5,3.5,6)),('sin',(.5,1.25,2.5,3.5)),('odd',(.5,1.25,2.5,3.5)),('sinh',(.5,1.25,2.5))]:
  for a in grid:
   h={'tanh':np.tanh,'asinh':np.arcsinh,'sin':np.sin,'odd':lambda q:q/(1+q*q),'sinh':np.sinh}[f](axes*a)
   for deg in (1,2,3):
    D=poly(h,deg); R=[]
    for j in folds:
     i=np.setdiff1d(np.arange(n),j); c=np.linalg.lstsq(D[i],tar[i],rcond=None)[0]; r=abs(D[j]@c-tar[j]); R.append([r[:,0].mean(),r[:,1:].mean()])
    if max(R[0])<.08 or max(np.max(R,axis=0))<.11: print('RESPONSE',name,f,a,deg,np.array(R).round(4).tolist())
