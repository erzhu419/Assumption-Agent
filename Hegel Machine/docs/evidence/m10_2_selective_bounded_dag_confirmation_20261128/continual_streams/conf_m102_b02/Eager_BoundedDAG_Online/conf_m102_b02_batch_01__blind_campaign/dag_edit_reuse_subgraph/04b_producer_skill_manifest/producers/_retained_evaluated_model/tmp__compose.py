exec(open('/tmp/probe.py').read().split("print('raw',score(x))")[0])
from scipy.optimize import minimize_scalar,least_squares
bank={}
for p,k,fams in [('panel_b','v',['sin','tanh','odd','asinh']),('panel_c','v1',['sin','tanh','odd','asinh']),('panel_c','v2',['even','sqcos'])]:
 u,v=s[p]['u'],s[p][k]
 for fam in fams:
  fun={'sin':np.sin,'tanh':np.tanh,'odd':lambda q:q/(1+q*q),'asinh':np.arcsinh,'even':lambda q:1/(1+q*q),'sqcos':np.cos}[fam]
  def design(w,a): return np.c_[np.ones(len(w)),w,w*w if fam=='sqcos' else np.empty((len(w),0)),fun(a*w)]
  def objective(a):
   d=design(u,a); c=np.linalg.lstsq(d,v,rcond=None)[0]; return np.mean((d@c-v)**2)
  g=np.linspace(.12,10,240); j=np.argmin([objective(a) for a in g]); o=minimize_scalar(objective,bounds=(g[max(0,j-1)],g[min(len(g)-1,j+1)]),method='bounded'); d=design(u,o.x); c=np.linalg.lstsq(d,v,rcond=None)[0]; print(p,k,fam,o.x,o.fun**.5,'coef',c.tolist()); bank[k+fam]=(lambda w,d=design,a=o.x,c=c:d(w,a)@c)
  # Bind the design's function and basis type before advancing the loop.
  bank[k+fam]=lambda w,f=fun,a=o.x,c=c,sq=fam=='sqcos':np.c_[np.ones(len(w)),w,w*w if sq else np.empty((len(w),0)),f(a*w)]@c
 for deg in (1,):
  c=np.linalg.lstsq(poly(u,deg),v,rcond=None)[0];bank[k+str(deg)]=lambda w,c=c,deg=deg:poly(w,deg)@c
bank['raw']=lambda w:w
results=[]
for i,f in bank.items():
 for j,g in bank.items():
  for scale in (.125,.25,.5,1,2):
   z=g(f(x))*scale; r=score(z);results.append((max(r[2:]),i,j,scale,r))
for i,f in bank.items():
 for j,g in bank.items():
  for a in (-2,-1,-.5,-.25,.25,.5,1,2):
   r=score(f(x)+a*g(x));results.append((max(r[2:]),i,'plus'+j,a,r))
for r in sorted(results,key=lambda v:v[0])[:30]: print('BEST',r)
