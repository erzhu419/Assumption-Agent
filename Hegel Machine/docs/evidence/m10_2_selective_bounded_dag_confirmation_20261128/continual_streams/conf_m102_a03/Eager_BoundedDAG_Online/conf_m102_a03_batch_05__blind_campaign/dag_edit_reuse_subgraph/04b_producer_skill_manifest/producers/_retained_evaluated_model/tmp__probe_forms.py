exec(open('/tmp/probe_dag.py').read().split('for d in (1,2,3,4):')[0])
from scipy.optimize import minimize, minimize_scalar, least_squares
q=s['panel_c']; lin=terms(1); cr=list(it.combinations(range(4),2)); diag3=[(j,j,j) for j in range(4)];
F={'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'rat':lambda x,k:x/(1+k*x*x),'cube_rat':lambda x,k:x**3/(1+k*x*x),'atan':lambda x,k:np.arctan(k*x),'erf':lambda x,k:__import__('scipy').special.erf(k*x),'sinh':lambda x,k:np.sinh(k*x),'cos':lambda x,k:np.cos(k*x),'cubic':lambda x,k:x**3,'pow':lambda x,k:np.sign(x)*np.abs(x)**k,'log':lambda x,k:np.sign(x)*np.log1p(k*abs(x))}
def opt(h,X,Y,base,split=None,shared=True,inter=False):
 ids=np.arange(len(X)); i,j=(ids[:split],ids[split:]) if split else (ids,ids); d=h(X,1).shape[1];
 def design(X,k):
  z=h(X,k); return np.c_[mat(X,base),z,mat(z,cr)] if inter else np.c_[mat(X,base),z]
 def solve(k):
  r=design(X[i],k); c=n._fit(r,Y[i]); return np.mean((r@c-Y[i])**2),c
 grid=np.geomspace(.12,8,70); errs=np.array([solve(k)[0] for k in grid]); k=grid[errs.argmin()]; l=minimize_scalar(lambda k:solve(k)[0],bounds=(k/1.2,k*1.2),method='bounded'); k=l.x
 if not shared:
  l=minimize(lambda k:solve(k)[0],np.repeat(k,d),method='L-BFGS-B',bounds=[(.12,8)]*d,options={'ftol':1e-14,'gtol':1e-10}); k=l.x
 e=design(X[j],k)@solve(k)[1]-Y[j]; return np.mean(abs(e)),np.round(k,4),np.sqrt(solve(k)[0])
for desc,X,Y,split in [('B',u,v,18),('C1',q['u'],q['v1'],18),('C2',q['u'],q['v2'],18),('scalar',x,y,48),('amp',x,yy,48)]:
 print('\n',desc)
 print('additive cube',scores(mat(X,lin+diag3),Y,split)); print('cube crossed',scores(mat(X,lin+cr+diag3),Y,split))
 for basis in [lin,lin+cr]:
  out=[]
  for key,h in F.items():
   for shared in [True,False] if desc in ('scalar','amp') else [True]:
    for inter in [False,True] if desc=='scalar' and len(basis)==len(lin) else [False]:
     r=opt(h,X,Y,basis,split,shared,inter);out.append((r[0],key,shared,inter,r[1],r[2]))
  print(len(basis), sorted(out,key=lambda r:r[0])[:10])
print('B vs C1 pooled affine cubic', scores(mat(np.r_[u,q['u']],lin+diag3),np.r_[v,q['v1']],24)); print('B C1 aff cub full',n._fit(mat(u,lin+diag3),v).round(4), n._fit(mat(q['u'],lin+diag3),q['v1']).round(4))
