import itertools as it, pickle, numpy as np
from scipy.optimize import minimize_scalar, least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; p=s['panel_c']; u=p['u']; y=p['v2']; a=np.c_[np.ones(len(u)),u]; Q=np.linalg.qr(a)[0]; center=lambda z:z-Q@(Q.T@z); R=center(y)
print('Residual singular values',np.linalg.svd(R,compute_uv=False)); print('Affine v2',np.linalg.lstsq(a,y,rcond=None)[0]); print('Panel B crosscheck',[(k,np.linalg.norm(v,axis=1).mean()) for k,v in s['panel_b'].items()]); print('Panel A differences',np.sort(np.linalg.norm(s['panel_a']['v']-s['panel_a']['u'],axis=1))[-10:])
bases={}; pairs=list(it.combinations(range(4),2)); pair=np.column_stack([u[:,i]*u[:,j] for i,j in pairs]); bases['pairs']=pair; bases['triples']=np.column_stack([np.prod(u[:,c],axis=1) for c in it.combinations(range(4),3)]); bases['u']=u
for shift in (1,2,3):
 v=np.roll(u,-shift,axis=1); bases[f'prod{shift}']=u*v
 for c in (-2,-1.5,-1,-.5,.5,1,1.5,2): bases[f'sum{shift},{c}']=u+c*v
for c in (-1,-.5,.5,1): bases[f'shift{c}']=u+c
for perm in it.permutations(range(4)):
 if all(i!=j for i,j in enumerate(perm)): bases[f'perm{perm}']=u*u[:,perm]
funcs={'identity':(lambda z,k:z,0,1),'square':(lambda z,k:z*z,0,1),'cube':(lambda z,k:z**3,0,1),'sat':(lambda z,k:z/(1+k*z*z),.001,8),'satabs':(lambda z,k:z/(1+k*abs(z)),.001,8),'rat':(lambda z,k:z*z/(1+k*z*z),.001,8),'sin':(lambda z,k:np.sin(k*z),.03,7),'cos':(lambda z,k:np.cos(k*z),.03,7),'tanh':(lambda z,k:np.tanh(k*z),.03,7),'log':(lambda z,k:np.log1p(k*z*z),.001,8)}
def score(z,cv=False):
 q,d,_=np.linalg.svd(center(z),full_matrices=False); q=q[:,d>max(d.max(),1e-12)*1e-10]; r=R-q@(q.T@R); h=1-(Q*Q).sum(1)-(q*q).sum(1); return np.mean((r/h[:,None] if cv else r)**2)**.5
results=[]
for name,b in bases.items():
 for fn,(f,l,h) in funcs.items():
  obj=lambda k:score(f(b,k)); g=np.linspace(l,h,35); i=np.argmin([obj(k) for k in g]); k=minimize_scalar(obj,bounds=(g[max(0,i-1)],g[min(len(g)-1,i+1)]),method='bounded').x
  results.append((obj(k),score(f(b,k),True),name,fn,k))
print('Candidate operators (fit RMSE, fixed-parameter PRESS RMSE):'); print(*sorted(results)[:27],sep='\n')
print('Cyclic input-shift/coefficient tests:')
fits=[]
for j in (1,2,3):
 v=np.roll(u,-j,axis=1)
 for name,fn,n in [('quad',lambda k:u*u+k[0]*u*v+k[1]*v*v,2),('paired_rat',lambda k:(u*v)/(1+k[0]*u*u+k[1]*v*v),2),('coupled_rat',lambda k:(u*u+k[0]*u*v+k[1]*v*v)/(1+k[2]*u*u+k[3]*v*v),4),('sum_cos',lambda k:np.cos(k[1]*(u+k[0]*v)+k[2]),3),('prod_cos',lambda k:np.cos(k[0]*u*v+k[1]),2),('prod_tanh',lambda k:np.tanh(k[0]*(u+k[1])*(v+k[2])+k[3]),4),('sum_rat',lambda k:(u+k[0]*v+k[1])**2/(1+k[2]*(u+k[0]*v+k[1])**2),3),('prod_and_square',lambda k:u*v+k[0]*u*u,1)]:
  from scipy.optimize import minimize
  rng=np.random.default_rng(27)
  bounds=[(-4,4)]*n
  if name=='paired_rat': bounds=[(0,5)]*2
  if name=='coupled_rat': bounds=[(-4,4)]*2+[(0,5)]*2
  if name=='sum_rat': bounds=[(-4,4)]*2+[(0,5)]
  opt=min((minimize(lambda k:score(fn(k))**2,np.array([rng.uniform(*lim) for lim in bounds]),bounds=bounds,method='L-BFGS-B',options={'ftol':1e-13}) for _ in range(10)),key=lambda o:o.fun)
  fits.append((opt.fun**.5,score(fn(opt.x),True),j,name,opt.x))
print(*sorted(fits)[:20],sep='\n')
