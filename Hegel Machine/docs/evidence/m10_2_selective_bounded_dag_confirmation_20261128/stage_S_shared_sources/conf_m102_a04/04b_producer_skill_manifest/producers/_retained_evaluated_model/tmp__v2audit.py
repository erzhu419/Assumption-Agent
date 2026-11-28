import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
sp=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; t=s['eval_times']; p=s['panel_c'];u=p['u'];v=p['v1']; N=len(tr['x']); Y=np.c_[tr['y'],tr['traj']]; a,b=m._align(s['panel_a']); X=(tr['x']-a)/b

def soft(x,k):return x/(1+k*abs(x))
def cfit(I,mode):
 def F(x,q):
  if mode=='tied':return m.one(soft(x,q[0])+q[1]*x)
  if mode=='untied':return np.c_[m.one(x),soft(x,q[0])]
  return m.one(soft(x,q[0]))
 def r(q):
  f=F(u[I],q);return (f@m.lin(f,v[I])-v[I]).ravel()
 q=least_squares(r,[.85,-.01] if mode=='tied' else [.85],bounds=([.02,-1],[10,1]) if mode=='tied' else (.02,10)).x
 B=m.lin(F(u[I],q),v[I]);return lambda x:F(x,q)@B,q
from itertools import combinations
v2=p['v2']; inds=np.arange(len(u)); panels=[np.array_split(np.random.default_rng(seed).permutation(inds),4) for seed in (42,17)]; c,_=cfit(inds,'tied')
def sq(w):return w*w
def cross(w):return np.column_stack([w[:,i]*w[:,j] for i,j in combinations(range(4),2)])
def f(w,typ,k):
    base=m._square(w)
    if typ=='square':return base
    if typ=='transformed_square':return np.c_[m.one(w),w*w/(1+k*w*w)]
    if typ=='even_softsign':return np.c_[m.one(w),w*w/(1+k*abs(w))]
    if typ=='rational+square':return np.c_[base,w*w/(1+k*w*w)]
    if typ=='softsign+square':return np.c_[base,soft(w,k)]
    if typ=='cube':return np.c_[base,w**3]
    if typ=='cross':return np.c_[base,cross(w)]
    if typ=='c1':return np.c_[base,c(w)]
    if typ=='even_softsign_squared':return np.c_[m.one(w),soft(w,k)**2]
    raise ValueError(typ)
print('square coefficients',m.lin(m._square(u),v2).round(5)); print('square residual singular values',np.linalg.svd(v2-m._square(u)@m.lin(m._square(u),v2),compute_uv=False)); rows=[]
for typ in ('square','transformed_square','even_softsign','rational+square','softsign+square','cube','cross','c1','even_softsign_squared'):
 for k in ((.1,.25,.5,.84,1.5) if ('softsign' in typ or 'rational' in typ or typ=='transformed_square') else (.84,)):
    F=f(u,typ,k); E=[]
    for folds in panels:
      E.append(np.sqrt(np.mean(np.vstack([F[j]@m.lin(F[np.setdiff1d(inds,j)],v2[np.setdiff1d(inds,j)])-v2[j] for j in folds])**2)))
    rows.append((max(E),typ,k,E,np.std(F@m.lin(F,v2)-v2)))
print(*sorted(rows)[:16],sep='\n')
