import pickle
import time
import numpy as np
from scipy.optimize import linprog

with open('/holdout_task/public_facility.pkl', 'rb') as stream:
    p = pickle.load(stream)
f, a, d, c = (p[k] for k in ('fixed_cost', 'capacity', 'demand', 'cost_matrix'))
m,n=c.shape
bits=((np.arange(1<<m)[:,None] >> np.arange(m))&1).astype(float)
feasible=bits@a>=sum(d)
fixed=bits@f
E=np.tile(np.eye(n),(1,m)); A=np.kron(np.eye(m),np.ones((1,n)))

def lp(mask):
    rows=np.flatnonzero(bits[mask]); cols=(rows[:,None]*n+np.arange(n)).ravel()
    sol=linprog(c[rows].ravel(), A_ub=A[rows][:,cols],b_ub=a[rows],A_eq=E[:,cols],b_eq=d,method='highs-ds')
    assert sol.success,sol.message
    return sol

def scores(price):
    q=np.maximum(0,(price-c).max(axis=1))
    return price@d + bits@(f-a*q)

start=time.perf_counter()
true=np.full(1<<m,np.inf); duals=[]
for mask in np.flatnonzero(feasible):
    sol=lp(mask); true[mask]=sol.fun+fixed[mask]
    duals.append(sol.eqlin.marginals)
opt=true.min(); best=true.argmin()
print('Feasible supports:',feasible.sum(),'enumerated objective:',opt,'optimal mask:',best,'all-LP time:',time.perf_counter()-start,flush=True)
B=np.full(1<<m,-np.inf); B[~feasible]=np.inf
B=np.maximum(B,scores(c.min(axis=0)))
ub=np.inf; count=0; one_cut_root=-np.inf
solved=np.zeros(len(B),bool)
while True:
    i=int(np.argmin(np.where(solved,np.inf,B)))
    if B[i]>=ub-1e-8: break
    sol=lp(i); count+=1; solved[i]=True; ub=min(ub,sol.fun+fixed[i]); s=scores(sol.eqlin.marginals)
    assert abs(s[i]-sol.fun-fixed[i])<1e-7
    B=np.maximum(B,s); one_cut_root=max(one_cut_root,s[feasible].min())
print('Adaptive LPs:',count,'adaptive objective error:',ub-opt,'minimum dual portfolio margin:',B[feasible].min()-opt,'best single-price root gap:',opt-one_cut_root,flush=True)
assert abs(ub-opt)<1e-8 and np.max(B[feasible]-true[feasible])<1e-8
