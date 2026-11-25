import pickle
import sys
import time
import numpy as np
from scipy.optimize import linprog
sys.path.insert(0, '/holdout_task')
from facility_cert import solve_and_certify_facilities

with open('/holdout_task/public_facility.pkl','rb') as stream:
    public = pickle.load(stream)
rng = np.random.default_rng(8081)
m,n=12,24
bits=((np.arange(1<<m)[:,None] >> np.arange(m))&1).astype(int)

def instance(f,a,d,c):
    return dict(fixed_cost=np.asarray(f,float),capacity=np.asarray(a,float),demand=np.asarray(d,float),cost_matrix=np.asarray(c,float))

def exhaustive(p):
    f,a,d,c=(p[k] for k in ('fixed_cost','capacity','demand','cost_matrix'))
    obj=np.full(len(bits),np.inf)
    for mask in np.flatnonzero(bits @ a >= d.sum()-1e-12):
        rows=np.flatnonzero(bits[mask]); k=len(rows)
        res=linprog(c[rows].ravel(), A_ub=np.kron(np.eye(k),np.ones((1,n))), b_ub=a[rows], A_eq=np.tile(np.eye(n),(1,k)), b_eq=d, method='highs-ipm', options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        if res.success:
            obj[mask]=f[rows].sum()+res.fun
        elif res.status!=2:
            raise AssertionError((mask,res.message))
    return obj

c=rng.uniform(.05,18,(m,n)); a=rng.integers(6,15,m); f=rng.uniform(2,35,m)
cluster=np.where(np.arange(m)[:,None]//3==np.arange(n)[None,:]//6, .2, 15)+rng.uniform(0,1,(m,n))
cases={
 'public':public,
 'zero_shipping_tied_openings':instance(np.full(m,5),np.full(m,6),np.full(n,1.25),np.zeros((m,n))),
 'exact_total_capacity':instance(f,a,np.full(n,a.sum()/32)+np.r_[np.ones(8)*a.sum()/32,np.zeros(16)],c),
 'near_capacity_critical_subset':instance(f,np.r_[np.full(6,6.),np.full(5,7.),6.-2**-36],np.full(n,1.5),c),
 'clustered_capacity_spill':instance(f,np.full(m,4.5),rng.uniform(1.1,1.9,n),cluster),
 'heterogeneous_units_and_costs':instance(f*1000,np.geomspace(.4,28,m),rng.uniform(.25,1.75,n),c*10000),
 'dense_high_fixed_cost':instance(f*20,a,rng.uniform(.7,2.8,n),c),
 'permuted_public':instance(public['fixed_cost'][::-1],public['capacity'][::-1],public['demand'][::-1],public['cost_matrix'][::-1,::-1]),
}
failed=[]
for label,p in cases.items():
    try:
        t=time.perf_counter(); r=solve_and_certify_facilities(p); runtime=time.perf_counter()-t
        exact=exhaustive(p)
        y=np.array(r['open_y']); x=r['flows_X']; ub=r['upper_bound']; leaves=r['proof_leaves']
        assert y.shape==(m,) and set(y)<={0,1} and x.shape==(m,n) and x.min()>=-1e-10
        primal=max(np.abs(x.sum(axis=0)-p['demand']).max(),(x.sum(axis=1)-p['capacity']*y).max())
        error=abs(ub-exact.min()); assert error<=1e-7,(ub,exact.min())
        assert primal<=1e-9,primal
        assert abs(ub-(p['fixed_cost']@y+np.sum(x*p['cost_matrix'])))<=1e-7
        covered=np.zeros(len(bits),int); gap=0.; overshoot=0.
        for leaf in leaves:
            z,o,b=(leaf[k] for k in ('fixed_closed','fixed_open','lower_bound'))
            selected=(bits[:,z]==0).all(axis=1)&(bits[:,o]==1).all(axis=1)
            assert selected.any(); covered[selected]+=1
            ref=min(exact[selected]); assert b>=ub-1e-5 and b<=ref+1e-7,(b,ref)
            gap=max(gap,ub-b); overshoot=max(overshoot,b-ref)
        assert (covered==1).all() and 1<=len(leaves)<=80
        print(label, 'error %.3g gap %.3g over %.3g flow %.3g leaves %d time %.3fs'%(error,gap,overshoot,primal,len(leaves),runtime),flush=True)
    except Exception as exc:
        failed.append(label); print(label,'FAILED',repr(exc),flush=True)
assert not failed,failed
