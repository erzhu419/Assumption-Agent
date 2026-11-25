import pickle, time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

class Model:
    def __init__(self, obj):
        self.f, self.u, self.d, self.c = (np.asarray(obj[k], dtype=float) for k in ('fixed_cost','capacity','demand','cost_matrix'))
        self.m, self.n = m, n = self.c.shape
        i = np.repeat(np.arange(m), n)
        j = np.tile(np.arange(n), m)
        x = m + np.arange(m*n)
        shape = (n,m+m*n)
        self.eq = coo_matrix((np.ones(m*n),(j,x)), shape=shape).tocsr()
        shape = (m+m*n,m+m*n)
        self.ub = coo_matrix((np.r_[-self.u,np.ones(2*m*n),-self.d[j]], (np.r_[np.arange(m),i,m+np.arange(m*n),m+np.arange(m*n)],np.r_[np.arange(m),x,x,i])),shape=shape).tocsr()
        self.obj = np.r_[self.f,self.c.ravel()]
        self.zero = np.zeros(shape[0])
        self.ids = np.arange(1<<m)
        self.bits = 1<<np.arange(m)
        self.Y = ((self.ids[:,None]&self.bits)!=0).astype(float)
        ratios = [float(a).as_integer_ratio() for a in np.r_[self.u, self.d]]
        den = max(b for a,b in ratios)
        weights = [a*(den//b) for a,b in ratios]
        cap = [0]*(1<<m)
        for s in range(1,1<<m):
            bit=s&-s
            cap[s]=cap[s^bit]+weights[bit.bit_length()-1]
        demand = sum(weights[m:])
        self.feasible = np.array([v>=demand for v in cap])
        self.best_per_mask = np.full(len(self.ids), -np.inf)

    def add_price(self, p):
        gain = np.maximum(p-self.c,0.)
        order = np.argsort(-gain, axis=1)
        g = np.take_along_axis(gain,order,axis=1)
        d = self.d[order]
        amount = np.maximum(0,np.minimum(d,self.u[:,None]-np.cumsum(d,axis=1)+d))
        benefit = np.sum(amount*g,axis=1)
        base = self.d@p
        bound = base+self.Y@(self.f-benefit)
        guard=2e-12*(1+abs(self.d)@abs(p)+np.sum(self.f+benefit))
        np.maximum(self.best_per_mask,bound-guard,out=self.best_per_mask)

    def incumbent(self):
        m,n = self.m,self.n
        res = milp(self.obj, integrality=np.r_[np.ones(m),np.zeros(m*n)], bounds=Bounds(0,np.r_[np.ones(m),np.full(m*n,np.inf)]),constraints=LinearConstraint(vstack([self.eq,self.ub]),np.r_[self.d,np.full(len(self.zero),-np.inf)],np.r_[self.d,self.zero]),options={'mip_rel_gap':1e-10})
        if res.x is None:
            raise RuntimeError(res.message)
        y = (res.x[:m]>.5).astype(int)
        transport = linprog(self.c.ravel(), A_eq=self.eq[:,m:],b_eq=self.d,A_ub=self.ub[:m,m:],b_ub=self.u*y, method='highs', options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        assert transport.success,transport.message
        self.add_price(transport.eqlin.marginals)
        return y,transport.x.reshape(m,n),float(self.f@y+transport.fun)

    def relaxation(self, o, c):
        bounds=list(zip(((o&self.bits)!=0).astype(int),((c&self.bits)==0).astype(int)))+[(0,None)]*(self.m*self.n)
        return linprog(self.obj,A_eq=self.eq,b_eq=self.d,A_ub=self.ub,b_ub=self.zero,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})

    def run(self):
        y,x,ub = self.incumbent()
        root=self.relaxation(0,0)
        self.add_price(root.eqlin.marginals)
        root_lb = self.best_per_mask[self.feasible].min()
        print('incumbent',ub,'root LP',root.fun,'root price knapsack',root_lb,flush=True)
        count=1
        def tree(o,c, ids, res=None):
            nonlocal count
            if not len(ids):
                return (o,c,ids,None)
            if self.best_per_mask[ids].min() >= ub-1e-7:
                return (o,c,ids,None)
            if res is None:
                count+=1
                res=self.relaxation(o,c)
                assert res.success,(o,c,res.message)
                self.add_price(res.eqlin.marginals)
            if self.best_per_mask[ids].min() >= ub-1e-7:
                return (o,c,ids,None)
            free=np.flatnonzero(((o|c)&self.bits)==0)
            if not len(free):
                raise RuntimeError((o,c,res.fun,ub,self.best_per_mask[ids]))
            yy=res.x[free]
            j=free[np.argmax(np.minimum(yy,1-yy))]
            b=int(self.bits[j]); opens=(ids&b)!=0
            return (o,c,ids,(tree(o,c|b,ids[~opens]),tree(o|b,c,ids[opens])))
        t=tree(0,0,self.ids[self.feasible],root)
        def sizes(t):
            o,c,ids,child=t
            if child is None: return 1,1
            a,b=map(sizes,child)
            return a[0]+b[0],1 if self.best_per_mask[ids].min()>=ub-1e-7 else a[1]+b[1]
        print('LP solves',count,'leaves raw/compacted',sizes(t),'final global disjunctive gap',ub-self.best_per_mask[self.feasible].min(),flush=True)

if __name__=='__main__':
    with open('/holdout_task/public_facility.pkl','rb') as f:
        instance=pickle.load(f)
    start=time.monotonic(); Model(instance).run(); print('seconds',time.monotonic()-start)
