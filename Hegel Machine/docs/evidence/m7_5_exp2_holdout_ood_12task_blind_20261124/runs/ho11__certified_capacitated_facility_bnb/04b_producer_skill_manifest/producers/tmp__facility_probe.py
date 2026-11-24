import pickle
import time
from math import fsum
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import csr_matrix, eye, hstack, kron, vstack, diags

class Model:
    def __init__(self, p):
        self.f = np.asarray(p['fixed_cost'], dtype=float)
        self.s = np.asarray(p['capacity'], dtype=float)
        self.d = np.asarray(p['demand'], dtype=float)
        self.c = np.asarray(p['cost_matrix'], dtype=float)
        m, n = self.c.shape
        self.m, self.n = m, n
        self.total = fsum(self.d)
        self.full = (1 << m)-1
        self.bits = 1 << np.arange(m)
        cap = hstack([-diags(self.s), kron(eye(m), np.ones((1,n)))], format='csr')
        links = hstack([-kron(eye(m), self.d[:,None]), eye(m*n)], format='csr')
        self.eq = hstack([csr_matrix((n,m)), kron(np.ones((1,m)),eye(n))],format='csr')
        self.ineq = vstack([cap,links],format='csr')
        self.obj = np.r_[self.f,self.c.ravel()]
        self.masks = np.arange(1 << m)
        self.open = (self.masks[:,None] & self.bits) != 0
        self.feas = np.array([fsum(self.s[row]) >= self.total for row in self.open])
        self.indices = np.argsort(self.c, axis=1)
        self.opts = {'primal_feasibility_tolerance':1e-9, 'dual_feasibility_tolerance':1e-9}

    def mip(self):
        mat = vstack([self.eq,self.ineq],format='csr')
        opts = {'mip_rel_gap':0.0}
        r = milp(self.obj, integrality=np.r_[np.ones(self.m),np.zeros(self.m*self.n)],bounds=Bounds(0,np.r_[np.ones(self.m), np.full(self.m*self.n,np.inf)]),constraints=LinearConstraint(mat,np.r_[self.d, np.full(self.ineq.shape[0],-np.inf)],np.r_[self.d,np.zeros(self.ineq.shape[0])]),options=opts)
        return r

    def relax(self,C,O,strong):
        m=self.m
        lower = (self.bits & O)!=0
        avail = (self.bits & C)==0
        A = self.ineq if strong else self.ineq[:m]
        r = linprog(self.obj,A_ub=A,b_ub=np.zeros(A.shape[0]),A_eq=self.eq,b_eq=self.d,bounds=np.column_stack([np.r_[lower,np.zeros(m*self.n)],np.r_[avail,np.full(m*self.n,np.inf)]]),method='highs',options=self.opts)
        if not r.success:
            return r,None,None
        q=np.minimum(r.ineqlin.marginals[:m],0)
        t=np.minimum(r.ineqlin.marginals[m:].reshape(m,self.n),0) if strong else np.zeros_like(self.c)
        u=np.minimum(r.eqlin.marginals, np.min(self.c-q[:,None]-t,axis=0))
        z=self.f+self.s*q+t@self.d
        lp = u@self.d+np.sum(z[lower])+np.sum(np.minimum(z[avail & ~lower],0))
        scores=self.site_scores(r.eqlin.marginals)
        ids=self.feas & ((self.masks & C)==0) & ((self.masks & O)==O)
        kn = r.eqlin.marginals@self.d + np.min(np.sum(self.open[ids]*scores,axis=1))
        return r,lp,kn

    def site_scores(self,u):
        # For a given demand-price vector, fill the best negative reduced-cost arcs first, up to demand or capacity.
        order=np.argsort(self.c-u,axis=1)
        d=self.d[order]
        v=np.take_along_axis(self.c-u,order,axis=1)
        a=np.clip(self.s[:,None]-np.cumsum(d,axis=1)+d,0,d)
        return self.f+np.sum(a*np.minimum(v,0),axis=1)

    def check_counts(self,strong,knapsack,upper):
        low=[]; opened=0; root=None
        def visit(C,O):
            nonlocal opened,root
            rest=self.full ^ (C|O)
            if not self.feas[self.full ^ C]:
                low.append(np.inf); return
            r,lp,kn=self.relax(C,O,strong)
            if not r.success:
                raise RuntimeError(r.message)
            opened+=1
            lb=max(lp,kn) if knapsack else lp
            if not (C|O): root=(lp,kn)
            if lb>=upper-9.5e-6 or not rest:
                low.append(lb); return
            y=r.x[:self.m]
            options=np.flatnonzero((self.bits & rest)!=0)
            i=options[np.argmax(np.minimum(y[options],1-y[options]))]
            bit=int(self.bits[i]); visit(C|bit,O); visit(C,O|bit)
        start=time.monotonic(); visit(0,0)
        print('strong',strong,'knapsack',knapsack,'root',root,'leaves',len(low),'LPs',opened,'gap',upper-min(low),'seconds',round(time.monotonic()-start,3),flush=True)

if __name__ == '__main__':
    with open('/holdout_task/public_facility.pkl','rb') as f: p=pickle.load(f)
    model=Model(p)
    t=time.monotonic(); r=model.mip()
    print('MIP',r.success,r.fun, 'seconds',time.monotonic()-t,flush=True)
    for strong,knapsack in [(False,False),(True,False),(True,True)]:
        model.check_counts(strong,knapsack,r.fun)
