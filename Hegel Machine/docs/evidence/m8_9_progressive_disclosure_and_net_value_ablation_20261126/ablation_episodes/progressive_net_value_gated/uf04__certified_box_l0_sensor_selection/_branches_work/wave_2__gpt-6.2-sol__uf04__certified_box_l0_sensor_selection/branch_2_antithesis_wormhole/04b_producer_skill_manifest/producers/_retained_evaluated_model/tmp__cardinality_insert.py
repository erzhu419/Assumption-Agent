
class _CardinalitySearch:
    """Branch on total cardinality, then on indicators within that cardinality."""
    def __init__(self, search):
        self.s = search

    def oracle(self, dual, k, state):
        s = self.s
        # An on indicator pays the charge across the whole box. Including the
        # inactive interval in that box does not change the global minimum.
        a = s.X.T @ dual
        x0 = -np.clip(a/s.ridge, -s.t, s.t)
        x1 = -np.clip(a/s.ridge, -s.m, s.m)
        q0 = x0*(a + .5*s.ridge*x0)
        q1 = x1*(a + .5*s.ridge*x1) + s.lam
        on = state == 1
        free = np.flatnonzero(state == 0)
        count = k - np.count_nonzero(on)
        order = np.argsort((q1-q0)[free], kind='stable')
        on[free[order[:count]]] = True
        w = np.where(on, x1, x0)
        lb = -dual @ (s.y + .5*dual) + np.where(on,q1,q0).sum() - s.err
        return w, on, float(lb)

    def relax(self, k, state, center, parent):
        from scipy.optimize import minimize
        s = self.s
        free = np.flatnonzero(state == 0)
        needed = k - np.count_nonzero(state == 1)
        if needed == 0 or needed == len(free):
            state = state.copy()
            state[free] = 1 if needed else -1
            s._polish(state == 1, center)
            # BVLS supplies a high-accuracy center for a terminal convex QP.
            from scipy.optimize import lsq_linear
            radius = np.where(state == 1, s.m, s.t)
            A = np.vstack([s.X, np.sqrt(s.ridge)*np.eye(s.p)])
            b = np.r_[s.y, np.zeros(s.p)]
            fit = lsq_linear(A*radius, b, bounds=(-1.,1.), method='bvls', tol=1e-14)
            w = np.clip(fit.x,-1.,1.)*radius
            s._accept(w)
            _,_,lb = self.oracle(s.X@w-s.y, k, state)
            return max(parent,lb), -1, w
        atoms, masks, costs = [], [], []
        alpha = np.empty(0)
        lb = parent
        w = center
        for it in range(40):
            atom,on,val = self.oracle(s.X@w-s.y, k, state)
            lb = max(lb,val)
            if lb >= s.best-8e-6:
                break
            if it == 0 or it % 10 == 9:
                s._polish(on, atom)
                if lb >= s.best-8e-6:
                    break
            if any(np.array_equal(atom,old) for old in atoms):
                break
            atoms.append(atom)
            masks.append(on)
            costs.append(.5*s.ridge*(atom@atom))
            alpha = np.r_[alpha,0.] if len(alpha) else np.ones(1)
            A = np.asarray(atoms)
            V = A @ s.X.T
            Q = V @ V.T
            c = np.asarray(costs)-V@s.y
            scale = max(1., abs(Q).max(), abs(c).max())
            Q, c = Q/scale, c/scale
            if len(alpha)>1:
                info = minimize(lambda z: (.5*(z@Q@z)+c@z, Q@z+c),alpha,jac=True,
                                method='SLSQP',bounds=[(0.,1.)]*len(alpha),
                                constraints={'type':'eq','fun':lambda z:z.sum()-1.,
                                             'jac':lambda z:np.ones(len(z))},
                                options={'ftol':1e-14,'maxiter':60})
                alpha = info.x
            nxt = alpha @ A
            if np.max(np.abs(s.X@(nxt-w))) < 1e-12:
                break
            w = nxt
        if lb >= s.best-8e-6:
            return lb, -1, w
        mass = alpha @ np.asarray(masks) if len(alpha) else on.astype(float)
        branch = free[np.argmax(np.minimum(mass[free],1-mass[free]))]
        return lb, int(branch), w

    def solve(self):
        s = self.s
        s._accept(s.w)
        start = np.clip(np.linalg.solve(s.G,s.c),-s.m,s.m)
        s._local(start)
        s._local(s.w)
        queue = []
        closed = np.inf
        def add(k,state,w,parent):
            nonlocal closed
            s.nodes += 1
            lb,j,w = self.relax(k,state,w,parent)
            if j < 0 or lb >= s.best-8e-6:
                closed = min(closed,lb)
            else:
                heapq.heappush(queue,(lb,s.nodes,k,state,j,w))
        s.nodes = 1
        x = s.X[:,0]
        sign = np.where(x@s.X >= 0.,1.,-1.)
        h = float(x@x)
        c = float(x@s.y)
        counts = sorted(range(s.p+1),key=lambda k:abs(k-np.count_nonzero(abs(s.w)>s.t)))
        for k in counts:
            w = sign*np.copysign(min(s.m,abs(c)/(h*max(1,k)+s.ridge)),c)*(k/s.p)
            add(k,np.zeros(s.p,dtype=np.int8),w,k*s.lam-s.err)
        while queue and s.best-min(closed,queue[0][0])>_GAP and s.nodes+2<=_LIMIT:
            lb,_,k,state,j,w = heapq.heappop(queue)
            if lb >= s.best-8e-6:
                closed = min(closed,lb)
                continue
            for v in (-1,1):
                child = state.copy()
                child[j] = v
                add(k,child,w,lb)
        return s.result(min(closed,queue[0][0] if queue else np.inf))

