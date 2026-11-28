from scipy.optimize import least_squares, minimize_scalar

def _structured_y(z,b,C,y,mode,pen):
    n=len(y); p=P(b); c=D(p)@C
    def scale(v):
        v=v-v[:n].mean(0)
        return v/np.maximum(.035,v[:n].std(0))
    z,p,c,b=map(scale,(z,p,c,b)); X=np.c_[z,p]; A=D(X); L=A[:n]; rng=np.random.default_rng(107)
    if mode in ('rank1','rank2','fullrank1','fullrank2'):
        full=mode.startswith('full'); d=8 if full else 4; rank=int(mode[-1]); U=X if full else z; V=X if full else p
        def prediction(q,A,U,V):
            a=q[9:9+d*rank].reshape(d,rank); h=q[9+d*rank:].reshape(d,rank)
            return A@q[:9]+np.sum((U@a)*(V@h),axis=1)
        def loss(q):return np.r_[prediction(q,L,U[:n],V[:n])-y,pen*q[9:]]
        def jac(q):
            a=q[9:9+d*rank].reshape(d,rank); h=q[9+d*rank:].reshape(d,rank)
            return np.r_[np.c_[L,(U[:n,:,None]*(V[:n]@h)[:,None,:]).reshape(n,-1),(V[:n,:,None]*(U[:n]@a)[:,None,:]).reshape(n,-1)],np.c_[np.zeros((2*d*rank,9)),pen*np.eye(2*d*rank)]]
        q0=fit(L,y); sol=min((least_squares(loss,np.r_[q0,rng.normal(0,.3,2*d*rank)],jac=jac,max_nfev=160) for _ in range(6)),key=lambda r:r.cost)
        return prediction(sol.x,A[n:],U[n:],V[n:])
    if mode.startswith('proj'):
        base=D(np.c_[z,p]); kind=mode[4:]; V=p if kind[-1]=='p' else c; kind=kind[:-1]
        fs={'sin':np.sin,'asinh':np.arcsinh,'tanh':np.tanh,'rat':lambda x:x/(1+x*x),'sq':np.square}
        f=fs[kind]
        def build(q):
            h=f(q[0]*V+q[1]); h=scale(h); return np.c_[base,h,(z[:,:,None]*h[:,None,:]).reshape(len(z),-1)]
        def fitq(q,ret=False):
            H=build(q); F=H[:n]; ridge=np.r_[np.full(9,.004),np.full(4,pen*.5),np.full(16,pen)]; coef=np.linalg.solve(F.T@F+np.diag(ridge),F.T@y)
            return H[n:]@coef if ret else np.r_[F@coef-y,np.sqrt(ridge)*coef,.01*q]
        res=min((least_squares(fitq,[k,s],bounds=([.03,-3],[6,3]),max_nfev=65) for k in (.35,1.5,3.5) for s in (0.,.8)),key=lambda r:r.cost)
        return fitq(res.x,True)
    raise ValueError(mode)
