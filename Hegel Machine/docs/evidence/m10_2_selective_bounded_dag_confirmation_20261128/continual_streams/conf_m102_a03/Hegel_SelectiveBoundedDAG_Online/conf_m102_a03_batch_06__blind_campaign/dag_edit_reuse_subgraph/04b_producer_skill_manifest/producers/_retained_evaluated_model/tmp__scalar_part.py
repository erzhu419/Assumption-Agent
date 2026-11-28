
def _scalar_kernel(study, z, y):
    """Estimate a panel response map and select sparse readout interactions."""
    from scipy.optimize import minimize_scalar, least_squares
    from itertools import combinations_with_replacement

    n, d = len(y), z.shape[1]
    u = np.asarray(study['panel_b']['u'], float)
    v = np.asarray(study['panel_b']['v'], float)
    uv = np.vstack((u, z))
    d2 = list(combinations_with_replacement(range(d), 2))
    one = np.ones((len(uv), 1))
    cross = np.column_stack([uv[:, i]*uv[:, j] for i,j in d2])
    cyc = uv*np.roll(uv, 1, axis=1)
    maps = []
    for fun, limits in ((np.tanh, (.8, 2.0)), (np.sin, (2.2, 3.4)),
                        (lambda x:x/(1+x*x), (.55, 1.25))):
        for extra in (one[:, :0], uv**2, cyc, cross):
            def form(w):
                return np.column_stack((one, uv, fun(w*uv), extra))
            def fit(w, ret=False):
                X = form(w)
                a = X[:len(u)]
                h = np.linalg.pinv(a, rcond=1e-9)
                c = h@v
                e = (a@c-v)/np.maximum(.025,1-np.einsum('ij,ji->i',a,h))[:,None]
                return (np.mean(e**2), X[len(u):]@c) if ret else np.mean(e**2)
            sol = minimize_scalar(fit, bounds=limits, method='bounded', options={'xatol':1e-4})
            maps.append(fit(sol.x, True))
    maps.sort(key=lambda x:x[0])
    weights = np.array([1/max(e,1e-12)**2 for e,_ in maps]); weights /= weights.sum()
    maps.append((maps[0][0], sum(w*b for w,(_,b) in zip(weights,maps))))
    Cs = [_basis(z,t) @ np.linalg.lstsq(_basis(study['panel_c']['u'],t),study['panel_c'][k],rcond=None)[0]
          for t,k in [('c','v1'),('q','v2')]]
    C,Q = Cs
    rng = np.random.default_rng(281)
    folds = [f for _ in range(2) for f in np.array_split(rng.permutation(n),4)]
    rows = np.arange(n)
    train_folds = [(np.setdiff1d(rows,f),f) for f in folds]
    candidates = []
    def feature(f):
        sd = np.maximum(np.std(f[:n], axis=0), 1e-7)
        return (f-np.mean(f[:n],axis=0))/sd
    def lin(A, b, ridge):
        return np.linalg.solve(A.T@A+np.diag(ridge), A.T@b)
    def gram(feat, penalties):
        F = feat / np.sqrt(penalties)
        K = F@F[:n].T
        return K[:n]+np.eye(n),K[n:]
    def losses(es):
        es=np.array(es)
        return np.mean(es)+.12*np.max(es)
    def fixed(F, pen, tag):
        K, V = gram(F,pen)
        es=[]
        for a,b in train_folds:
            c=cho_solve(cho_factor(K[np.ix_(a,a)],check_finite=False),y[a],check_finite=False)
            es.append(np.mean(abs(K[np.ix_(b,a)]@c-y[b])))
        candidates.append((losses(es),K,V,tag))
    nl_jobs=[]
    stack=lambda f,g:(f[:,:,None]*g[:,None,:]).reshape(len(z),-1)
    for ib,(_,B) in enumerate(maps):
        L = np.column_stack((np.ones(len(z)),feature(B),feature(z*np.roll(B,1,axis=1))))
        G = np.column_stack((z,B))
        q = np.column_stack([G[:,i]*G[:,j] for i,j in combinations_with_replacement(range(2*d),2)])
        core = np.column_stack((z[:,-1]**2, B[:,1]*B[:,3], z[:,-1]*B[:,0]))
        Aug = np.column_stack((z,q,C,Q,z*np.roll(B,-1,axis=1),stack(z,C),stack(z,Q)))
        for j,f in enumerate((core,np.column_stack((core,q)),Aug,np.column_stack((Aug,stack(z,q),stack(B,q))))):
            if j==3 and ib not in (0,1,len(maps)-1):continue
            U=feature(f)
            for a in (0.03,0.2,1.,5.,25.,150.):
                fixed(np.column_stack((L,U)),np.r_[np.full(L.shape[1],.001),np.full(U.shape[1],a)],('ridge',ib,j,a))
        for sh in (1,-1,2):
            for F in (B,z,C):
                for H in (B,z):
                    T=F*np.roll(H,sh,axis=1)
                    for sc in (.1,.4,1.):
                        R=T/(1+sc*np.roll(H,sh,axis=1)**2)
                        U=feature(np.column_stack((core,R,T)))
                        fixed(np.c_[L,U],np.r_[np.full(L.shape[1],.001),np.full(U.shape[1],2.)],('cyclic',ib,sh,sc))
        if ib in (0,1,len(maps)-1):
            nl_jobs.extend([(L,z,ib,'z'),(L,B,ib,'B'),(np.c_[L,feature(core)],B,ib,'coreB')])
        if ib in (0,1,2,len(maps)-1):
            U = feature(np.column_stack((core,Aug)))
            F = np.c_[L,U]
            def ard(a, trunc):
                T=F[a]; yy=y[a]; k=L.shape[1]
                alp=np.r_[np.full(k,.01),np.full(U.shape[1],30.)]
                bet=400.
                for _ in range(75):
                    cov=cho_solve(cho_factor(bet*T.T@T+np.diag(alp),check_finite=False),np.eye(T.shape[1]),check_finite=False)
                    coef=bet*(cov@(T.T@yy)); gamma=np.maximum(0,1-alp*np.diag(cov))
                    alp[k:]=np.clip(gamma[k:]/(coef[k:]**2+1e-8),.03,1e6)
                    bet=np.clip((len(yy)-sum(gamma))/max(1e-9,np.sum((yy-T@coef)**2)),30,3e4)
                pen=alp/bet
                if trunc:pen[k:]=np.where(abs(coef[k:])>.012,pen[k:],1e5)
                return pen
            for trunc in (False,True):
                es=[]
                for a,b in train_folds:
                    pen=ard(a,trunc); es.append(np.mean(abs(F[b]@lin(F[a],y[a],pen)-y[b])))
                K,V=gram(F,ard(rows,trunc)); candidates.append((losses(es),K,V,('ARD',ib,trunc)))
    for L,W,ib,tag in nl_jobs:
        W=feature(W); D=np.c_[np.ones(len(W)),W]
        for typ in ('rat','rat2','sin','tanh','gate'):
            def profile(a):
                l=L[a]; x=D[a]; yy=y[a]
                U=np.linalg.qr(l,mode='reduced')[0]; yr=yy-U@(U.T@yy)
                def col(mat,h):
                    r=mat@h
                    if typ=='rat':return (r/(1+r*r))[:,None]
                    if typ=='rat2':return (1/(1+r*r))[:,None]
                    if typ=='sin':return np.sin(r)[:,None]
                    if typ=='tanh':return np.tanh(r)[:,None]
                    return mat/(1+r[:,None]**2)
                def res(h):
                    f=col(x,h); g=f-U@(U.T@f)
                    if g.shape[1]==1:
                        coef=(g.T@yr)/(np.sum(g*g)+1e-7)
                    else:
                        coef=lin(g,yr,np.full(g.shape[1],.001))
                    return np.r_[g@coef-yr,.007*h]
                starts = np.vstack((np.r_[0.,np.ones(d)*.25], np.eye(d+1)[1:],np.random.default_rng(29).normal(0,.6,(4,d+1))))
                rr=min((least_squares(res,h,bounds=(-5,5),max_nfev=65) for h in starts),key=lambda r:r.cost)
                f=np.c_[L,col(D,rr.x)]
                return f,np.r_[np.full(L.shape[1],.0002),np.full(f.shape[1]-L.shape[1],.001)]
            es=[]
            for a,b in train_folds[:4]:
                F,pen=profile(a); es.append(np.mean(abs(F[b]@lin(F[a],y[a],pen)-y[b])))
            F,pen=profile(rows); K,V=gram(F,pen)
            candidates.append((losses(es)*1.045,K,V,('profile',tag,ib,typ)))
    candidates.sort(key=lambda r:r[0]); best=candidates[0][0]
    # Blend near-tied models in prediction space by adding equivalent
    # cross-covariances against a shared positive-definite training block.
    eligible=[c for c in candidates if c[0]<=best*1.13][:10]
    weights=np.array([np.exp(-22*(c[0]/best-1)) for c in eligible]); weights/=sum(weights)
    K=eligible[0][1]
    V=sum(w*(cho_solve(cho_factor(k,check_finite=False),v.T,check_finite=False).T@K)
          for w,(_,k,v,_) in zip(weights,eligible))
    return K,V
