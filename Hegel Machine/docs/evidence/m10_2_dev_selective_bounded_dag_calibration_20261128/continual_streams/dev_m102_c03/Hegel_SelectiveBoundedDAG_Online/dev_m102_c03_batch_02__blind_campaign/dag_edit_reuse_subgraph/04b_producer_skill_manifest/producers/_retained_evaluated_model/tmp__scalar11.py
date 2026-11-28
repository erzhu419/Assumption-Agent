def _scalar_kernels(z, study):
    # Recover the two panel transfer laws independently of the response labels.
    p=study['panel_c']; u=np.asarray(p['u']); v=p['v1']; w=p['v2']
    def rfun(x,p):return x/(1+p[0]*abs(x)+p[1]*x*x+p[2]*abs(x)**3)
    def residual(f,y):return (f@np.linalg.lstsq(f,y,rcond=None)[0]-y).ravel()
    par=least_squares(lambda p:residual(_linear(rfun(u,p)),v),[.89,.30,-.033],bounds=([.6,.12,-.06],[1.2,.46,.03])).x
    fr=least_squares(lambda p:residual(_linear(np.sin(p[0]*u)**2),w),[1.2],bounds=(.85,1.5)).x[0]
    return [z,np.sin(fr*z),rfun(z,par),np.tanh(z),np.tanh(1.35*z),np.arctan(1.8*z)]


def _scalar_models(z,study,y,configs):
    from itertools import combinations_with_replacement
    from scipy.spatial.distance import cdist
    n=len(y); mu=y.mean(); sd=max(y.std(),1e-9); yy=(y-mu)/sd; eye=np.eye(n)
    coords=_scalar_kernels(z,study); polys={}; dists={}; features={}
    def gram(x):
        x=x-x[:n].mean(0); x=x/np.maximum(np.sqrt(np.mean(x[:n]**2,axis=0)),1e-9)/np.sqrt(x.shape[1]); return x@x[:n].T
    for k,x in enumerate(coords):
        x=x/np.sqrt(np.mean(x[:n]**2,axis=0)); dists[k]=cdist(x,x[:n],metric='sqeuclidean'); P={}
        for deg in range(1,6):
            by_support={}
            for j in combinations_with_replacement(range(4),deg):by_support.setdefault(tuple(sorted(set(j))),[]).append(np.prod(x[:,j],axis=1))
            P[deg]={j:gram(np.column_stack(f)) for j,f in by_support.items()}
        polys[k]=P
    def summed(k,deg,power):
        return sum(power**(d-1)*sum(polys[k][d].values())/len(polys[k][d]) for d in range(1,deg+1))
    def fit(K,reg):
        C=cho_factor(K[:n]+reg*eye,check_finite=False); A=cho_solve(C,eye,check_finite=False); a=A@yy
        return sd*np.sqrt(np.mean((a/A.diagonal())**2)),mu+sd*K[n:]@a
    def optimize(K,lo,criterion):
        K=np.asarray(K); m=len(K); F=K[:,:n].reshape(m,-1).copy(); G=K[:,n:].reshape(m,-1).copy()
        def fun(p,predict=False):
            w=np.exp(p); C=cho_factor((w[:-1]@F).reshape(n,n)+w[-1]*eye,check_finite=False); A=cho_solve(C,eye,check_finite=False); a=A@yy
            err=a/A.diagonal(); score=sd*np.sqrt(np.mean(err**2))
            if predict:return score,mu+sd*(w[:-1]@G).reshape(-1,n)@a
            if criterion=='ml':
                D=.5*(A-np.outer(a,a)); loss=.5*yy@a+np.log(np.diag(C[0])).sum()
            else:
                d=A.diagonal(); ax=A@(err/d); D=(-2*np.outer(ax,a)+2*(A*(err**2/d)[None,:])@A)/n; loss=np.mean(err**2)
            return loss,w*np.r_[F@D.ravel(),np.trace(D)]
        p=minimize(fun,np.r_[np.full(m,-3.),-6.],jac=True,method='L-BFGS-B',bounds=[(-15,6)]*m+[(lo,-1)],options={'maxiter':90,'ftol':2e-7}).x
        return fun(p,True)
    cache={}; res=[]
    for typ,ks,d,g,reg in configs:
        if typ in ('poly','rbf','matern','plus'):
            key=typ,ks,d,g
            if key not in cache:
                if typ=='poly':cache[key]=sum(summed(k,d,g) for k in ks)/len(ks)
                elif typ=='rbf':cache[key]=sum(np.exp(-g*dists[k]) for k in ks)/len(ks)
                elif typ=='matern':cache[key]=sum((1+np.sqrt(g*dists[k])+g*dists[k]/3)*np.exp(-np.sqrt(g*dists[k])) for k in ks)/len(ks)
                else:cache[key]=sum(summed(k,d,.6)+np.exp(-g*dists[k]) for k in ks)/len(ks)
            out=fit(cache[key],reg)
        else:
            K=[]
            if typ in ('tensor','loo'):
                for k in ks:
                    for j in sorted(set.union(*(set(polys[k][q]) for q in range(1,d+1)))):
                        K.append(sum(g**(q-len(j))*polys[k][q][j] for q in range(len(j),d+1) if j in polys[k][q]))
            else:
                for k in ks:
                    K.extend(sum(polys[k][q].values())/len(polys[k][q]) for q in range(1,d+1)); K.extend(np.exp(-h*dists[k]) for h in (.04,.15,.5))
            out=optimize(K,reg,'loo' if typ=='loo' else 'ml')
        res.append(out)
    return res
