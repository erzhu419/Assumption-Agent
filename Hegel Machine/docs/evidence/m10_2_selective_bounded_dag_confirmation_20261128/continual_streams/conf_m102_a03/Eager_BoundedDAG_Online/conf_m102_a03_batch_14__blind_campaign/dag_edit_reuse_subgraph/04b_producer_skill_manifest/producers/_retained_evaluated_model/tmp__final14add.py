

def _structured(st, enc):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    Y = np.column_stack((st['train']['y'], st['train']['traj']))
    n = len(Y)
    u, z, v, w, panels = _components(st, enc)
    sine = next(a for _, name, a in panels if name == 'sin')
    z = (z + sine) / 2
    def standard(x):
        return (x - x[:n].mean(0)) / np.maximum(x[:n].std(0), 1e-8)
    base = [standard(x) for x in (u,z,v,w)]
    kernels = []
    def add_kernel(K):
        K = 0.5 * (K + K.T)
        kernels.append(K / max(np.mean(np.diag(K)[:n]), 1e-10))
    def add_features(D):
        D = np.asarray(D).reshape(len(u), -1)
        D = D - D[:n].mean(0)
        add_kernel(D @ D.T)
    pair_kernels = []
    for a, X in enumerate(base):
        add_features(X)
        add_features(_quad(X))
        add_features(np.column_stack([X[:, i]*X[:, j]*X[:, k] for i in range(4) for j in range(i,4) for k in range(j,4)]))
        terms = []
        for j in range(4):
            x = X[:, j]
            F = np.column_stack([np.sin(f*x) for f in (.5,1,2)] + [np.cos(f*x)-1 for f in (.5,1,2)] + [np.tanh(x),x/(1+x*x)])
            F = standard(F)
            K = F@F.T / F.shape[1]
            terms.append(K)
            p = x[:, None] - x[None, :]
            add_kernel(K + .25*np.exp(-p*p/2))
        pairs = []
        for j in range(4):
            for k in range(j+1,4):
                F = np.column_stack([X[:,j]*X[:,k], X[:,j]**2*X[:,k], X[:,j]*X[:,k]**2, X[:,j]**2*X[:,k]**2])
                F = standard(F)
                K = .5 * (terms[j]*terms[k] + F@F.T/F.shape[1])
                pairs.append(K)
                if a < 2:
                    add_kernel(K)
        pair_kernels.append(sum(pairs)/len(pairs))
    add_kernel(sum(pair_kernels))
    for X in (np.c_[base[0],base[1]], np.c_[base[1],base[2],base[3]],np.column_stack(base)):
        add_features(_quad(X))
        G = X@X.T/X.shape[1]
        dist = np.maximum(np.diag(G)[:,None]+np.diag(G)[None,:]-2*G,0)
        for h in (.7,1.3,2.2,4.):
            add_kernel(np.exp(-dist/(2*h*h)))
    K = np.array(kernels)
    I = np.eye(n); one=np.ones(n)
    At = np.concatenate((K[:,:n,:n],I[None]),axis=0).reshape(len(K)+1,-1)
    Xt = K[:,n:,:n]
    U, S, _ = np.linalg.svd(Y[:,1:]-Y[:,1:].mean(0),full_matrices=False)
    amp = U[:,0]*S[0]
    amp *= np.sign(amp @ (Y[:,-1]-Y[:,-1].mean())) or 1
    yy = standard(Y[:,0]); tt = standard(amp)
    smooth = standard(yy + np.sign(yy@tt)*tt)
    delta = standard(yy - np.sign(yy@tt)*tt)
    models=[]
    def fit_r(R, penalty):
        R=np.asarray(R).reshape(n,-1)
        def obj(logw):
            weights=np.exp(logw)
            L=cho_factor((weights@At).reshape(n,n),check_finite=False)
            J=cho_solve(L,I,check_finite=False); j=J@one
            P=J-np.outer(j,j)/sum(j); a=P@R
            loss=.5*np.sum(R*a)+R.shape[1]*(np.log(np.diag(L[0])).sum()+.5*np.log(sum(j)))
            G=.5*(R.shape[1]*P-a@a.T)
            grad=weights*(At@G.ravel())
            loss+=penalty*np.log1p(weights[:-1]/.05).sum()
            grad[:-1]+=penalty*weights[:-1]/(.05+weights[:-1])
            return loss,grad
        t0=np.r_[np.full(len(K),-5.),-6.]
        p=minimize(obj,t0,jac=True,method='L-BFGS-B',bounds=[(-14,5)]*len(K)+[(-11,1)],options={'maxiter':155,'ftol':2e-8})
        return np.exp(p.x)
    def prediction(weights):
        L=cho_factor((weights@At).reshape(n,n),check_finite=False)
        J=cho_solve(L,I,check_finite=False); j=J@one; P=J-np.outer(j,j)/sum(j)
        beta=P@Y
        return Y-beta/np.diag(P)[:,None], np.einsum('i,ijk->jk',weights[:-1],Xt)@beta+(j@Y)/sum(j)
    for R in (yy,smooth,tt,np.c_[smooth,delta]):
        for penalty in (.08,.4,1.2):
            models.append(prediction(fit_r(R,penalty)))
    return models


def _final_prediction(st,enc):
    from scipy.optimize import minimize
    Y = np.column_stack((st['train']['y'],st['train']['traj']))
    incumbent = _candidate(st,enc)
    e0 = incumbent[0][0]
    reference = np.mean([m[3] for m in [m for m in incumbent if m[0] <= 1.035*e0][:6]],axis=0)
    models = _structured(st,enc)
    R = np.stack([a-Y for a,b in models]); P = np.stack([b for a,b in models]); k=len(P)
    out=reference.copy()
    for inds, blend in [(slice(0,1),.8),(slice(1,None),.55)]:
        A=R[:,:,inds].reshape(k,-1); scale=max(np.median(abs(A)),.005)
        def obj(w):
            r=w@A; q=np.sqrt(r*r+(.15*scale)**2)
            return np.mean(q)+.095*scale*np.sum(w*w), A@(r/q)/r.size+.19*scale*w
        opt=minimize(obj,np.ones(k)/k,jac=True,method='SLSQP',bounds=[(0,1)]*k,constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(k)},options={'maxiter':80,'ftol':1e-10})
        w=np.maximum(opt.x,0); w/=w.sum()
        if np.mean(abs(w@A))<=1.10*(incumbent[0][2][0 if inds.start==0 else 1]):
            out[:,inds]=(1-blend)*reference[:,inds]+blend*np.tensordot(w,P[:,:,inds],axes=1)
    return out


def run_node(inputs: dict) -> dict:
    study, encoded=inputs['__input__'],inputs['node_00']
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        p=_final_prediction(study,encoded)
    else:
        with threadpool_limits(limits=1,user_api='blas'):
            p=_final_prediction(study,encoded)
    y,t=p[:,0],p[:,1:]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
