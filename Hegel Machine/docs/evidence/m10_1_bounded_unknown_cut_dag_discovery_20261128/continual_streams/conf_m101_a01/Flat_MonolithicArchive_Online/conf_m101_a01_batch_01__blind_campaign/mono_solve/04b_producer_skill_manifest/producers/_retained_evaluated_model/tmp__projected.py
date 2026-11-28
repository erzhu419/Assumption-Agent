def _projected_readout(x, z, y, family):
    n = len(x)
    center, scale = x.mean(0), x.std(0)
    x, z = (x-center)/scale, (z-center)/scale
    F, G = _trend(x, 2), _trend(z, 2)
    beta = np.linalg.lstsq(F, y, rcond=None)[0]
    Q = np.zeros((4,4))
    for value,(a,b) in zip(beta[5:], ((a,b) for a in range(4) for b in range(a,4))):
        Q[a,b] = Q[b,a] = value if a == b else value/2
    _, V = np.linalg.eigh(Q)
    a,b = V[:,0],V[:,-1]
    directions = [np.eye(4), np.column_stack((a,b)), (a+b)[:,None], (a-b)[:,None], beta[1:5,None]]
    if family == 1:
        directions += [np.eye(4)[:,j:j+1] for j in range(4)]
    if family == 2:
        directions += [a[:,None],b[:,None]]
    U = np.linalg.qr(F, mode='complete')[0][:,F.shape[1]:]
    r = U.T@y
    K, C = [], []
    for j,V in enumerate(directions):
        w, v = x@V, z@V
        w, v = w/w.std(0), v/w.std(0)
        for length in ((1.2,2.4) if j < 2 else (.7,1.5)):
            K.append(np.exp(-cdist(w,w,'sqeuclidean')/(2*length**2)))
            C.append(np.exp(-cdist(v,w,'sqeuclidean')/(2*length**2)))
    K.append(np.eye(n)); K = np.asarray(K)
    P = U.T@K@U
    eye = np.eye(len(r))
    def loss(p):
        w = np.exp(p)
        A = np.einsum('k,kij->ij',w,P)
        L = cholesky(A,lower=True)
        alpha = cho_solve((L,True),r)
        grad = cho_solve((L,True),eye)-np.outer(alpha,alpha)
        return np.log(np.diag(L)).sum()+.5*r@alpha, .5*w*np.einsum('ij,kij->k',grad,P)
    p = minimize(loss,np.r_[np.full(len(K)-1,-5.),-9.],jac=True,bounds=[(-19,3)]*len(K),method='L-BFGS-B',options={'maxiter':120}).x
    w = np.exp(p)
    L = cholesky(np.einsum('k,kij->ij',w,K),lower=True)
    t = solve_triangular(L,np.column_stack((F,y)),lower=True)
    beta = np.linalg.lstsq(t[:,:-1],t[:,-1],rcond=None)[0]
    alpha = solve_triangular(L.T,t[:,-1]-t[:,:-1]@beta)
    return G@beta + np.einsum('k,kij->ij',w[:-1],C)@alpha
