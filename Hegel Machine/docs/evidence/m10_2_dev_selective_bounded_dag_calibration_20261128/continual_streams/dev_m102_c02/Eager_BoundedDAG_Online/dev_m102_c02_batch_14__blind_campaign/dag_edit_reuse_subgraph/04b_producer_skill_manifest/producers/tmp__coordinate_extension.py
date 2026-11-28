
def _basis_bank(z, raw, study):
    """Construct coordinate hypotheses from calibration panels and their interactions."""
    B = _map_coordinates(study['panel_b'], z, 'sin')
    C = _map_coordinates(study['panel_c'], z, 'atan', 'v1')
    D = _map_coordinates(study['panel_c'], z, 'quad', 'v2')
    features = [z, B, C, D]
    for v in (B,C,D):
        features += [np.column_stack([v[:,i]*v[:,j] for i in range(4) for j in range(i,4)])]
    for a,b in ((B,C),(B,D),(C,D),(z,B),(z,C),(z,D)):
        features += [a*b, (a[:,:,None]*b[:,None,:]).reshape(len(a),16)]
        for k in (.5,1.,2.,4.):
            features += [a/(1+k*b*b), b/(1+k*a*a), a*np.tanh(k*b), b*np.tanh(k*a), np.tanh(k*(a+b)), np.tanh(k*(a-b))]
    for k in (.5, 1., 1.5, 2., 2.4, 2.8, 3.2, 4., 5.):
        features += [np.sin(k*z),np.tanh(k*z),np.arctan(k*z),np.cos(k*z)]
    from itertools import combinations_with_replacement
    polynomial = []
    for degree in (2,3,4):
        polynomial.append(np.column_stack([np.prod(z[:,ix],axis=1) for ix in combinations_with_replacement(range(4),degree)]))
    composed=[]
    for u in (raw,B,C,D):
        composed += [_map_coordinates(study['panel_b'],u,'sin'),_map_coordinates(study['panel_c'],u,'atan','v1'),_map_coordinates(study['panel_c'],u,'quad','v2')]
    return [z, B,C,D],polynomial,features,composed


def _fit_indices(z,raw,study,n):
    """Estimate two trajectory indices and a scalar index with training-only model selection."""
    trajectory=np.asarray(study['train']['traj'],float)
    _,_,modes=np.linalg.svd(trajectory,full_matrices=False)
    y=np.asarray(study['train']['y'],float).reshape(-1)
    target=np.column_stack((trajectory@modes[:2].T,y))
    center=target.mean(axis=0)
    target=target-center
    basic,poly,features,composed=_basis_bank(z,raw,study)
    opts=[]
    def standardized(parts):
        a=np.column_stack(parts)
        a=a-a[:n].mean(axis=0)
        return a/np.maximum(a[:n].std(axis=0),1e-8)
    def evaluate(kernel):
        K=kernel[:n,:n]
        mu=K.mean(0); m=mu.mean()
        K=(K-mu[:,None]-mu[None,:]+m)
        cross=kernel[:,:n]-kernel[:,:n].mean(1,keepdims=True)-mu+m
        eig,Q=np.linalg.eigh(K)
        eig=np.maximum(eig,0.)
        QT=Q.T@target
        for a in (1e-5,1e-4,1e-3,.01,.05,.2,1.,4.):
            inv=1/(eig+a)
            dual=Q@(QT*inv[:,None]); denom=(Q*Q)@inv-1/(n*a)
            err=dual/np.maximum(denom[:,None],1e-7)
            opts.append((np.mean(abs(err),0),cross@dual))
    for parts in (basic[:1]+poly[:1],basic[:1]+poly[:2],basic[:1]+poly,basic,features[:4]+features[4:14],features,features+poly,features+poly+composed):
        X=standardized(parts); evaluate(X@X.T/X.shape[1])
    for points in (basic[0],np.column_stack(basic),np.column_stack(basic[1:])):
        X=standardized([points]); gram=X@X.T/X.shape[1]; sq=np.diag(gram); d=np.maximum(sq[:,None]+sq-2*gram,0)
        for scale in (.07,.2,.5,1.,2.):
            evaluate(np.exp(-scale*d)); r=np.sqrt(3*scale*d); evaluate((1+r)*np.exp(-r))
    # Sparse forward selection complements the smooth kernels for sharply structured coordinates.
    for parts in (features[:14]+poly,features+poly,features+poly+composed):
        X=standardized(parts); A=X[:n]; N=X.shape[1]; J=np.ones((n,1))/np.sqrt(n)
        for mode in (0,1,2,3):
            Q=J.copy(); R=target.copy(); Ares=A.copy(); selected=[]
            for step in range(min(18,n//3)):
                norm=np.linalg.norm(Ares,axis=0); norm[selected]=np.inf
                cor=(Ares.T@R)/np.maximum(norm[:,None],1e-5)
                merit=np.abs(cor[:,mode]) if mode<3 else np.linalg.norm(cor/np.maximum(target.std(axis=0),.015),axis=1)
                j=int(np.argmax(merit)); q=Ares[:,j]/max(norm[j],1e-12)
                if norm[j]<1e-5 or not np.isfinite(norm[j]):break
                selected.append(j); Q=np.column_stack((Q,q)); R-=np.outer(q,q@R); Ares-=np.outer(q,q@Ares)
                err=R/np.maximum(1-(Q*Q).sum(1,keepdims=True),.01)
                # Discount the optimism from searching a large feature dictionary.
                penalty=1+(.08+.015*step)*np.log2(N)/8
                score=np.mean(abs(err),0)*penalty
                if mode<3: score[np.arange(3)!=mode]*=1.20
                opts.append((score,X[:,selected]@np.linalg.lstsq(A[:,selected],target,rcond=None)[0]))
    rates=np.stack([e for e,_ in opts]); predictions=np.stack([p for _,p in opts]); out=[]
    for j in range(3):
        order=np.argsort(rates[:,j]); best=order[:min(8,len(order))]; scale=max(rates[best[0],j],1e-7)
        w=np.exp(-18*(rates[best,j]/scale-1)); out.append(w@predictions[best,:,j]/w.sum()+center[j])
    indices=np.column_stack(out)
    # The orthogonal residual reference direction completes the coordinate chart.
    anchor=basic[1][:,0]; des=np.column_stack((np.ones(len(z)),indices)); anchor=anchor-des@np.linalg.lstsq(des[:n],anchor[:n],rcond=None)[0]
    return np.column_stack((indices[:,:2],indices[:,2],.1*anchor))


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; train=study['train']['x']; test=study['test']['x']
    x=np.concatenate((train,test)); z=_undo_sensor(x,_fit_sensor(study['panel_a']))
    coords=_fit_indices(z,x,study,len(train))
    return {'c':coords[:len(train)],'query':coords[len(train):]}
