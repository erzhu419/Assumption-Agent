"""Recover nonlinear calibration coordinates and fit the two readouts."""
import itertools
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _poly(x, order):
    return np.column_stack([np.prod(x[:, ix], axis=1) for ix in itertools.combinations_with_replacement(range(x.shape[1]), order)])


def _cube_inverse(x, k):
    return 2 / np.sqrt(3*k) * np.sinh(np.arcsinh(1.5*np.sqrt(3*k)*x)/3)


def _panel_models(x, v, q, is_b):
    models = []
    funcs = [(lambda x,k: np.sin(k*x), 2.9), (lambda x,k: x/(1+k*x*x), .8), (_cube_inverse, 2.)]
    for f, start in funcs:
        def design(a,k):
            return np.c_[_aff(a),f(a,k)]
        def residual(k):
            m = design(x,k)
            return (m @ np.linalg.lstsq(m,v,rcond=None)[0] - v).ravel()
        fit = least_squares(residual, np.full(x.shape[1],start), bounds=(.05,8.),max_nfev=55)
        m = design(x,fit.x)
        h = m @ np.linalg.pinv(m)
        err = np.mean(np.abs((h@v-v)/np.maximum(1-h.diagonal(),1e-6)[:,None]))
        models.append((err,design(q,fit.x) @ np.linalg.lstsq(m,v,rcond=None)[0],f(q,fit.x)))
    models.sort(key=lambda m:m[0]); return models[:2 if is_b else 1]


def _readout(u, b, c, labels, q_train, q_test):
    n = len(labels)
    blocks, families = [], []
    def add(x, group):
        x = np.asarray(x)
        if x.ndim == 1: x = x[:,None]
        scale = x[:n].std(0)
        good = scale > 1e-8
        x = (x[:,good] - x[:n,good].mean(0))/scale[good]
        if x.shape[1]:
            blocks.append(x); families.extend([group]*x.shape[1])
    add(u,0)
    add(u*u,1)
    add(_poly(u,2)[:,[i!=j for i,j in itertools.combinations_with_replacement(range(4),2)]],2)
    add(np.c_[u**3,u**5],3)
    add(np.c_[u**4,u**6],4)
    add(_poly(u,3),5)
    add(_poly(u,4),6)
    for j, (_,w,f) in enumerate(b+c):
        g = 7 + 10*j
        add(w,g); add(_poly(w,2),g+1); add(_poly(w,3),g+2)
        add(_poly(w,4),g+3)
        add(f,g+4); add(np.c_[f*f,u*f],g+5)
        add(np.c_[*[u[:,i,None]*f for i in range(4)]],g+6)
        add(np.c_[*[w[:,i,None]*f for i in range(4)]],g+7)
        add(np.c_[np.sin(2*w),np.tanh(2*w),w/(1+w*w)],g+8)
        a,d,e,h = w.T
        pairs = np.column_stack([w[:,i]+sg*w[:,j] for i,j in itertools.combinations(range(4),2) for sg in (1,-1)])
        add(np.c_[np.sin(2*pairs),np.tanh(2*pairs),np.cos(2*pairs),a*e+d*h],g+9)
    A = np.column_stack(blocks)
    fam = np.asarray(families)
    K = []
    for g in np.unique(fam):
        x=A[:,fam==g]; K.append(x@x[:n].T/x.shape[1])
    for x in (u,b[0][1],np.c_[u,b[0][1]]):
        x=x/np.maximum(x[:n].std(0),.05)
        dr=(x[:,None]-x[None,:n])**2
        for ell in (.13,.5,1.4):
            R=np.exp(-ell*dr)
            K.extend((R.prod(axis=2), R.mean(axis=2)))
    K=np.asarray(K); train=K[:,:n,:].copy(); eye=np.eye(n)
    mu=labels.mean(0); yc=labels-mu
    scale=np.maximum(np.linalg.norm(yc,axis=0)/np.sqrt(n),1e-8)
    Y=yc/scale
    # Evidence estimates shrink unhelpful feature groups. Cross-validation then
    # compares the grouped kernel with models having individual sparse terms.
    def evidence(t):
        weights=np.exp(t[1:]); noise=np.exp(t[0]); G=np.tensordot(weights,train,axes=1)+noise*eye+5.
        L=cho_factor(G,check_finite=False); P=cho_solve(L,Y,check_finite=False)
        Z=cho_solve(L,eye,check_finite=False)-P@P.T/Y.shape[1]
        loss=np.log(L[0].diagonal()).sum()+0.5*np.sum(Y*P)/Y.shape[1]+.002*np.sum(t[1:]**2)
        return loss,np.r_[.5*noise*np.trace(Z),.5*weights*np.einsum('gij,ij->g',train,Z)+.004*t[1:]]
    init=np.full(len(K)+1,-3.); init[0]=-4
    opt=minimize(evidence,init,jac=True,method='L-BFGS-B',bounds=[(-11,1)]+[(-10,3)]*len(K),options={'maxiter':90})
    kw=np.exp(opt.x[1:]); noise=np.exp(opt.x[0]); candidates=[]
    def kernel_candidate(G, eta):
        L=cho_factor(G[:n]+eta*eye+5.,check_finite=False)
        P=cho_solve(L,Y,check_finite=False)
        diag=cho_solve(L,eye,check_finite=False).diagonal()
        candidates.append((Y-P/diag[:,None], (G[n:]+5.)@P))
    GK=np.tensordot(kw,K,axes=1)
    for f in (.2,1.,4.): kernel_candidate(GK,noise*f)
    for g in (.002,.02,.15): kernel_candidate(K.mean(0),g)
    cols=[np.ones(n)]
    AN=A[:n]; H=AN/np.sqrt(n)
    # Greedy sparse correction atop compact priors, with analytic row holdout.
    for prior in (np.array([0,1,2,3]),np.array([7,8]),np.array([0,1,2,3,7,8])):
        ids=np.where(np.isin(fam,prior))[0]
        for strength in (.012,.11):
            pp=A[:,ids]; G=pp@pp[:n].T
            kernel_candidate(G,strength)
    # Threshold each output's feature weights separately; their physical paths
    # do not need the same algebraic terms.
    for iters in (5,12,24):
        loo_all=[]; pred_all=[]
        for j in range(Y.shape[1]):
            y=Y[:,j]; penalty=np.ones(A.shape[1]); penalty *= np.where(fam<5,.4,1.)
            for _ in range(iters):
                var=1/penalty
                G=(H*var)@H.T
                L=cho_factor(G+.008*eye+5.,check_finite=False)
                p=cho_solve(L,y,check_finite=False)
                w=var*(H.T@p)
                penalty=np.clip(1/np.sqrt(w*w+.001),.04,35.)
            loo_all.append(y-p/cho_solve(L,eye,check_finite=False).diagonal())
            pred_all.append(A[n:]@w/np.sqrt(n)+5.*p.sum())
        candidates.append((np.column_stack(loo_all),np.column_stack(pred_all)))
    cv=np.stack([a for a,b in candidates],axis=-1)
    forecasts=np.stack([b for a,b in candidates],axis=-1)
    # Regularized leave-one-out stacking; weights are fit only on the current training rows.
    result=[]
    for j in range(Y.shape[1]):
        C=cv[:,j]; p=forecasts[:,j]; target=Y[:,j]
        e=np.mean((C-target[:,None])**2,axis=0)
        base=np.exp(-4*(e/e.min()-1)); base/=base.sum()
        w=minimize(lambda w: np.mean((C@w-target)**2)+.08*e.min()*np.sum((w-base)**2),base,
                   method='SLSQP',bounds=[(0,1)]*len(base),constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':50,'ftol':1e-9}).x
        result.append(p@w)
    return np.column_stack(result)*scale+mu


def run_node(inputs: dict) -> dict:
    study, enc = inputs['__input__'], inputs['node_00']
    panel = study['panel_b']
    a = np.linalg.lstsq(_aff(panel['u']), panel['v'], rcond=None)[0]
    z = np.concatenate((enc['z'],enc['zt']))
    q = np.linalg.solve(a[1:].T,(z-a[0]).T).T
    b = _panel_models(panel['u'],panel['v'],q,True)
    cp = study['panel_c']; c = _panel_models(cp['u'],cp['v1'],q,False)
    train=study['train']; traj=np.asarray(train['traj']); avg=traj.mean(0)
    _,_,V=np.linalg.svd(traj-avg,full_matrices=False)
    modes=V[:2]; n=len(traj)
    pred=_readout(q,b,c,np.c_[train['y'],(traj-avg)@modes.T],q[:n],q[n:])
    yh=pred[:,0]; sp=avg+pred[:,1:]@modes
    return {'y_hat': yh,'state_pred':sp,'diagnostics_vec':np.array([yh.mean(),yh.std(),sp[:,-1].mean(),sp[:,-1].std()])}
