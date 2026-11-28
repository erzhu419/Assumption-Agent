"""Recover latent coordinates from panel A and apply the panel-B operator."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, root, minimize


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _latent(panel, x):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    opposite = (np.arange(u.shape[1]) + 2) % u.shape[1]
    residual = np.linalg.norm(v - _affine(u) @ _ls(_affine(u), v), axis=1)
    keep = residual < max(6 * np.median(residual), 0.1)
    if keep.sum() >= 8:
        u, v = u[keep], v[keep]

    def basis(z, p):
        return np.stack((np.ones_like(z), z, np.tanh(p[0]*z), np.sin(p[1]*z[:, opposite])), axis=-1)

    def project(p, return_coef=False):
        a = basis(u, p)
        coef = np.stack([_ls(a[:, j], v[:, j]) for j in range(u.shape[1])])
        return coef if return_coef else (np.sum(a*coef, axis=-1)-v).ravel()

    p = least_squares(project, [1.8, 2.4], bounds=([0.3, 0.8], [4., 4.])).x
    coef = project(p, True)
    return np.stack([root(lambda z: np.sum(basis(z[None], p)[0]*coef, axis=-1)-q, q).x for q in x])


def _panel_b(panel, z):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def basis(q, a):
        return np.c_[_affine(q / (1 + a*q*q)), q]
    def loss(a):
        d = basis(u, a)
        return np.mean((d @ _ls(d, v) - v)**2)
    a = minimize_scalar(loss, bounds=(.05, 4.), method='bounded').x
    return basis(z, a) @ _ls(basis(u, a), v)


def _gram_blocks(study, x, n):
    z = _latent(study['panel_a'], x)
    b = _panel_b(study['panel_b'], z)
    # Separate first-order, mixed second-order and smooth remainder priors.
    def gram(f):
        f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1.e-7)
        return (f@f.T)/f.shape[1]
    h = float(np.asarray(study['telemetry_vec'])[3])
    zc = gram(z)
    bc = gram(b)
    rc = gram(z/(1+h*z*z))
    gc = gram(z**3/(1+h*z*z))
    ec = gram(z*z)
    sc = gram(np.sin(float(np.asarray(study['telemetry_vec'])[1])*z))
    pc = []
    p=study['panel_c'];u=np.asarray(p['u'],float)
    for key, basis in [('v1',lambda a:np.c_[_affine(a),np.arctan(2.4*a),a**3]),('v2',lambda a:np.c_[_affine(a),a*a,a**3])]:
        pc.append(gram(basis(z)@_ls(basis(u),np.asarray(p[key],float))))
    fc=(zc+bc)/2
    dc=np.maximum(fc.diagonal()[:,None]+fc.diagonal()[None,:]-2*fc,0.)
    blocks=[zc,bc,rc,gc,ec,sc,*pc,zc*zc,zc*bc,bc*bc,zc*gc,gc*gc,zc*ec,bc*ec,bc*pc[0],bc*pc[1],fc**3,fc**4,fc**5,np.exp(-.3*dc),np.exp(-.1*dc)]
    return np.stack([k/max(np.mean(k.diagonal()[:n]),1.e-10) for k in blocks])


def _smooth_models(grams, response):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    n=len(response); eye=np.eye(n); ones=np.ones((n,1));
    avg=response.mean(); sd=max(response.std(),1.e-7); y=(response-avg)/sd
    gs=np.ascontiguousarray(grams[:,:n,:n]); flat=gs.reshape(len(gs),-1)
    def solve(w, full=False):
        k=(w[:-1]@flat).reshape(n,n)+w[-1]*eye+1.
        fact=cho_factor(k,check_finite=False); ki=cho_solve(fact,eye,check_finite=False); a=ki@y
        loo=y-a/np.diag(ki)
        if full:
            pred=(np.einsum('k,kij->ij',w[:-1],grams[:,:,:n])+1.)@a
            return np.mean(np.abs(y-loo)),pred*sd+avg,loo*sd+avg
        v=ki-np.outer(a,a)
        val=.5*(y@a)+np.log(np.diag(fact[0])).sum()
        grad=np.r_[.5*(flat@v.ravel()),.5*np.trace(v)]*w
        return val,grad
    def objective(a): return solve(np.exp(a))
    results=[]
    for initial in (.003,.03):
        w0=np.full(len(gs)+1,.04);w0[-1]=initial
        opt=minimize(objective,np.log(w0),jac=True,method='L-BFGS-B',bounds=[(-13,4)]*len(gs)+[(-11,-.65)],options={'maxiter':145,'ftol':2e-8})
        results.append(solve(np.exp(opt.x),True))
    for ii in ([0,1,8,9,10],[0,1,8,9,10,17],[0,1,3,8,9,10,11,12,17],[0,1,2,3,8,9,10,11,12],[0,1,8,9,10,17,18],[0,1,8,9,10,17,18,19],[0,1,8,9,10,20,21],list(range(len(gs)))):
        for order in (.1,1.):
            for noise in (2.e-5,.0002,.002,.01,.04):
                w=np.full(len(gs)+1,0.);w[ii]=1./len(ii);w[np.array(ii)[np.array(ii)>=17]]*=order;w[-1]=noise
                results.append(solve(w,True))
    results.sort(key=lambda q:q[0]); best=results[0][0]
    sel=[q for q in results[:12] if q[0]<best*1.10+1.e-8]
    wt=np.exp(-35*(np.array([q[0] for q in sel])/max(best,1.e-8)-1)); wt/=wt.sum()
    return sum(q[1]*w for q,w in zip(sel,wt)),sum(q[2]*w for q,w in zip(sel,wt)),best*sd


def _scalar_profile(grams, y, amplitude, fits):
    """Fit the scalar-specific residual while profiling the shared modes."""
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize

    n = len(y)
    ymean, yscale = y.mean(), max(y.std(), 1.e-10)
    target = (y-ymean)/yscale
    center = amplitude.mean(0)
    scale = np.maximum(amplitude.std(0), 1.e-8)
    observed = (amplitude-center)/scale
    smooth = (np.column_stack([f[0] for f in fits])-center)/scale
    loo_amp = (np.column_stack([f[1] for f in fits])-center)/scale
    eye = np.eye(n)
    candidates = []
    base = (grams[0]+grams[1])/2
    for nonlinear in (False, True):
        def shared(a):
            m = _affine(a)
            if nonlinear:
                u = a[:,0]
                m = np.c_[m, u/(1.+.7*u*u), u*a[:,1]]
            return m
        design = shared(observed)
        pred_design = shared(smooth)
        loo_design = shared(loo_amp)
        delta = loo_design-design
        # Broad prior on the linear shared terms; shrink optional nonlinear terms.
        prior = np.array([1., 2., 1.] + ([.1, .02] if nonlinear else []))
        cov_shared = (design*prior)@design.T
        blocks = np.stack([grams[i] for i in (0,1,2,3,4,5,8,9,10,11,13,14,16,17,18)] + [base**3, base**4])
        flat = np.ascontiguousarray(blocks[:,:n,:n]).reshape(len(blocks),-1)
        mat = np.ascontiguousarray(blocks[:,:,:n])
        def solve(w, predict=False):
            k = cov_shared+(w[:-1]@flat).reshape(n,n)+w[-1]*eye
            factor = cho_factor(k,check_finite=False)
            inv = cho_solve(factor,eye,check_finite=False)
            coef = inv@target
            if predict:
                diag = inv.diagonal()
                # Sherman-Morrison deletion for the profiled shared coefficients.
                beta = prior*(design.T@coef)
                deletion = ((inv@design)*prior)*(coef/diag)[:,None]
                out = target-coef/diag+delta@beta-np.sum(delta*deletion,axis=1)
                full = pred_design@beta+np.einsum('k,kij,j->i',w[:-1],mat,coef)
                return np.mean(abs(out-target))*yscale, full*yscale+ymean, out*yscale+ymean
            score = .5*(target@coef)+np.log(np.diag(factor[0])).sum()
            v = inv-np.outer(coef,coef)
            gradient = .5*np.r_[flat@v.ravel(),np.trace(v)]*w
            return score,gradient
        for start in (.002,.015):
            init = np.full(len(blocks)+1,.015); init[-1]=start
            opt = minimize(lambda p: solve(np.exp(p)),np.log(init),jac=True,method='L-BFGS-B',bounds=[(-13.,2.)]*len(blocks)+[(-10.,-3.)],options={'maxiter':120,'ftol':1.e-8})
            candidates.append(solve(np.exp(opt.x),True))
        for width in (.03,.15,.6):
            for nugget in (.0001,.001,.006):
                w = np.zeros(len(blocks)+1)
                w[[0,1,6,7,8,13,14]] = width*np.array([1,1,1,1,1,.15,.025])
                w[-1] = nugget
                candidates.append(solve(w,True))
    candidates.sort(key=lambda m:m[0]); top=candidates[:8]
    error=np.array([m[0] for m in top]); w=np.exp(-19.*(error/error.min()-1));w/=w.sum()
    full=sum(v[1]*a for v,a in zip(top,w)); loo=sum(v[2]*a for v,a in zip(top,w))
    return full,loo,float(np.mean(abs(loo-y)))


def _coordinates(study, x, q):
    n=len(x); g=_gram_blocks(study,np.r_[x,q],n)
    traj=np.asarray(study['train']['traj'],float); y=np.asarray(study['train']['y'],float)
    _,_,v=np.linalg.svd(traj,full_matrices=False)
    amp=traj@v[:2].T
    fits=[_smooth_models(g,amp[:,j]) for j in range(2)]
    ys=_smooth_models(g,y)
    # Estimate the shared component separately from the scalar-specific response.
    # Only out-of-fold model estimates are used to pick the decomposition strength.
    yt=[]
    for coupling in (.5,1.):
        aa=_affine(amp)
        beta=_ls(aa,y)*coupling
        res=_smooth_models(g,y-aa@beta)
        full=res[0]+_affine(np.column_stack([f[0] for f in fits]))@beta
        loo=res[1]+_affine(np.column_stack([f[1] for f in fits]))@beta
        yt.append((full,loo,float(np.mean(abs(loo-y)))))
    profile = _scalar_profile(g,y,amp,fits)
    yy=[ys,*yt,profile]; err=np.array([k[2] for k in yy]); wt=np.exp(-13*(err/err.min()-1));wt/=wt.sum()
    scalar=sum(k[0]*w for k,w in zip(yy,wt))
    c=np.column_stack((scalar, fits[0][0], fits[1][0], np.zeros(len(scalar))))
    return c, (err, [f[2] for f in fits])


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; x=np.asarray(study['train']['x'],float); q=np.asarray(study['test']['x'],float)
    c,_=_coordinates(study,x,q)
    return {'c':c[:len(x)],'query':c[len(x):]}
