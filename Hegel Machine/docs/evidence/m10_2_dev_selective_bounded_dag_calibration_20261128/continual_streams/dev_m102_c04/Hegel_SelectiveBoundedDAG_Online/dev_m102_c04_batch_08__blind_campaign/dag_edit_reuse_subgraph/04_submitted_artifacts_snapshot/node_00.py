import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _cal(panel):
    u,v=panel['u'],panel['v']; a=_affine(u); mask=np.ones(len(u),bool)
    for _ in range(8):
        w=_ls(a[mask],v[mask]); e=np.linalg.norm(a@w-v,axis=1)
        new=e<=max(1e-9,4*np.median(e))
        if new.sum()<8 or np.array_equal(mask,new): break
        mask=new
    d=np.array([_ls(a[mask][:,[0,j+1]],v[mask,j]) for j in range(4)]).T
    if np.median(np.linalg.norm((u*d[1]+d[0]-v)[mask],axis=1))<=1.35*np.median(e[mask]):
        return d[0],np.diag(d[1])
    return w[0],w[1:]


def _quad(z):
    return z[:,[0,0,1,2]]*z[:,[1,3,2,3]]


def _ops(panel):
    u=panel['u']; v=panel['v2']
    def err(k):
        f=_affine(np.cos(k*u)); return np.linalg.norm(f@_ls(f,v)-v)
    grid=np.linspace(.2,7,69); j=np.argmin([err(k) for k in grid]); k=minimize_scalar(err,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded').x
    return k,_ls(_affine(_quad(u)),panel['v1']),_ls(_affine(np.cos(k*u)),v)


from itertools import combinations_with_replacement


def _poly(x, deg):
    return np.column_stack([np.prod(x[:, j], axis=1)
                            for d in range(1, deg+1)
                            for j in combinations_with_replacement(range(x.shape[1]), d)])


def _kernel_library(x, n, a, A, b, B, k, wp, wr, telemetry):
    q = np.linalg.solve(A.T, (x-a).T).T
    bi = np.linalg.inv(B)
    alpha, beta = np.asarray(telemetry).ravel()[:2]
    if not (0.15 < abs(alpha) < 5): alpha = 1.25
    if not (0.01 < beta < 1): beta = .22
    poly = _poly(q, 2)
    def gram(f):
        f = np.asarray(f).reshape(len(x), -1)
        f = f-f[:n].mean(axis=0)
        f = f/np.maximum(f[:n].std(axis=0), 1e-6)
        return f@f.T/f.shape[1]
    def radial(z):
        d = np.sum((z[:, None, :]-z[None, :, :])**2, axis=2)
        for l in [.45, .75, 1.1, 1.7, 2.6]:
            yield np.exp(-d/(2*l*l))
    gp = gram(poly)
    for basis in [_poly(q, i) for i in (1, 2, 3, 4, 5)]:
        yield gram(basis)
    for z in [q, q@B+b, np.c_[q, q@B+b], np.arcsinh(q)]:
        for g in radial(z):
            yield g
            yield g + gp
    routes = [q, x, q@B+b, (q-b)@bi, q@B, q@bi, q+beta*(q@B+b), q-beta*(q@B+b), q+beta*((q-b)@bi), q-beta*((q-b)@bi), alpha*q, q/alpha, q+beta*q*q, q-beta*q*q, np.arctanh(np.clip(q/1.55,-.97,.97))*1.55, np.sinh(q), ((x-b)@bi@A+a)@B+b]
    ps = [_affine(_quad(z))@wp for z in routes]
    rs = [_affine(np.cos(k*z))@wr for z in routes]
    for j, (z,p,r) in enumerate(zip(routes,ps,rs)):
        h = p*r
        f = np.c_[q,p,r,h]
        yield gram(f)
        yield gram(f)+gp
        yield gram(np.c_[poly, p, r, p*p, r*r, h, h.sum(axis=1)])
        yield gram(np.c_[poly, p, r, (p[:,:,None]*r[:,None,:]).reshape(len(q),-1)])
        for c in [beta, abs(alpha), 1.]:
            rp = p/(1+c*r*r)
            rr = r/(1+c*p*p)
            dot = h.sum(1)
            nr = (r*r).sum(1)
            yield gram(np.c_[q,p,r,rp,rr,dot/(1+c*nr)])
            yield gram(np.c_[q,p,r,h, np.sin(c*(p+r)), np.cos(c*(p-r)), p*np.tanh(c*r)])
            yield gram(np.c_[f, f/(1+c*nr[:,None])])
        for w in [abs(alpha), k, k*abs(alpha), k/abs(alpha), 4.8]:
            cs, sn = np.cos(w*z), np.sin(w*z)
            yield gram(np.c_[poly,cs,sn])
            yield gram(np.c_[q, p, cs, sn, p*cs, p*sn, q*cs, q*sn])
            yield gram(np.c_[q,p,cs,sn,(p[:,:,None]*cs[:,None,:]).reshape(len(q),-1),(p[:,:,None]*sn[:,None,:]).reshape(len(q),-1)])
        for l,g in enumerate(radial(f[: ,4:])):
            if l in (2,3,4): yield g + gram(f)
        for r2 in rs[:9]:
            yield gram(np.c_[q,p,r2,p*r2])
            yield gram(np.c_[poly,p,r2,p*r2])
    yield gram(np.c_[q, ps[-1], rs[0], ps[-1]*rs[0]])


def run_node(inputs: dict) -> dict:
    s=inputs['__input__']; x=s['train']['x']; xt=s['test']['x']; n=len(x)
    a,A=_cal(s['panel_a']); b,B=_cal(s['panel_b']); k,wp,wr=_ops(s['panel_c'])
    y=s['train']['y']; t=s['train']['traj']; target=np.column_stack([y,t]); mean=target.mean(axis=0)
    centered=target-mean; kernels=[]; losses=[]
    grid=np.array([1e-7, 1e-5, 5e-4, .004, .02, .085, .3, 1., 4.])
    for f in _kernel_library(np.vstack([x,xt]),n,a,A,b,B,k,wp,wr,s['telemetry_vec']):
        # Center against training rows only, and leave the intercept unpenalized.
        m=f[:,:n].mean(axis=1); f=f-m[:,None]-m[None,:]+m[:n].mean()
        e,u=np.linalg.eigh(f[:n,:n]); e=np.maximum(e,0.)
        inv=1/(e[:,None]+grid)
        weights=u.T@centered
        loo=np.einsum('ni,ig,io->ngo',u,inv,weights,optimize=True)/np.maximum((u*u)@inv-1/(n*grid),1e-14)[:,:,None]
        err=np.abs(loo)
        score=np.column_stack([err[:,:,0].mean(axis=0),err[:,:,1:].mean(axis=(0,2))])
        jj=score.argmin(axis=0); losses.append(score[jj,np.arange(2)]); kernels.append((f,grid[jj]))
    loss=np.asarray(losses)
    def ensemble(j):
        rank=np.argsort(loss[:,j]); best=loss[rank[0],j]
        sel=rank[:8]; sel=sel[loss[sel,j]<=1.12*best]
        w=np.exp(-28*(loss[sel,j]/max(best,1e-15)-1)); w/=w.sum()
        return sum(c*kernels[i][0] for i,c in zip(sel,w)),sum(c*kernels[i][1][j] for i,c in zip(sel,w))
    f,pen=ensemble(0)
    pred=mean[0]+f[n:,:n]@np.linalg.solve(f[:n,:n]+pen*np.eye(n),centered[:,0])
    f,pen=ensemble(1)
    c=np.linalg.cholesky(f[:n,:n]+pen*np.eye(n))
    g=np.linalg.solve(c,f[n:,:n].T).T
    return {'train_features':_affine(c), 'test_features':_affine(g), 'regression':pred}
