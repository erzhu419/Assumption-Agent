import numpy as np


def _latent(study):
    a = study['panel_a']
    u = np.asarray(a['u'], dtype=float)
    v = np.asarray(a['v'], dtype=float)
    design = np.c_[np.ones(len(u)), u]
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        coef = np.linalg.lstsq(design[keep], v[keep], rcond=None)[0]
        residual = np.linalg.norm(design @ coef - v, axis=1)
        proposed = residual < max(0.001, 3 * np.median(residual))
        if proposed.sum() < design.shape[1] or np.array_equal(proposed, keep):
            break
        keep = proposed
    x = np.vstack((study['train']['x'], study['test']['x']))
    return np.linalg.solve(coef[1:].T, (x - coef[0]).T).T


def _law(u, kind, a):
    if kind == 'cube': return u**3/(1+a*u*u)
    if kind == 'asinh': return np.arcsinh(a*u)
    if kind == 'square': return u*u/(1+a*u*u)
    if kind == 'tanh': return np.tanh(a*u)
    if kind == 'softcube': return u-np.arctan(a*u)/a
    if kind == 'sinres': return u-np.sin(a*u)/a
    if kind == 'power': return np.sign(u)*abs(u)**a
    return u


def _coordinates(x, study, spec, n):
    pk,vk,kind,a,mode,center,post,b,sc=spec
    if pk:
        p=study[pk]; u=np.asarray(p['u'],float); v=np.asarray(p[vk],float)
        if mode in ('single','unmix','polar','inverse'):
            h=_law(u,kind,a); c=np.linalg.lstsq(np.c_[np.ones(len(u)),h],v,rcond=None)[0]
            if mode=='single': z=_law(x,kind,a)@c[1:]+c[0]
            elif mode=='unmix': z=(x-c[0])@np.linalg.pinv(c[1:])
            elif mode=='polar':
                l,_,r=np.linalg.svd(c[1:]); z=_law(x,kind,a)@(l@r)
            else:
                q=(x-c[0])@np.linalg.pinv(c[1:]); grid=np.linspace(-4,4,2049); z=np.interp(q,_law(grid,kind,a),grid)
        else:
            c=np.linalg.lstsq(np.c_[np.ones(len(u)),u,_law(u,kind,a)],v,rcond=None)[0]
            h=_law(x,kind,a)
            if mode=='nonlinear': z=h@c[5:]
            elif mode=='full': z=x@c[1:5]+h@c[5:]+c[0]
            else: z=_law(x,kind,a)@c[5:]+.5*x@c[1:5]
    else: z=_law(x,kind,a)
    if center: z=z-z[:n].mean(0)
    if post=='add': z=np.c_[z,b*x]
    elif post=='resid': z=z+b*x
    elif post=='compress': z=np.arcsinh(b*z)
    elif post=='axis': z=z/np.maximum(z[:n].std(0),1e-8)
    return sc*z


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; n=len(study['train']['x']); x=_latent(study)
    z=_coordinates(x,study,('', '', 'asinh', 1, 'raw', 0, 'add', 0.15, 0.18200000000000002),n)
    return {'z':z[:n].copy(),'zt':z[n:].copy()}
