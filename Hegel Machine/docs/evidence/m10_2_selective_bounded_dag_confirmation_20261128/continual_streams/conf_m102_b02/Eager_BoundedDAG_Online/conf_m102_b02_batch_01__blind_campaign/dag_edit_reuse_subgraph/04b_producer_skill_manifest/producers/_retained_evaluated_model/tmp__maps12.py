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
