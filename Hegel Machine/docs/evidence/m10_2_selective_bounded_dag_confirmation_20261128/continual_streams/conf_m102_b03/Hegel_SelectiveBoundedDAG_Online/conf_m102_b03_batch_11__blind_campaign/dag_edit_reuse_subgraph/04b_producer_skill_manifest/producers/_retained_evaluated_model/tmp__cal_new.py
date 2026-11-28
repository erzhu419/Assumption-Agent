def _calibrate(panel, x):
    u, v = (np.asarray(panel[k]) for k in ('u','v'))
    w = np.roll(u, 2, axis=1)
    rng = np.random.default_rng(420)
    samples = [rng.choice(len(u),6,replace=False) for _ in range(90)]
    def basis(z, f):
        return np.stack((np.ones_like(z),z,np.sin(f*np.roll(z,2,axis=1))),axis=2)
    def coeff(a, ids):
        return np.stack([_ls(a[ids,j], v[ids,j]) for j in range(u.shape[1])])
    def pred(a,b):
        return np.einsum('njk,jk->nj',a,b)
    best = np.inf
    for f in np.arange(0.8,5.01,.15):
        a = basis(u,f)
        for ids in samples:
            b = coeff(a,ids)
            er = np.linalg.norm(pred(a,b)-v,axis=1)
            score = np.median(er)
            if score < best:
                best, start, er0 = score, f, er
    ids = np.argsort(er0)[:max(6,int(len(u)*.75))]
    for _ in range(3):
        def residual(p):
            a = basis(u,p[0]); return (pred(a, coeff(a,ids))[ids]-v[ids]).ravel()
        f = least_squares(residual,[start],bounds=(.1,8.)).x[0]
        a = basis(u,f); b = coeff(a,ids)
        er = np.linalg.norm(pred(a,b)-v,axis=1)
        ids = np.flatnonzero(er < max(.002, 4*np.median(er)))
        start=f
    z = (x-b[:,0])/b[:,1]
    for _ in range(80):
        nxt = (x-b[:,0]-b[:,2]*np.sin(f*np.roll(z,2,axis=1)))/b[:,1]
        if np.max(np.abs(nxt-z),initial=0.) < 1e-12: break
        z = nxt
    return z
