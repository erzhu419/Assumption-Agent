from pathlib import Path
p=Path('/holdout_task/node_03.py'); s=p.read_text(); start=s.index('    def latent(p):'); end=s.index('\n\ndef _calibrated',start)
s=s[:start]+'''    def latent(p):
        a = (c*np.exp(p[:4])+p[4:]-C1[0])@inv
        # Invert the softsign map on its central monotone branch.
        b = abs(a)
        q, r = Q1
        B = 1+r-q*b
        disc = B*B+4*r*q*b
        return np.sign(a)*2*b/np.maximum(B+np.sqrt(np.maximum(disc, 1e-12)), .02), disc
    def residual(p):
        z, disc = latent(p)
        v = _curvature(z)
        v = (v-v[:n].mean(0))/np.maximum(v[:n].std(0), .1)
        return np.r_[(v-d).ravel(), 8*np.minimum(disc, 0).ravel()]
    low = np.r_[np.full(4,-3.4),C1[0]-.65]
    high = np.r_[np.full(4,.2),C1[0]+.65]
    best = None
    for scale in (.22, .12, .32, .45):
        p0 = np.r_[np.full(4,np.log(scale)), C1[0]]
        opt = least_squares(residual, p0, bounds=(low,high), max_nfev=110)
        if best is None or opt.cost < best.cost:
            best = opt
        if opt.cost < 1e-13:
            break
    # Resolve any rare folds of the softsign using both panel maps together.
    z, _ = latent(best.x)
    def full(p):
        zz = p[8:].reshape(-1,4)
        soft = _one(zz/(1+Q1[0]*abs(zz))+Q1[1]*zz)@C1
        v = _curvature(zz)
        v = (v-v[:n].mean(0))/np.maximum(v[:n].std(0), .1)
        return np.r_[(4*(soft-c*np.exp(p[:4])-p[4:])).ravel(), (v-d).ravel()]
    if best.cost > 1e-12:
        opt = least_squares(full, np.r_[best.x,z.ravel()], max_nfev=45, ftol=1e-10, gtol=1e-10)
        z = opt.x[8:].reshape(-1,4)
        loss = np.sqrt(np.mean(opt.fun**2))
    else:
        loss = np.sqrt(np.mean(best.fun**2))
    return z, loss
''' + s[end:]; p.write_text(s)
