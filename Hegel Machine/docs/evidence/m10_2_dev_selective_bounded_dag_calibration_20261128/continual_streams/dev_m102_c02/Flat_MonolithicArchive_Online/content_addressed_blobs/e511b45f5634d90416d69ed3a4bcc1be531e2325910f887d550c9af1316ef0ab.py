import numpy as np
from scipy.optimize import least_squares

def fit(x,y): return np.linalg.lstsq(x,y,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]

def maps(s):
    u,v=(s['panel_a'][k] for k in ('u','v')); ok=np.linalg.norm(v,axis=1)<5; u,v=u[ok],v[ok]
    w=np.roll(u,-1,axis=1); off,scale=np.array([fit(aff(u[:,j]),v[:,j]-.32*(u[:,j]-1)*w[:,j]-.16*u[:,j]**3) for j in range(4)]).T
    def f(z): return off+scale*z+.32*(z-1)*np.roll(z,-1,axis=-1)+.16*z**3
    def inv(x): return np.array([least_squares(lambda z:f(z)-h,(h-off)/scale,gtol=1e-12).x for h in x])
    def op(panel,k,fn,p,lo,hi):
        u,v=panel['u'],panel[k]
        def res(p):
            h=aff(fn(u,p));return (h@fit(h,v)-v).ravel()
        p=least_squares(res,p,bounds=(lo,hi),gtol=1e-12).x
        c=fit(aff(fn(u,p)),v)
        return lambda x: aff(fn(x,p))@c
    b=op(s['panel_b'],'v',lambda x,p:np.cos(p[0]*x),[2.6],[2],[3.2])
    c=op(s['panel_c'],'v1',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.62,1.07,1.57],[.1,.1,.2],[4,4,5])
    d=op(s['panel_c'],'v2',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.2,0],[1,-2],[4,2])
    return f,inv,b,c,d


from itertools import combinations_with_replacement


def _predictions(b, u, tr, rows):
    """Keep the affine cosine response unpenalized and shrink higher-order terms."""
    h = aff(b)
    y = tr['y'][rows]
    q, rr = np.linalg.qr(h[rows], mode='reduced')
    residual = y - q @ (q.T @ y)
    leverage = np.sum(q*q, axis=1)
    p = b[:, 0]*b[:, 2] + b[:, 1]*b[:, 3]
    prior = np.column_stack((p, p/(1+.5*(b[:,2]+b[:,3])**2),
                             u[:,1]**3/(1+.35*(u[:,1]+u[:,2])**2)))
    prior /= np.maximum(prior[rows].std(axis=0), 1e-8)
    predictions = {}
    losses = {}
    for label, base in [('u',u),('b',b)]:
        x = base / np.maximum(base[rows].std(axis=0), 1e-8)
        polys = []
        for degree in range(2,6):
            w = np.column_stack([np.prod(x[:, cols], axis=1)
                                for cols in combinations_with_replacement(range(4), degree)])
            polys.append(w / np.maximum(w[rows].std(axis=0), 1e-8))
        for degree in (3,4,5):
            for scale in (.5,.8,1.):
                w = np.column_stack([poly*scale**j for j, poly in enumerate(polys[:degree-1])])
                for j, aug in enumerate((w, np.c_[w, 3*prior[:,:1]], np.c_[w,3*prior])):
                    a = aug[rows]
                    uu, ss, vv = np.linalg.svd(a-q@(q.T@a),full_matrices=False)
                    rv = uu.T@residual
                    for lam in (.1,.3,.6,1.,2.):
                        shrink = ss**2/(ss**2+lam)
                        coeff = vv.T @ (ss/(ss**2+lam)*rv)
                        e = (residual-uu@(shrink*rv))/np.maximum(1e-8,1-leverage-(uu*uu)@shrink)
                        key = (label,degree,scale,j,lam)
                        losses[key] = np.mean(np.abs(e))
                        predictions[key] = h@np.linalg.solve(rr,q.T@(y-a@coeff)) + aug@coeff
    for label, keys in [('ensemble',list(losses)),('cosine_ensemble',[k for k in losses if k[0]=='u']),('no_prior',[k for k in losses if k[0]=='u' and k[3]==0]),('paired_prior',[k for k in losses if k[0]=='u' and k[3]==1]),('rational_prior',[k for k in losses if k[0]=='u' and k[3]==2])]:
        keys = sorted(keys,key=losses.get)[:12]
        weights = np.exp(-(np.array([losses[k] for k in keys])-losses[keys[0]])/.0025)
        predictions[label] = np.average([predictions[k] for k in keys],axis=0,weights=weights)
    return predictions


def _features(study, x):
    _, inv, op, _, _ = maps(study)
    z = inv(x)
    return op(z), np.cos(2.6*z)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    b, u = _features(study, np.r_[tr['x'], study['test']['x']])
    h = aff(b)
    state = h[n:] @ fit(h[:n], tr['traj'])
    y = _predictions(b, u, tr, np.arange(n))[_SCALAR_MODEL][n:]
    terminal = state[:, -1]
    return {'y_hat':y, 'state_pred':state,
            'diagnostics_vec':np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}


_SCALAR_MODEL = ('u', 4, 0.5, 2, 2.0)
