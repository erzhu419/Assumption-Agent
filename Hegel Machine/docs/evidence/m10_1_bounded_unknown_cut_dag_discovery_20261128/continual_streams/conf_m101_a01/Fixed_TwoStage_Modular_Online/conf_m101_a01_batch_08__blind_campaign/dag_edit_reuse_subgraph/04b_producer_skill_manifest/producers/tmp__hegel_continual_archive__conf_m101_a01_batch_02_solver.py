from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _quad(x):
    return np.column_stack([_aff(x)] + [x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2)])


def _ols(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _panels(study, x):
    a, b, c = (study[k] for k in ('panel_a', 'panel_b', 'panel_c'))
    au, av = _aff(a['u']), a['v']
    good = np.ones(len(au), dtype=bool)
    for _ in range(8):
        w = _ols(au[good], av[good]); r = np.linalg.norm(au @ w - av, axis=1)
        good = r < max(3 * np.median(r), .001)
    shift, scale = w[0], np.diag(w[1:])
    bw = _ols(_aff(b['u']), b['v'])
    bf = lambda z: _aff(z) @ bw
    bi = lambda z: np.linalg.solve(bw[1:].T, (z-bw[0]).T).T
    roots = dict(x=x, a=x*scale+shift, i=(x-shift)/scale, b=bf(x), j=bi(x), ib=bf((x-shift)/scale), ij=bi((x-shift)/scale), bi=(bf(x)-shift)/scale, ji=(bi(x)-shift)/scale)
    z = roots.copy()
    for key, func, start in [('v1', np.arctan, 2.4), ('v2', np.cos, 1.)]:
        u, v = c['u'], c[key]
        def design(d, p):
            return np.column_stack((_aff(d), func(d*p)))
        def residual(p):
            f = design(u, p)
            return (f @ _ols(f, v)-v).ravel()
        p = least_squares(residual, np.full(4, start), bounds=(.25, 5)).x
        cv = _ols(design(u, p), v)
        for k, d in roots.items():
            op = design(d, p) @ cv
            z[k+key[-1]] = op
            z[k+key[-1]+'b'] = bf(op)
            z[k+key[-1]+'j'] = bi(op)
    return z


def _ridge(f, target, lam):
    n = len(target)
    scale = np.maximum(f[:n].std(0), .08); scale[0] = 1.
    g = f / scale
    a = g[:n]; b = g[n:]
    return b @ np.linalg.solve(a.T@a + lam*np.eye(a.shape[1]), a.T@target)


def _scalar_parts(z, y):
    x = z['x']; n = len(y); q = _quad(x); result = []
    for key, p in [('bi1', 1.), ('i2j', 2.3), ('bi1', 1.6)]:
        d = z[key]
        v = np.sin(p*d)/p if p == 1.6 else d/(1+(p*d)**2)
        f = np.c_[q,v,(d[:,:,None]*v[:,None,:]).reshape(-1,16)]
        result.append(_ridge(f, y, .07))
    d = np.c_[z['bi1'], z['bi2']]
    result.append(np.arcsinh(_ridge(_quad(np.c_[x,d]), np.sinh(y), .3)))
    tar = 1.65*np.arctanh(np.clip(y/1.65,-.999,.999))
    a = _aff(x)
    r = z['x1'] - a @ _ols(a[:n],z['x1'][:n]); f = np.c_[a,r]
    h = np.zeros((4,4)); h[np.triu_indices(4)] = _ols(q[:n],y)[5:]; h=(h+h.T)/2
    ev,ve=np.linalg.eigh(h); u=ve[:,-1]*np.sqrt(abs(ev[-1])); v=ve[:,0]*np.sqrt(abs(ev[0]));
    def design(p):
        a,b=x@p[:4],x@p[4:]
        return np.c_[f,a*b]
    def res(p):
        d=design(p)[:n]
        return np.r_[d@_ols(d,tar)-tar,.03*(np.linalg.norm(p.reshape(2,4),axis=1)-1)]
    opt=least_squares(res,np.r_[u+v,u-v],max_nfev=90)
    d=design(opt.x)
    result.append(1.65*np.tanh((d[n:]@_ols(d[:n],tar))/1.65))
    return np.array(result)


def _base_solve(study):
    train = study['train']
    x = np.concatenate((train['x'],study['test']['x']))
    z = _panels(study,x)
    y = _scalar_parts(z, train['y']).mean(0)
    preds=[]
    for k in ('ib2j','b2j','i2j'):
        d=z[k]; f=np.c_[_quad(x),_quad(d)[:,1:]]
        preds.append(_ridge(f,train['traj'],.01))
    state = np.mean(preds,axis=0)
    end = state[:,-1]
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}


def _library(study,x,y,tr):
    z=_panels(study,x); n=len(y); q=_quad(x)
    roots={k:z[k] for k in ('x','a','i','b','j','ib','ij','bi','ji')}
    cov=np.zeros((4,4)); cov[np.triu_indices(4)]=_ols(q[:n],y)[5:];cov=(cov+cov.T)/2
    _,v=np.linalg.eigh(cov);roots['e']=x@v[:,[0,3,1,2]]
    _,_,v=np.linalg.svd(tr,full_matrices=False)
    amp=tr@v[:2].T
    h=q@_ols(q[:n],amp)
    yield ('state','xamp'),np.c_[q,h*h,h[:,0]*h[:,1],(x[:,:,None]*h[:,None,:]).reshape(-1,8)]
    for key,d in roots.items():
        if key=='e':
            yield (key,'ridges'),np.c_[q,d[:,:2]**3,d[:,:2]**4,d[:,:2]**5]
        cubic=np.column_stack([np.prod(d[:,idx],axis=1) for idx in combinations_with_replacement(range(4),3)])
        yield (key,'cube'),np.c_[q,cubic]
        yield (key,'diag'),np.c_[q,d**3,d**4,d**5]
        pairs=np.column_stack([d[:,i]+sign*d[:,j] for i in range(4) for j in range(i+1,4) for sign in (-1,1)])
        for p in (0.8,1.45,2.15):
            for mode in ('s','t','r'):
                if mode=='s': v=np.sin(p*d); w=np.cos(p*d)
                elif mode=='t': v=np.tanh(p*d);w=v*v
                else: v=d/(1+(p*d)**2);w=v*v
                yield (key,mode,p,'sat'),_quad(v)
                yield (key,mode,p,'mix'),np.c_[q,v,w,(d[:,:,None]*v[:,None,:]).reshape(-1,16)]
                yield (key,mode,p,'add'),np.c_[q,v,w]
            yield (key,'pairs',p),np.c_[q,np.sin(p*pairs),np.cos(p*pairs)]
        if key=='e':continue
        v,w=z[key+'1'],z[key+'2']
        for label,a in [('1',v),('2',w),('12',np.c_[v,w])]:
            yield (key,label,'add'),np.c_[q,a,a*a]
            yield (key,label,'xprod'),np.c_[q,a,a*a,(x[:,:,None]*a[:,None,:]).reshape(len(x),-1)]
            yield (key,label,'quad'),_quad(np.c_[x,a])
        for label in ('1','1j','2','2j'):
            a=z[key+label]
            for p in (1.,1.7,2.3):
                for mode in ('s','r'):
                    v=np.sin(p*a)/p if mode=='s' else a/(1+(p*a)**2)
                    yield (key,label,p,mode),np.c_[q,v,(a[:,:,None]*v[:,None,:]).reshape(-1,16)]

def _fits(f,y):
    n=len(y); scale=np.maximum(f[:n].std(0),.05);scale[0]=1
    f=f/scale;U,S,V=np.linalg.svd(f[:n],full_matrices=False); sq=S*S; J=f[n:]@V.T
    for link in (0,1,1.4,1.75,2.4,-1.,-1.6):
        if link>0 and abs(y).max()>=link:continue
        t=y if not link else np.arctanh(y/link)*link if link>0 else np.sinh(y/(-link))*(-link)
        u=U.T@t
        def inv(x):return x if not link else link*np.tanh(x/link) if link>0 else -link*np.arcsinh(x/(-link))
        for l in (.006,.03,.15,.55,1.7):
            fac=sq/(sq+l); loo=t-(t-U@(fac*u))/np.maximum(1-U**2@fac,1e-8)
            yield (link,l),inv(J@(S*u/(sq+l))),np.mean(abs(inv(loo)-y))

def _predict(study, x, y, tr, selected):
    s=dict(selected);pred=[]
    for k,f in _library(study,x,y,tr):
        if k in s:
            g={m:p for m,p,_ in _fits(f,y)}
            pred.extend(g[m] for m in s[k] if m in g)
    return np.mean(pred,axis=0)


_SELECTED = [(('bi', '1', 1.7, 's'), [(2.4, 0.15)]), (('bi', '1', 1.0, 'r'), [(2.4, 0.55)]), (('i', '2j', 2.3, 'r'), [(-1.6, 0.15)])]

def solve_batch(study):
    out = _base_solve(study)
    tr = study["train"]
    x = np.concatenate((tr["x"], study["test"]["x"]))
    y = 0.0 * out["y_hat"] + 1.0 * _predict(study, x, tr["y"], tr["traj"], _SELECTED)
    out["y_hat"] = y
    out["diagnostics_vec"][:2] = (y.mean(), y.std())
    return out
