import numpy as np
from scipy.optimize import minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _a_basis(x, j):
    h, q = x[:, j], x[:, (j + 1) % 4]
    return np.column_stack((np.ones(len(x)), h, q, h*q, h**3))


def _inverse(panel, x):
    u, v = panel['u'], panel['v']
    a = _linear(u)
    w = np.ones(len(u))
    for _ in range(10):
        r = np.linalg.norm(v-a@_fit(a*w[:,None],v*w[:,None]),axis=1)
        w = 1/np.sqrt(1+(2*r/max(np.median(r),1e-10))**2)
    k = np.argsort(r)[:-3]
    c = np.array([_fit(_a_basis(u[k], j), v[k,j]) for j in range(4)])
    z = x.copy()
    j = np.arange(4)
    qj = (j+1)%4
    for _ in range(12):
        q = z[:,qj]
        f = c[:,0]+c[:,1]*z+c[:,2]*q+c[:,3]*z*q+c[:,4]*z**3
        jac = np.zeros((len(x),4,4))
        jac[:,j,j] = c[:,1]+c[:,3]*q+3*c[:,4]*z*z
        jac[:,j,qj] = c[:,2]+c[:,3]*z
        z -= np.linalg.solve(jac,(f-x)[...,None])[...,0]
    return z


def _c1(x):
    return np.column_stack((np.ones(len(x)),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]))


def _b(x):
    return np.column_stack((_linear(x),np.sin(2.8*x)))


def _assays(s):
    p = s['panel_c']
    u = p['u']
    c = _fit(_c1(u), p['v1'])
    def phi(t, a):
        return np.c_[_linear(t), t*t/(1+a*t*t)]
    def loss(a):
        f = phi(u,a)
        return np.mean((f@_fit(f,p['v2'])-p['v2'])**2)
    a = minimize_scalar(loss, bounds=(.15,.8), method='bounded').x
    d = _fit(phi(u,a),p['v2'])
    p = s['panel_b']; u,v = p['u'],p['v']
    b = _fit(_b(u),v)
    def B(t): return _b(t)@b
    return B, lambda t:_c1(t)@c, lambda t:phi(t,a)@d


def _kernels(s,x,z,n):
    B,C,D = _assays(s)
    kernels = []
    def add(f):
        f = np.asarray(f).reshape(len(x),-1)
        f = f-f[:n].mean(0)
        k = f@f.T
        scale = np.mean(np.diag(k)[:n])
        if scale > 1e-12:
            kernels.append(k/scale)
    def cross(a,b):
        return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
    for t in (x,z):
        b,c,d = B(t),C(t),D(t)
        for f in (t,b,c,d,t**2,t**3,np.sin(2*t),np.cos(2*t),np.sin(4*t),np.cos(4*t)):
            add(f)
        for i,a in enumerate((t,b,c,d)):
            for j,q in enumerate((t,b,c,d)):
                if j<i: continue
                add(cross(a,q)); add(a*q)
                if j>i:
                    add(a/(1+q*q)); add(q/(1+a*a))
                    add(np.tanh(a+q)); add(np.tanh(a-q))
        for a in (c,d,c+d,c*d):
            for w in (-1.,-.45,.45,1.):
                add(B(t+w*a)); add(B(a+w*t)); add(B(b+w*a))
        for a in (t,np.c_[t,b],np.c_[t,b,c,d]):
            a = (a-a[:n].mean(0))/np.maximum(a[:n].std(0),.05)
            h = a@a.T/a.shape[1]; e = np.maximum(np.diag(h)[:,None]+np.diag(h)-2*h,0)
            for w in (.12,.45,1.):
                k = np.exp(-w*e)
                k = k-k[:n].mean(0)-k[:,:n].mean(1)[:,None]+k[:n,:n].mean()
                kernels.append(k/np.mean(np.diag(k)[:n]))
        # Coordinate-wise assay coupling: test bounded shifts and gates before B.
        for a in (c,d,c+d):
            for h in (np.tanh(a), a/(1+a*a), np.sin(a)):
                for w in (-.65,.35,.7):
                    r = t+w*h
                    add(B(r)); add(B(r)-b); add(B(r)*h)
            for w in (.35,.9):
                add(B(t/(1+w*a*a))); add(b/(1+w*a*a))
                add(np.sin(w*a)*b); add(np.sin(t+w*a))
        for a in (c,d):
            for shift in (1,-1):
                h = np.roll(a,shift,axis=1)
                add(B(t+.55*h)); add(B(t+.55*np.tanh(h))); add(b*h)
        # Projected assay channels can also contribute independently.
        for a in (t,b,c,d):
            for w in (.65,1.4,2.8):
                add(np.tanh(w*a)); add(np.sin(w*a)); add(np.cos(w*a))
        for a in (np.c_[t,b,d],np.c_[t,b,c],np.c_[b,d]):
            a = (a-a[:n].mean(0))/np.maximum(a[:n].std(0),.05)
            h = a@a.T/a.shape[1]
            e = np.maximum(np.diag(h)[:,None]+np.diag(h)-2*h,0)
            for w in (.08,.25,.7):
                k = (1+w*e)**-1
                k = k-k[:n].mean(0)-k[:,:n].mean(1)[:,None]+k[:n,:n].mean()
                kernels.append(k/np.mean(np.diag(k)[:n]))
    return np.asarray(kernels)


def _learn_kernel(ks,Y,mode):
    from scipy.linalg import cho_factor,cho_solve
    from scipy.optimize import minimize
    n = len(Y)
    p = len(ks)
    tri = np.tril_indices(n)
    flat = np.ascontiguousarray(ks[:,:n,:n][:,tri[0],tri[1]])
    ident = np.eye(n)
    multiplicity = np.where(tri[0]==tri[1],1.,2.)
    prior = (.35 if mode==0 else 1.4 if mode==1 else .075)
    def objective(w):
        a = np.zeros((n,n)); a[tri] = w[:-1]@flat
        a = a+np.tril(a,-1).T+w[-1]*ident
        fact = cho_factor(a,check_finite=False)
        coef = cho_solve(fact,Y,check_finite=False)
        inv = cho_solve(fact,ident,check_finite=False)
        derivative = (Y.shape[1]*inv-coef@coef.T)[tri]*multiplicity
        value = 2*Y.shape[1]*np.log(fact[0].diagonal()).sum()+np.sum(Y*coef)+prior*np.sum(w[:-1])
        return value,np.r_[flat@derivative+prior,np.trace(Y.shape[1]*inv-coef@coef.T)]
    initial = np.r_[np.full(p,1/p),.025]
    opt = minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(0.,20.)]*p+[(.00015,1.)],options={'maxiter':240,'ftol':1e-8})
    return np.tensordot(opt.x[:-1],ks,axes=1), opt.x[-1]


_MODE = 2


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']; tr = s['train']; x = np.r_[tr['x'],s['test']['x']]; n = len(tr['x'])
    z = _inverse(s['panel_a'],x)
    ks = _kernels(s,x,z,n)
    t = tr['traj']; pc = t@np.linalg.svd(t,full_matrices=False)[2][0]
    y = np.column_stack((tr['y'],pc)); y = (y-y.mean(0))/np.maximum(y.std(0),1e-8)
    if _MODE==3:
        k = ks.mean(0); noise = .007
    else:
        k,noise = _learn_kernel(ks,y,_MODE)
    # Map the learned covariance to explicit features for the unchanged ridge.
    val,vec = np.linalg.eigh(k); keep = val>1e-10
    f = vec[:,keep]*np.sqrt(val[keep]*.0012/noise)
    empty = np.empty((len(x),0))
    return {'g':np.c_[np.ones(len(x)),f],'fd':empty,'cd':empty,'y':tr['y'],'traj':t}
