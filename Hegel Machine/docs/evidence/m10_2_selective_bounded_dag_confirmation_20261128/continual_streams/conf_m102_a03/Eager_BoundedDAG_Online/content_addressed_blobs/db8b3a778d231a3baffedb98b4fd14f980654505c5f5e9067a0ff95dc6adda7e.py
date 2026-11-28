"""Panel-calibrated scalar and trajectory prediction."""
import numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-11)[0]


def _roll(x):
    return np.roll(x, -1, axis=1)


def _coords(study, x):
    u, v = [np.asarray(study['panel_a'][k], float) for k in ('u','v')]
    good = np.linalg.norm(v, axis=1) < 5
    u, v = u[good], v[good]
    def features(z):
        return np.stack((np.ones_like(z), z, _roll(z), z*_roll(z), z**3), axis=2)
    a = features(u)
    c = np.stack([_ls(a[:, j], v[:, j]) for j in range(4)], axis=1)
    z = np.array(x, dtype=float, copy=True)
    for _ in range(28):
        delta = np.sum(features(z)*c.T, axis=2) - x
        jac = np.eye(4)[None,:,:]*(c[1]+c[3]*_roll(z)+3*c[4]*z*z)[:,:,None]
        jac += np.roll(np.eye(4), 1, axis=1)[None,:,:]*(c[2]+c[3]*z)[:,:,None]
        z -= np.clip(np.linalg.solve(jac, delta[...,None])[...,0], -.5, .5)
    return z


def _poly(h, degree):
    return np.column_stack([np.ones(len(h))] + [np.prod(h[:, idx], axis=1) for d in range(1,degree+1) for idx in combinations_with_replacement(range(h.shape[1]),d)])


def _panel_b(study, z):
    u, v = [np.asarray(study['panel_b'][k], float) for k in ('u','v')]
    def f(x, w):
        return np.c_[np.ones(len(x)), x, np.sin(w*x)]
    w = least_squares(lambda w: (f(u,w)@_ls(f(u,w),v)-v).ravel(), [2.9], bounds=(1., 4.)).x
    return f(z,w)@_ls(f(u,w),v)


from itertools import combinations
from scipy.linalg import cho_factor, cho_solve


def _readout_features(study, z, n):
    groups = {}
    def add(name, a):
        a = np.array(a, dtype=float, copy=True).reshape(len(z), -1)
        a -= a[:n].mean(0)
        a /= np.maximum(np.sqrt(np.mean(a[:n] ** 2, axis=0)), 1e-9)
        groups[name] = a
    def expand(name, a):
        add(name+'L', a)
        add(name+'Q', a*a)
        add(name+'X', np.column_stack([a[:,i]*a[:,j] for i,j in combinations(range(4),2)]))
    def cross(a,b):
        return (a[:,:,None]*b[:,None,:]).reshape(len(z),-1)
    h0 = _panel_b(study, z)
    h = h0-h0[:n].mean(0)
    expand('H',h)
    expand('Z',z)
    add('HZ',cross(h,z))
    coeff = _ls(np.c_[np.ones(n),z[:n]],h[:n]); r = h-coeff[0]-z@coeff[1:]
    r = r@np.linalg.svd(r[:n],full_matrices=False)[2].T
    expand('R',r); add('RZ',cross(r,z))
    for tag, a in [('H',h),('Z',z)]:
        add(tag+'D3',_poly(a,3)[:,15:])
        add(tag+'T',np.column_stack([np.prod(a[:,i],axis=1) for i in combinations(range(4),3)]))
    # Compact auxiliary operator bases estimated without any downstream labels.
    for key in ('v1','v2'):
        u,v = [study['panel_c'][k] for k in ('u',key)]
        if key=='v1':
            def f(x): return np.c_[np.ones(len(x)),x,np.arctan(2.43*x)]
        else:
            def f(x): return np.c_[np.ones(len(x)),x,x*x,np.cos(2.569*x)]
        c=_ls(f(u),v)
        a=f(z)@c; b=f(h0)@c
        a-=a[:n].mean(0); b-=b[:n].mean(0)
        for suffix, val in [('Z',a),('H',b)]:
            tag=key+suffix
            expand(tag,val); add(tag+'joint',cross(val,h))
    for tag,g in [('rat',z**3/(1+.65*z*z)),('sin',z-.4*np.sin(2.89*z)),('soft',np.tanh(1.25*z))]:
        expand(tag,g); add(tag+'mix',cross(z,g))
    # Mixed-basis polynomial kernels keep higher-order interactions shrunk as a group.
    for tag,a in [('A',np.c_[groups['HL'],groups['ZL']]),('B',groups['HL']),('C',np.c_[groups['HL'],groups['v1ZL'],groups['v2ZL']])]:
        for power in (2,3,4):
            a0=a/np.sqrt(a.shape[1]); add(tag+'p'+str(power),_poly(a0,power)[:,_poly(a0,power-1).shape[1]:])
    return groups


def _stable_predict(study, z, Y, n):
    """Fit smooth readouts with explicit native/operator interactions and blocked validation."""
    groups = _readout_features(study, z, n)
    kernel = {k: a @ a.T / a.shape[1] for k, a in groups.items()}
    z0 = groups['ZL']; h0 = groups['HL']
    v = study['panel_b']['v']; u = study['panel_b']['u']
    # Recover the componentwise near-odd source in panel B, with an optional
    # smooth residual correction tied to the panel measurements.
    def base(u, w):
        return np.c_[np.ones(len(u)), u, np.sin(w*u)]
    def panel_res(w):
        F = base(u,w); return (F@_ls(F,v)-v).ravel()
    w = least_squares(panel_res,[2.9],bounds=(1.,4.)).x
    coef = _ls(base(u,w),v)
    residual = v-base(u,w)@coef
    for scale in (.25,.8):
        d1 = ((u[:,None,:]-u[None,:,:])**2).sum(2)
        d2 = ((z[:,None,:]-u[None,:,:])**2).sum(2)
        a = base(z,w)@coef + np.exp(-scale*d2)@np.linalg.solve(np.exp(-scale*d1)+.04*np.eye(len(u)),residual)
        a -= a[:n].mean(0); a /= np.maximum(a[:n].std(0), 1e-8)
        dot = a@a.T/4
        kernel['bc'+str(scale)] = dot + dot**2 + dot*(z0@z0.T/4)
    H = sum(kernel[k] for k in ('HL','HQ','HX'))
    Z = sum(kernel[k] for k in ('ZL','ZQ','ZX'))
    M = kernel['HZ']
    D = sum(kernel[k] for k in ('v1ZL','v1ZQ','v1ZX','v2ZL','v2ZQ','v2ZX'))
    S = sum(kernel[k] for k in ('sinL','sinQ','sinX','sinmix'))
    R = sum(kernel[k] for k in ('ratL','ratQ','ratX','ratmix'))
    C = sum(kernel[k] for k in ('v1HL','v1HQ','v1HX','v2HL','v2HQ','v2HX'))
    # A single observed coordinate may be relevant in both even and odd
    # parity channels. Keep cross-coordinate degrees separate.
    def sparse_tensor(a):
        x = a / np.maximum(a[:n].std(0), 1e-9)
        k = x @ x.T / x.shape[1]
        d = ((x[:,None,:]-x[None,:,:])**2).sum(2)/x.shape[1]
        return k, d
    hk,hd = sparse_tensor(h0); zk,zd = sparse_tensor(z0)
    families = []
    families.append(('native',H+Z+M))
    for w0 in (.08,.35,1.,3.):
        B=H+w0*(Z+M)
        families.extend([('mix',B),('rat',B+w0*R),('sin',B+w0*S),('aux',B+w0*(D+C)),('panel',B+w0*(kernel['bc0.25']+kernel['bc0.8'])),('branch',B+w0*(kernel['v1Zjoint']+kernel['v2Zjoint']+kernel['v1Hjoint']+kernel['v2Hjoint']))])
        for w1 in (.025,.1):
            families.append(('poly',B+w1*(hk**3+zk**3+(hk+zk)**3/8)))
        for gamma in (.1,.35,.9):
            families.append(('kernel',B+.35*np.exp(-gamma*hd)+w0*np.exp(-gamma*zd)+w0*np.exp(-gamma*(hd+zd)/2)))
    for w0 in (.1,.5,2.):
        families.append(('interaction',H+.2*Z+w0*(kernel['RZ']+kernel['RQ']+kernel['RX'])))
    # Random folds plus coordinate-tail blocks penalize unstable extrapolation.
    rng = np.random.default_rng(326)
    ids = np.arange(n)
    folds = []
    for _ in range(2):
        for val in np.array_split(rng.permutation(n),4):
            folds.append((np.setdiff1d(ids,val),val,1.))
    for j in range(z.shape[1]):
        order=np.argsort(z[:n,j]); val=np.r_[order[:n//9],order[-(n//9):]]
        folds.append((np.setdiff1d(ids,val),val,.45))
    def predict(K,reg,fit,ev):
        A=K[np.ix_(fit,fit)] + reg*np.eye(len(fit))
        cf=cho_factor(A,check_finite=False)
        return K[np.ix_(ev,fit)]@cho_solve(cf,Y[fit],check_finite=False)
    rows=[]; ev=np.arange(n,len(z)); regs=(.00004,.00015,.0005,.0015,.004,.01,.025)
    for name,K0 in families:
        K=K0+1
        for reg in regs:
            err=np.zeros(2); weight=0.
            for fit,val,fw in folds:
                e=abs(predict(K,reg,fit,val)-Y[val]); err+=fw*np.array([e[:,0].mean(),e[:,1:].mean()]);weight+=fw
            rows.append((err/weight,name,predict(K,reg,ids,ev)))
    P=[]
    for i in (0,1):
        rows.sort(key=lambda r:r[0][i]); seen=set(); best=[]
        for e,name,p in rows:
            if name in seen: continue
            seen.add(name); best.append((e[i],p))
            if len(best)>=5:break
        e=np.array([r[0] for r in best]); W=np.exp(-(e-e.min())/(.08*max(e.min(),1e-6))); W/=W.sum()
        P.append(np.tensordot(W,np.array([r[1] for r in best]),axes=1))
    return np.column_stack((P[0][:,0],P[1][:,1:]))


def solve_batch(study):
    tr, te = study['train'], study['test']
    n=len(tr['x']); z=_coords(study,np.concatenate((tr['x'],te['x']),axis=0))
    target=np.column_stack((tr['y'],tr['traj']))
    pred=_stable_predict(study,z,target,n)
    y,t=pred[:,0],pred[:,1:]
    return {'y_hat':y,'state_pred':t,'diagnostics_vec':np.array([y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std()])}
