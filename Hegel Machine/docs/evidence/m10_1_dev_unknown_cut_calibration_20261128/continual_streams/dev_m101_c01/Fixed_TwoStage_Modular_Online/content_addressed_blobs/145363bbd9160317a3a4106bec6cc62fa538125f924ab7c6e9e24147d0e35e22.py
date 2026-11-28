"""Panel-informed kernel regression for scalar and trajectory observations."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize


def _lstsq(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _operators(study):
    a, b, c = (study[k] for k in ('panel_a', 'panel_b', 'panel_c'))
    u, v = a['u'], a['v']
    A = np.c_[np.ones(len(u)), u]
    idx = np.arange(len(u))
    for _ in range(7):
        p = _lstsq(A[idx], v[idx])
        err = np.linalg.norm(A @ p - v, axis=1)
        idx = np.argsort(err)[:3 * len(u) // 4]
    idx = err <= 5 * np.median(err)
    shift, gain = np.array([_lstsq(A[idx][:, [0, j + 1]], v[idx, j]) for j in range(4)]).T
    u = b['u']
    B = _lstsq(np.c_[np.ones(len(u)), u], b['v'])
    u = c['u']
    def fit_operator(f, start, bounds, target):
        def design(w, p):
            return np.c_[np.ones(len(w)), w, f(w, p)]
        def res(p):
            M = design(u, p)
            return (M @ _lstsq(M, target) - target).ravel()
        p = least_squares(res, start, bounds=bounds).x
        coeff = _lstsq(design(u, p), target)
        return lambda w: design(w, p) @ coeff
    f1 = fit_operator(lambda w,p: np.tanh(p[0]*w)+p[1]*np.arctan(p[2]*w),
                      [1.5,.2,5.7], ([.2,0.,.5],[5.,2.,9.]), c['v1'])
    f2 = fit_operator(lambda w,p: np.logaddexp(p[0]*w,-p[0]*w),
                      [1.1], ([.2],[4.]), c['v2'])
    return shift, gain, B, f1, f2


def _kernels(study, x, n):
    shift, gain, B, f1, f2 = _operators(study)
    a = (x - shift) / gain
    b = B[0] + x @ B[1:]
    ab = B[0] + a @ B[1:]
    bi = np.linalg.solve(B[1:].T, (x - B[0]).T).T
    def normalize(w):
        return (w-w[:n].mean(0))/np.maximum(w[:n].std(0), 1e-8)
    def poly(w, degree=4):
        g = normalize(w)
        Q = g @ g.T / g.shape[1]
        return [Q**k for k in range(1, degree+1)]
    def panel(w):
        return poly(f1(w), 3) + poly(f2(w), 3)
    def interact(w):
        return poly(f1(w), 2) + poly(f2(w), 2) + poly(np.c_[w,f1(w),f2(w)], 3)[1:]
    def add(w, pairs=False):
        g = normalize(w)
        R = np.exp(-.5*(g[:,None]-g[None,:])**2).transpose(2,0,1)
        return list(R) + ([R[i]*R[j] for i in range(4) for j in range(i+1,4)] if pairs else [])
    def trig(w):
        return [poly(f(w), 1)[0] for f in (lambda q:np.arctan(2.3*q),lambda q:np.cos(1.28*q),lambda q:np.sin(1.7*q))]
    def radial(w):
        g = normalize(w)
        q = g @ g.T / g.shape[1]
        d = ((g[:, None] - g[None, :])**2).sum(-1)
        return [q, q**2] + [np.exp(-h*d) for h in (.15, .5, 1.5)]
    def warp(w):
        u, v = f1(w), f2(w)
        return radial(u) + radial(v) + poly(np.c_[w,u,v])[1:]
    def products(w):
        q = poly(w, 1)[0]
        k = [poly(f(w), 1)[0] for f in (
            lambda w: np.tanh(1.7*w),
            lambda w: np.logaddexp(1.1*w,-1.1*w),
            lambda w: np.arctan(2.3*w),
        )]
        return k + [q*h for h in k] + [q**2*h for h in k]
    raw = poly(x)
    groups = {
        'a_panel': panel(a), 'abi_panel': panel(np.linalg.solve(B[1:].T,(a-B[0]).T).T),
        'a_interact': interact(a), 'raw_add': add(x, True), 'ab_add': add(ab, True),
        'bi_add': add(bi, True), 'ab_trig': trig(ab) + add(ab, True),
        'joint': panel(a)+add(x)+add(ab)+add(bi), 'plain': [],
    }
    constant = np.ones((len(x),len(x)))
    kernels = {key: np.array([constant] + raw + more) for key,more in groups.items()}
    base = [constant] + raw[:3]
    abi = np.linalg.solve(B[1:].T, (a-B[0]).T).T
    kernels['abi_warp'] = np.array(base + warp(abi))
    kernels['ab_product'] = np.array(base + products(ab))
    return kernels


def _predict(kernels, targets):
    n, m = targets.shape
    K = np.concatenate((kernels[:, :n,:n], np.eye(n)[None]))
    E = K[-1]
    def objective(p):
        weights = np.exp(p)
        C = cho_factor(np.einsum('i,ijk->jk',weights,K),lower=True,check_finite=False)
        A = cho_solve(C, targets, check_finite=False)
        G = m*cho_solve(C,E,check_finite=False)-A@A.T
        loss = .5*np.sum(targets*A)+m*np.log(C[0].diagonal()).sum()
        return loss, .5*weights*np.einsum('jk,ijk->i',G,K)
    init = np.r_[0., np.full(len(K)-2,-3.), -8.]
    p = minimize(objective,init,jac=True,method='L-BFGS-B',bounds=[(-16.,5.)]*(len(K)-1)+[(-16.,0.)],options={'maxiter':160}).x
    weights = np.exp(p)
    C = cho_factor(np.einsum('i,ijk->jk',weights,K),lower=True,check_finite=False)
    return np.einsum('i,ijk->jk',weights[:-1],kernels[:,n:,:n]) @ cho_solve(C,targets,check_finite=False)


def _predictions(study, scalar_keys, state_keys):
    train = study['train']
    n = len(train['x'])
    K = _kernels(study, np.r_[train['x'],study['test']['x']], n)
    y = np.asarray(train['y'])
    z = np.asarray(train['traj'])
    ym, ys = y.mean(), y.std()
    zm, zs = z.mean(0), z.std(0)
    zstd = (z-zm)/zs
    V = np.linalg.svd(zstd,full_matrices=False)[2][:2]
    T = zstd @ V.T
    ts = T.std(0)
    scalar = {k: _predict(K[k],((y-ym)/ys)[:,None])[:,0]*ys+ym for k in scalar_keys}
    state = {k: (_predict(K[k],T/ts)*ts)@V*zs+zm for k in state_keys}
    return scalar,state


def solve_batch(study: dict) -> dict:
    scalar, state = _predictions(study, ('a_panel','raw_add','abi_panel','abi_warp'), ('a_interact','ab_trig','bi_add','ab_product'))
    y = .5*scalar['a_panel']+.25*scalar['raw_add']+.25*scalar['abi_panel']
    z = .4*state['a_interact']+.35*state['ab_trig']+.25*state['bi_add']
    y = .5*(y+scalar['abi_warp'])
    z = .5*(z+state['ab_product'])
    end = z[:,-1]
    return {'y_hat':y,'state_pred':z,'diagnostics_vec':np.array([y.mean(),y.std(),end.mean(),end.std()])}
