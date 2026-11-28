

def _dictionary(z, c, d, mode):
    from itertools import combinations_with_replacement
    def mon(x, degree):
        return np.column_stack([np.prod(x[:, j], axis=1) for j in combinations_with_replacement(range(x.shape[1]), degree)])
    base = np.column_stack((z,c,d))
    blocks = [base, mon(base,2)]
    if mode in (1,3):
        blocks.extend([mon(z,3), mon(z,4), np.sin(1.5*z), np.tanh(z), z/(1+.7*abs(z)), z*z/(1+.25*z*z)])
    if mode in (2,3):
        pairs = _pairs(z)
        for x in (z,c,d):
            blocks.append((x[:,:,None]*pairs[:,None,:]).reshape(len(z),-1))
        blocks.extend([mon(c,3), mon(d,3), c/(1+.25*c*c), d/(1+.25*d*d)])
    return np.column_stack(blocks)


def _feature_metric(f, y, n, prior):
    # Multi-response empirical Bayes: feature variances estimated from training rows only.
    from scipy.linalg import cho_factor, cho_solve
    mean = f[:n].mean(0)
    std = np.maximum(f[:n].std(0),1e-7)
    f = np.clip((f-mean)/std,-20,20)
    t = y-y.mean(0)
    t /= np.maximum(t.std(0),1e-7)
    A = f[:n]
    power = np.full(A.shape[1], 1/A.shape[1]); noise = .025
    eye = np.eye(n)
    for _ in range(105):
        cov = (A*power)@A.T+noise*eye
        fac = cho_factor(cov,check_finite=False)
        H = cho_solve(fac,A,check_finite=False)
        T = cho_solve(fac,t,check_finite=False)
        denom = np.sum(A*H,axis=0).clip(1e-10)
        num = np.mean((A.T@T)**2,axis=1)
        power = np.maximum(1e-10, power*np.clip(num/denom,.15,8)**.72)
        dn = np.trace(cho_solve(fac,eye,check_finite=False))
        noise = max(prior,noise*np.clip(np.mean(np.sum(T*T,axis=0))/dn,.2,5)**.72)
    return f, power, noise


def _library(study):
    z = _coordinates(study)[0]
    c,d = _maps(study, (z,))[0]
    return z,c,d


def _representation(study, config, library=None):
    z,c,d = _library(study) if library is None else library
    n = len(study['train']['x'])
    k, a, b, g = config
    if k < 0:
        blocks = [z,c,d,_pairs(z),z*z/(1+.25*z*z)]
        ids = {-1:(0,1,2),-2:(0,1,4),-3:(0,2,3),-4:(0,1,2,3,4)}[k]
        multipliers = [1.,a,b] + ([.45,.4] if len(ids)>3 else [])
        pieces = []
        for i,weight in zip(ids,multipliers):
            q = blocks[i]; pieces.append(weight*(q-q[:n].mean(0))/np.maximum(q[:n].std(0),.1))
        return g*np.column_stack(pieces)
    y = np.column_stack((study['train']['y'],study['train']['traj']))
    y = y[:, list(dict.fromkeys([0,1, min(3,y.shape[1]-1),y.shape[1]-1]))]
    f,p,noise = _feature_metric(_dictionary(z,c,d,k),y,n,a)
    sel = p > max(p.max()*b,1e-9)
    p = p[sel]
    return f[:,sel]*(np.sqrt(p/p.sum())*np.sqrt(len(p))*g)
