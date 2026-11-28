import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def _linear(x):
    return np.column_stack((np.ones(len(x)), x))


def _panel_features(panel, z):
    u = panel['u']
    features = []
    for key, bounds, nonlinear in (
        ('v1', (.1, 2.), lambda x,k: x/(1+k*abs(x))),
        ('v2', (.02, 1.), lambda x,k: x*x/(1+k*x*x)),
    ):
        def fit(k):
            a = np.column_stack((_linear(u), nonlinear(u,k)))
            coef = np.linalg.lstsq(a,panel[key],rcond=None)[0]
            return np.mean((a@coef-panel[key])**2), coef
        k = minimize_scalar(lambda k: fit(k)[0], bounds=bounds, method='bounded').x
        features.append(np.column_stack((_linear(z), nonlinear(z,k)))@fit(k)[1])
    return features


def _blocks(z,b,c,d,x,f):
    n = len(z)
    def products(a,b):
        return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
    # The first block spans the calibrated affine readout. Other blocks
    # represent departures from it rather than duplicating its directions.
    yield np.ones((n,1))
    yield z
    for v in (x,f,c,d):
        yield v
    for a in (z,b,c,d,x,f):
        yield a*a
        yield a**3
        yield np.sin(2*a)
        yield np.cos(2*a)
        yield np.tanh(2*a)
        yield a/(1+abs(a))
        yield products(a,a)
    for a,g in ((z,b),(z,c),(z,d),(b,c),(b,d),(c,d),(c,c),(d,d),(x,f),(z,x),(z,f),(b,f)):
        yield products(a,g)
        yield np.column_stack((products(a*a,g),products(a,g*g)))
        q = a[:,:,None]+g[:,None,:]
        r = a[:,:,None]-g[:,None,:]
        yield np.column_stack((np.sin(q).reshape(n,-1),np.sin(r).reshape(n,-1),np.cos(q).reshape(n,-1),np.cos(r).reshape(n,-1)))
        yield np.column_stack((products(a,1/(1+g*g)),products(1/(1+a*a),g)))
    for center in (z,np.column_stack((z,b,c,d))):
        yield ('distance', center)


def _predict(z,b,c,d,x,f,y):
    n = len(y)
    m = len(z)
    affine = _linear(z)
    kernels = []
    for i, block in enumerate(_blocks(z,b,c,d,x,f)):
        if isinstance(block,tuple):
            q = block[1]; q = (q-q[:n].mean(0))/np.maximum(q[:n].std(0),1e-8)
            distance = cdist(q,q[:n])/np.sqrt(q.shape[1])
            for rate in (.7,1.5,3.):
                r = rate*distance
                kernels.append((1+r+r*r/3)*np.exp(-r))
            continue
        if i > 1:
            block = block-affine@np.linalg.lstsq(affine[:n],block[:n],rcond=None)[0]
        sd = block[:n].std(0) if i else np.ones(1)
        keep = sd > 1e-7
        if not np.any(keep):
            continue
        q = block[:,keep]/sd[keep]
        kernels.append((q@q[:n].T)/q.shape[1])
    kernels.append(np.eye(m,n))
    K = np.asarray(kernels)
    ktrain = K[:,:n].copy().reshape(len(K),-1)
    ktest = K[:,n:].copy().reshape(len(K),-1)
    prediction = []
    for target in y.T:
        scale = max(target.std(),1e-8)
        t = (target-target.mean())/scale
        def objective(theta):
            weights = np.exp(theta)
            G = (weights@ktrain).reshape(n,n)
            factor, lower = cho_factor(G,lower=True,check_finite=False)
            alpha = cho_solve((factor,lower),t,check_finite=False)
            inv = cho_solve((factor,lower),np.eye(n),check_finite=False)
            penalty = .03*np.sum((theta-init)**2)
            loss = np.log(np.diag(factor)).sum()+.5*t@alpha+penalty
            grad = .5*weights*(ktrain@(inv-np.outer(alpha,alpha)).ravel())+.06*(theta-init)
            return loss,grad
        init = np.full(len(K),-6.); init[:2] = 0; init[-1] = -6.
        opt = minimize(objective,init,jac=True,method='L-BFGS-B',bounds=[(-15,5)]*(len(K)-1)+[(-12,1)],options={'maxiter':120,'ftol':1e-8})
        w = np.exp(opt.x)
        alpha = np.linalg.solve((w@ktrain).reshape(n,n),t)
        prediction.append(target.mean()+scale*((w@ktest).reshape(m-n,n)@alpha))
    return np.column_stack(prediction)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['y'])
    z = inputs['node_02']['z']
    x = np.vstack((train['x'],study['test']['x']))
    f = inputs['node_00']['f']
    b = f+(z-x)@np.linalg.lstsq(_linear(x[:n]),f[:n],rcond=None)[0][1:]
    c,d = _panel_features(study['panel_c'],z)
    basis = np.linalg.svd(train['traj'],full_matrices=False)[2][:2]
    target = np.column_stack((train['y'],train['traj']@basis.T))
    p = _predict(z,b,c,d,x,f,target)
    yh = p[:,0]
    sp = p[:,1:]@basis
    return {'y_hat':yh,'state_pred':sp,'diagnostics_vec':np.array([yh.mean(),yh.std(),sp[:,-1].mean(),sp[:,-1].std()])}
