import numpy as np
from scipy.optimize import minimize


def _handoff(q, v, y, traj, configuration):
    i, j = np.triu_indices(4)
    Q = q[:, i]*q[:, j]
    A = np.column_stack((np.ones(len(q)), v))
    C = np.linalg.lstsq(np.column_stack((A,Q)),traj,rcond=None)[0]
    carrier = np.linalg.svd(C[5:],full_matrices=False)[0][:,0]
    carrier /= carrier[1]
    g = Q @ carrier
    C = np.linalg.lstsq(np.column_stack((A,g)),traj,rcond=None)[0]
    L = C[1:5]
    mode, w_mix, w_carrier, w_traj = configuration
    off = ~np.eye(4,dtype=bool)
    keep = np.arange(10)!=1
    extra = mode == 2
    dm = 12 if mode else 0
    dc = 9 if mode else 0
    de = 40 if extra else 0
    size = 11+dm+dc+de
    penal = np.r_[np.full(4,.012),np.full(7,.006),np.full(dm,w_mix),np.full(dc,w_carrier),np.full(de,.24)]**2
    y = np.asarray(y).reshape(-1)
    F = np.ones((len(q),7))
    D = np.ones((len(q),8))

    def unpack(p):
        M = np.eye(4)
        c = carrier.copy()
        if mode:
            M[off] = p[11:23]
            c[keep] += p[23:32]
        E = np.outer(c,p[:4])
        if extra:
            E += p[32:].reshape(10,4)
        return M,c,E

    def objective(p):
        M,c,E = unpack(p)
        z = v@M+Q@E
        F[:,1:5] = z
        F[:,5] = z[:,0]*z[:,2]
        F[:,6] = z[:,1]*z[:,3]
        theta = p[4:11]
        h = np.tanh(F@theta)
        D[:,:7] = F
        D[:,7] = h
        beta = np.linalg.lstsq(D,y,rcond=1e-10)[0]
        ry = D@beta-y
        T = F[:,:5]
        gamma = np.linalg.lstsq(T,traj,rcond=1e-10)[0]
        rt = T@gamma-traj
        rh = ry*beta[-1]*(1-h*h)
        dF = ry[:,None]*beta[:7]+rh[:,None]*theta
        dz = dF[:,1:5].copy()+w_traj*(rt@gamma[1:].T)
        dz[:,0] += dF[:,5]*z[:,2]
        dz[:,2] += dF[:,5]*z[:,0]
        dz[:,1] += dF[:,6]*z[:,3]
        dz[:,3] += dF[:,6]*z[:,1]
        dE = Q.T@dz
        grad = penal*p
        grad[:4] += c@dE
        grad[4:11] += F.T@rh
        if mode:
            grad[11:23] += (v.T@dz)[off]
            grad[23:32] += (dE@p[:4])[keep]
        if extra:
            grad[32:] += dE.ravel()
        return .5*(ry@ry+w_traj*np.sum(rt*rt)+np.dot(penal*p,p)),grad

    rng = np.random.default_rng(130)
    direction = np.array([0.,1.,1.,1.,-1.,0.,0.])
    ds = [np.linalg.solve(L@L.T+r*r*np.eye(4),L@C[5]) for r in (.045,.12,.3)]
    ds += [np.eye(4)[k]*(L[k]@C[5])/(L[k]@L[k]) for k in range(4)]
    starts = []
    for k in range(18):
        d = ds[k%len(ds)] if k < 14 else ds[1]+rng.normal(0,.55,4)
        theta = direction if k < 7 else rng.normal(0,.85,7)
        p = np.r_[d,theta,np.zeros(size-11)]
        if mode and k >= 14:
            p[11:23] = rng.normal(0,.22,12)
        starts.append(p)
    best = None
    for p in starts:
        sol = minimize(objective,p,method='L-BFGS-B',jac=True,options={'maxiter':420,'ftol':2e-10,'gtol':1e-7,'maxls':30,'maxcor':16})
        if best is None or sol.fun < best.fun:
            best = sol
    M,_,E = unpack(best.x)
    return M,E


_CONFIGURATION = (1,.14,1.1,2.)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train,test = study['train'],study['test']
    a,b = _fit_panel_a(study['panel_a']),_fit_panel_b(study['panel_b'])
    q,qt = _inverse_a(train['x'],a),_inverse_a(test['x'],a)
    v,vt = q@b[1:]+b[0],qt@b[1:]+b[0]
    M,E = _handoff(q,v,train['y'],train['traj'],_CONFIGURATION)
    i,j = np.triu_indices(4)
    return {'z':v@M+(q[:,i]*q[:,j])@E,'z_test':vt@M+(qt[:,i]*qt[:,j])@E,'y':train['y'],'traj':train['traj']}
