import numpy as np
from itertools import combinations_with_replacement


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _latent(x, panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    a = np.column_stack((np.ones(len(u)), u))
    keep = np.linalg.norm(v-a@_fit(a,v), axis=1) < 2
    w = []
    for i in range(4):
        j = (i+1)%4
        h = np.column_stack((a[:,0], u[:,i], u[:,j], u[:,i]*u[:,j], u[:,i]**3))
        w.append(_fit(h[keep], v[keep,i]))
    off,g,c,b,d = np.array(w).T
    z = (x-off)/g
    ix = np.arange(4)
    jx = (ix+1)%4
    for _ in range(14):
        zn = z[:,jx]
        r = off + g*z+c*zn+b*z*zn+d*z**3-x
        jac = np.zeros((len(z),4,4))
        jac[:,ix,ix] = g+b*zn+3*d*z*z
        jac[:,ix,jx] = c+b*z
        z -= np.clip(np.linalg.solve(jac, r[...,None])[...,0], -0.75, 0.75)
    return z


def _warp(z, panel):
    u, v = np.asarray(panel['u']), np.asarray(panel['v2'])
    a = np.column_stack((np.ones(len(u)), u))
    def profile(rates):
        h = np.column_stack((a, np.expm1(-rates*u*u)))
        w = _fit(h,v)
        return np.sum((h@w-v)**2), w
    rates = np.full(4, .2934)
    for _ in range(3):
        for j in range(4):
            lo, hi = .25,.34
            for _ in range(16):
                p,q = (2*lo+hi)/3,(lo+2*hi)/3
                rates[j]=p
                ep=profile(rates)[0]
                rates[j]=q
                eq=profile(rates)[0]
                if ep < eq:
                    hi=q
                else:
                    lo=p
            rates[j]=(lo+hi)/2
    w = profile(rates)[1]
    return z + np.expm1(-rates*z*z) @ np.linalg.solve(w[1:5].T,w[5:].T).T


def _poly(z, degree):
    return np.column_stack([np.prod(z[:,j],axis=1) for k in range(1,degree+1) for j in combinations_with_replacement(range(4),k)])


def _temporal_basis(times, traj):
    # Variable projection: optimize rates, solving amplitudes exactly at each step.
    t = np.asarray(times)[:, None]
    rates = np.array([.408, 1.815])
    def profile(r):
        q = -np.expm1(-t*r)
        return q, (q@_fit(q,traj.T)-traj.T).ravel()
    q, r = profile(rates)
    for _ in range(10):
        h = 1.e-4
        J = np.column_stack([(profile(rates+d)[1]-profile(rates-d)[1])/(2*h) for d in h*np.eye(2)])
        step = np.clip(_fit(J,r), -.35, .35)
        if np.linalg.norm(step) < 1.e-8:
            break
        for shrink in (1., .5, .25, .125):
            cand = np.clip(rates-shrink*step, [.1,1.], [.9,4.])
            qq, rr = profile(cand)
            if rr@rr < r@r:
                rates, q, r = cand, qq, rr
                break
        else:
            break
    return q


def _predict(study):
    tr = study['train']
    n = len(tr['x'])
    x = np.vstack((tr['x'],study['test']['x']))
    z = _warp(_latent(x,study['panel_a']),study['panel_c'])
    f = np.column_stack((_poly(z,3), np.tanh(4*z)))
    f = (f-f[:n].mean(0))/np.maximum(f[:n].std(0),1e-8)
    traj = np.asarray(tr['traj'])
    q = _temporal_basis(study['eval_times'],traj)
    y = np.column_stack((tr['y'], _fit(q,traj.T).T))
    mu = y.mean(0)
    a = f[:n]
    g = a.T@a+.003*np.eye(a.shape[1])
    rhs = a.T@(y-mu)
    coef = np.linalg.solve(g,rhs)
    scale = np.sqrt(np.diag(np.linalg.solve(g,np.eye(len(g)))))
    pred = []
    for k in range(y.shape[1]):
        ix = np.argsort(np.abs(coef[:,k])/scale)[-28:]
        pred.append(f[n:,ix]@np.linalg.solve(g[np.ix_(ix,ix)],rhs[ix,k])+mu[k])
    return pred[0], np.column_stack(pred[1:])@q.T


def run_node(inputs: dict) -> dict:
    y, traj = _predict(inputs['__input__'])
    sparse = inputs['node_00']
    panel_y = 2*y-np.asarray(sparse['y_hat'])
    panel_traj = (traj-.35*np.asarray(sparse['state_pred']))/.65
    return {'panel_pred': np.column_stack((panel_y,panel_traj))}
