"""Invert the reference calibration and predict the response and relaxation modes."""
from itertools import combinations, combinations_with_replacement
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    nxt = np.roll(u, -1, axis=1)
    basis = [np.column_stack((np.ones(len(u)), x, z, x*z, x**3))
             for x, z in zip(u.T, nxt.T)]
    ids = np.arange(len(u))
    for _ in range(8):
        coef = np.column_stack([np.linalg.lstsq(f[ids], v[ids, j], rcond=None)[0]
                                for j, f in enumerate(basis)])
        residual = np.column_stack([f @ coef[:, j] for j, f in enumerate(basis)]) - v
        ids = np.argsort(np.linalg.norm(residual, axis=1))[:len(u)-3]
    return coef


def _inverse(x, c):
    def residual(z, row):
        w = np.roll(z, -1)
        return c[0] + c[1]*z + c[2]*w + c[3]*z*w + c[4]*z**3 - row
    return np.array([least_squares(residual, (row-c[0])/c[1], args=(row,),
                                   gtol=1e-11).x for row in x])


def _modes(t, traj):
    def basis(r):
        return -np.expm1(-t[:, None]*r)
    def project(h):
        return np.linalg.lstsq(h, traj.T, rcond=None)[0]
    def residual(r):
        h = basis(r)
        return (h @ project(h) - traj.T).ravel()
    r = least_squares(residual, [.5, 2.], bounds=(.01, 10.)).x
    h = basis(np.sort(r))
    return h, project(h).T


def _groups(z):
    gr=[z]; nonlinear=[]
    for a in z.T:
        g=[a*a,a**3,a**4]
        for k in (.5,1.,1.5,2.,2.5,3.,3.5,4.):
            g.extend((np.sin(k*a),np.cos(k*a),np.tanh(k*a),1/(1+k*a*a),a/(1+k*a*a),np.sqrt(1+k*a*a)))
        nonlinear.append(np.column_stack(g))
    gr.extend(nonlinear)
    for i,j in combinations(range(4),2):
        a,b=z[:,i],z[:,j]; w=a*b
        gr.append(np.column_stack([w]+[w*h for h in (a,b,a*a,b*b,w)]+[f(k*w) for k in (.7,1.5,2.5,3.5) for f in (np.sin,np.cos,np.tanh)]))
        gr.append(nonlinear[i]*b[:,None]);gr.append(nonlinear[j]*a[:,None])
    gr.append(np.column_stack([np.prod(z[:,p],axis=1) for n in (3,4) for p in combinations_with_replacement(range(4),n) if len(set(p))>=3]))
    return gr

def _covariances(gs,a):
    k=[]
    for g in gs:
        f=(g-g[a].mean(0))/np.maximum(g[a].std(0),1e-12)
        k.append(f@f.T/f.shape[1])
    return np.array(k)

def _predict(k,a,b,y,pen):
    ya=y[a]; ya=(ya-ya.mean())/ya.std(); ks=k[:,a][:,:,a]; ks=np.concatenate((ks,np.eye(len(a))[None])); flat=np.ascontiguousarray(ks.reshape(len(ks),-1)); prefer=np.r_[0,np.full(len(ks)-2,-3.),-8.]
    def objective(h):
        v=np.exp(h); c=cho_factor((v@flat).reshape(len(a),len(a))); w=cho_solve(c,ya); ci=cho_solve(c,np.eye(len(a))); diff=h-prefer
        return .5*ya@w+np.log(np.diag(c[0])).sum()+.5*pen*(diff@diff),.5*v*(flat@(ci-np.outer(w,w)).ravel())+pen*diff
    fit=minimize(objective,prefer,jac=True,method='L-BFGS-B',bounds=[(-13,5)]*(len(ks)-1)+[(-13,-3)],options={'maxiter':120}); w=np.exp(fit.x); c=cho_factor((w@flat).reshape(len(a),len(a)))
    return y[a].mean()+y[a].std()*np.einsum('k,kij,j->i',w[:-1],k[:,b][:,:,a],cho_solve(c,ya))



def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    z = _inverse(np.vstack((train['x'], test['x'])), _calibrate(study['panel_a']))
    h, modes = _modes(study['eval_times'], train['traj'])
    a = np.arange(len(train['x']))
    b = np.arange(len(a), len(z))
    kernels = _covariances(_groups(z), a)
    targets = np.column_stack((train['y'], modes))
    prediction = np.column_stack([_predict(kernels, a, b, v, .15) for v in targets.T])
    y = prediction[:, 0]
    state = prediction[:, 1:] @ h.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
