"""Scalar kernel transfer and separate trajectory-mode regressions."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _ridge(f, y, lam):
    k = f @ f[:len(y)].T
    return k[len(y):] @ cho_solve(cho_factor(k[:len(y)] + lam*np.eye(len(y))), y)


def _gp(x, n, y):
    y0 = y.mean()
    scale = max(y.std(), 1e-12)
    u = (y-y0)/scale
    dist = (x[:, None, :] - x[None, :, :])**2
    d = x.shape[1]
    def terms(ds, p):
        e = -ds*np.exp(p[:d])
        joint = np.exp(e.sum(axis=2))
        add = np.exp(e+p[d])
        return e, joint, add
    def loss(p):
        e, j, a = terms(dist[:n,:n], p)
        noise = np.exp(p[-1])
        fact = cho_factor(j+a.sum(axis=2)+noise*np.eye(n))
        v = cho_solve(fact, u)
        g = .5*(cho_solve(fact,np.eye(n))-np.outer(v,v))
        grad = np.einsum('ij,ijk->k',g,e*(j[:,:,None]+a))
        return .5*u@v+np.log(np.diag(fact[0])).sum(), np.r_[grad,np.sum(g*a.sum(2)),noise*np.trace(g)]
    p = minimize(loss,np.r_[np.zeros(d),-2.,-9.],jac=True,method='L-BFGS-B',bounds=[(-11,7)]*d+[(-10,5),(-15,-2)]).x
    _, j, a = terms(dist[:,:n], p)
    k = j+a.sum(axis=2)
    return y0+scale*k[n:]@cho_solve(cho_factor(k[:n]+np.exp(p[-1])*np.eye(n)),u)


def run_node(inputs: dict) -> dict:
    study, parent = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['x'])
    y = .8*_ridge(parent['hybrid'],train['y'],.007)+.2*_ridge(parent['raw'],train['y'],.012)
    x = np.concatenate((train['x'],study['test']['x']))
    z = train['traj']
    basis = np.linalg.svd(z,full_matrices=False)[2][:2]
    state = np.column_stack([_gp(x,n,a) for a in (z@basis.T).T])@basis
    return {'y_hat':y,'state_pred':state,'diagnostics_vec':np.array([y.mean(),y.std(),state[:,-1].mean(),state[:,-1].std()])}
