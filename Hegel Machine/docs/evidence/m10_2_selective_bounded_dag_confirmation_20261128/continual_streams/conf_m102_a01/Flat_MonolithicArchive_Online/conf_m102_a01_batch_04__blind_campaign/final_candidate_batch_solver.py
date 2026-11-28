"""Panel-informed covariance models and a shared low-rank trajectory predictor."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    # Pairwise slopes are insensitive to a minority of contaminated readings.
    p = study['panel_a']
    offset, gain = [], []
    for u, v in zip(p['u'].T, p['v'].T):
        i, j = np.triu_indices(len(u), 1)
        d = u[i] - u[j]
        good = np.abs(d) > .08
        slope = np.median((v[i[good]] - v[j[good]]) / d[good])
        bias = np.median(v - slope * u)
        error = np.abs(v - bias - slope * u)
        keep = error < max(.001, 4 * np.median(error))
        if np.count_nonzero(keep) >= 2:
            bias, slope = np.linalg.lstsq(_affine(u[keep]), v[keep], rcond=None)[0]
        offset.append(bias)
        gain.append(slope)
    z = (x - offset) / gain
    p = study['panel_b']
    b = np.linalg.lstsq(_affine(p['u']), p['v'], rcond=None)[0]
    h = np.linalg.solve(b[1:].T, (z - b[0]).T).T
    return z, h


def _polynomial(z, n, target):
    features = [np.ones(len(z))]
    degrees = [0]
    for d in range(1, 6):
        for idx in combinations_with_replacement(range(z.shape[1]), d):
            features.append(np.prod(z[:, idx], axis=1))
            degrees.append(d)
    f = np.column_stack(features)
    f /= np.maximum(np.sqrt(np.mean(f[:n]**2, axis=0)), 1e-10)
    d = np.array(degrees)
    penalty = np.where(d <= 1, 1e-9, .025 * 5.0**(d - 2))
    g = f / np.sqrt(penalty)
    return (g[n:] @ g[:n].T) @ cho_solve(cho_factor(np.eye(n) + g[:n] @ g[:n].T), target)


def _kernel(x, z, n, target, length):
    g = z @ z.T
    diag = g.diagonal()
    d = np.maximum(diag[:, None] + diag[None, :] - 2 * g, 0)
    mats = np.stack((g**2, np.exp(-d / (2 * length**2))))
    mats /= np.maximum(np.sqrt(np.mean(mats[:, :n, :n]**2, axis=(1, 2))), 1e-12)[:, None, None]
    c = np.concatenate((mats[:, :n, :n], np.eye(n)[None]), axis=0)
    base = 2 + 4 * (x @ x[:n].T)
    cross = mats[:, n:, :n]
    pred = []
    for response in target.T:
        def loss(theta):
            w = np.exp(theta)
            k = base[:n] + np.einsum('i,ijk->jk', w, c)
            factor = cho_factor(k, check_finite=False)
            alpha = cho_solve(factor, response, check_finite=False)
            inv = cho_solve(factor, np.eye(n), check_finite=False)
            value = .5 * (response @ alpha) + np.log(np.diag(factor[0])).sum()
            grad = .5 * w * np.einsum('ij,kij->k', inv - np.outer(alpha, alpha), c)
            return value, grad
        fit = minimize(loss, [-3, -3, -9], jac=True, method='L-BFGS-B',
                       bounds=[(-17, 3), (-17, 3), (-13, -3)], options={'maxiter': 80})
        w = np.exp(fit.x)
        k = base[:n] + np.einsum('i,ijk->jk', w, c)
        k_cross = base[n:] + np.einsum('i,ijk->jk', w[:2], cross)
        pred.append(k_cross @ cho_solve(cho_factor(k), response))
    return np.column_stack(pred)


def _components(study):
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    z, h = _coordinates(study, x)
    u, s, vt = np.linalg.svd(train['traj'], full_matrices=False)
    r = min(2, len(s))
    target = np.column_stack((train['y'], u[:, :r] * s[:r]))
    poly = .75 * _polynomial(h, n, target) + .25 * _polynomial(x, n, target)
    smooth = (_kernel(x, x, n, target, 1.) + _kernel(x, z, n, target, 2.)
              + _kernel(x, h, n, target, 2.)) / 3
    return poly, smooth, vt[:r]


def _covariance_blocks(x, coordinates, n, mode):
 tr = slice(0, n)
 Ws = coordinates
 K=[]
 def add(f):
  if f.ndim==1:f=f[:,None]
  q=f@f.T; scale=np.sqrt(np.mean(q[tr,tr]**2)); K.append(q/max(scale,1e-12))
 def gram(q):K.append(q / np.sqrt(np.mean(q[tr,tr]**2)))
 def pol(w,d):return np.stack([np.prod(w[:,ids],1) for ids in combinations_with_replacement(range(4),d)],1)
 for w in ([x] if mode in ['iso','ard','bare'] else Ws[:5] if mode in ['full','sparse'] else Ws):
  v=w-w[tr].mean(0); G=v@v.T; d=np.maximum(G.diagonal()[:,None]+G.diagonal()[None,:]-2*G,0)
  if mode!='bare':
   for l in [.65,1.15,2.2]:gram(np.exp(-d/(2*l*l)))
  for j in range(2,6):
   f=pol(w,j)
   add(f)
   if (mode in ['ard','sparse'] and j<=3):
    for k in f.T:add(k)
  if mode not in ['full','sparse','all']:continue
  for j in [.22,1.25]:
   for f in [np.tanh(j*w),np.sin(j*w),w/(1+j*abs(w)),w/(1+j*w*w),np.sqrt(1+j*w*w)-1]:
    add(f);add(np.einsum('ni,nj->nij',f,w).reshape(len(w),-1))
   i,k=np.triu_indices(4,1); dd=w[:,i]-w[:,k]; ss=w[:,i]+w[:,k]
   for q in [dd,ss]:
    add(np.tanh(j*q));add(np.sin(j*q));add(np.einsum('ni,nj->nij',q/(1+j*q*q),w).reshape(len(w),-1))
  for j in [.22,1.25]:
   f=pol(w,2); add((f[:,:,None]/(1+j*w[:,None,:]**2)).reshape(len(w),-1))
 return np.array(K)

def _gram_prediction(x, blocks, response, penalty):
    n = len(response)
    eye = np.eye(n)
    train_components = np.concatenate((blocks[:, :n, :n], eye[None]))
    flat = np.ascontiguousarray(train_components.reshape(len(train_components), -1))
    base = 2 + 2 * (x @ x[:n].T)

    def covariance(weights):
        return base[:n] + np.einsum('i,ij->j', weights, flat).reshape(n, n)

    def objective(theta):
        weights = np.exp(theta)
        factor = cho_factor(covariance(weights), check_finite=False)
        alpha = cho_solve(factor, response, check_finite=False)
        inv = cho_solve(factor, eye, check_finite=False)
        regularizer = penalty * weights[:-1]**.35
        value = .5 * (response @ alpha) + np.log(factor[0].diagonal()).sum() + regularizer.sum()
        grad = .5 * weights * np.einsum('ij,j->i', flat, (inv - np.outer(alpha, alpha)).ravel())
        grad[:-1] += .35 * regularizer
        return value, grad

    initial = np.full(len(flat), -7.)
    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   bounds=[(-16, 1)] * len(blocks) + [(-13, -2)],
                   options={'maxiter': 120, 'ftol': 1e-8})
    weights = np.exp(fit.x)
    factor = cho_factor(covariance(weights), check_finite=False)
    cross = base[n:] + np.einsum('i,ijk->jk', weights[:-1], blocks[:, n:, :n])
    return cross @ cho_solve(factor, response, check_finite=False)


def _panel_coordinates(study, x):
    z, h = _coordinates(study, x)
    panel = study['panel_b']
    b = np.linalg.lstsq(_affine(panel['u']), panel['v'], rcond=None)[0]
    coordinates = [x, z, h, _affine(z) @ b, _affine(x) @ b]
    panel = study['panel_c']
    for w in (x, z, h):
        for key in ('v1', 'v2'):
            def basis(a):
                transformed = a / (1 + .84 * abs(a)) if key == 'v1' else np.sqrt(1 + .959**2 * a*a)
                return np.column_stack((_affine(a), transformed))
            c = np.linalg.lstsq(basis(panel['u']), panel[key], rcond=None)[0]
            coordinates.append(basis(w) @ c)
    return coordinates


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    response = np.asarray(train['y'], dtype=float)
    n = len(response)
    poly, smooth, vt = _components(study)
    predictions = [.65 * poly[:, 0] + .35 * smooth[:, 0]]
    coords = _panel_coordinates(study, x)
    for mode, penalty in (('iso', 0.), ('full', 3.), ('all', 0.)):
        blocks = _covariance_blocks(x, coords, n, mode)
        predictions.append(_gram_prediction(x, blocks, response, penalty))
    y = np.mean(predictions, axis=0)
    state = (.4 * poly[:, 1:] + .6 * smooth[:, 1:]) @ vt
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
