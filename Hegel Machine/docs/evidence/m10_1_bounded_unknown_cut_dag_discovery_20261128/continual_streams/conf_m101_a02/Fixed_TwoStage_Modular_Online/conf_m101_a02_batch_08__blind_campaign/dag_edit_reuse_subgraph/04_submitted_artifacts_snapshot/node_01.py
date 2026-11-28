"""Fit the label and trajectory readouts from the frozen panel representation."""
import numpy as np


def _poly(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j]))


def _reference(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    def design(u, j):
        p, q = u[:, j], u[:, (j + 1) % 4]
        return np.column_stack((np.ones(len(u)), p, q, p*q, p**3))
    ds = [design(u, j) for j in range(4)]
    w = np.ones(len(u))
    for _ in range(20):
        c = np.array([np.linalg.lstsq(d*w[:, None], v[:, j]*w, rcond=None)[0] for j, d in enumerate(ds)])
        e = np.linalg.norm(np.column_stack([d@b for d, b in zip(ds, c)]) - v, axis=1)
        w = 1/np.hypot(1, e/max(np.median(e), 1e-8))
    keep = np.argsort(e)[:-3]
    bias, lin, neighbor, pair, cubic = np.array([np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0] for j, d in enumerate(ds)]).T
    affine = lambda x: np.column_stack((np.ones(len(x)), x))
    q = affine(x) @ np.linalg.lstsq(affine(v[keep]), u[keep], rcond=None)[0]
    j = np.arange(4)
    for _ in range(20):
        p = np.roll(q, -1, axis=1)
        r = bias + lin*q + neighbor*p + pair*q*p + cubic*q**3 - x
        if np.max(np.abs(r), initial=0) < 1e-11:
            break
        J = np.zeros((len(q), 4, 4))
        J[:, j, j] = lin + pair*p + 3*cubic*q*q
        J[:, j, (j+1)%4] = neighbor + pair*q
        q -= np.clip(np.linalg.solve(J, r[..., None])[..., 0], -.5, .5)
    return q


def _fit(d, y, n, penalty):
    scale = np.maximum(np.linalg.norm(d[:n], axis=0), 1e-10)
    d = d / scale
    a = d[:n]
    return d[n:] @ np.linalg.solve(a.T@a + penalty*np.eye(a.shape[1]), a.T@y)


def run_node(inputs: dict) -> dict:
    s, enc = inputs['__input__'], inputs['node_00']
    tr = s['train']
    n = len(tr['x'])
    z = np.concatenate((enc['z_tr'], enc['z_te']))
    u = _reference(s, np.concatenate((tr['x'], s['test']['x'])))
    a = _poly(z)
    trajectory = np.column_stack((a, u, (z[:, :, None]*u[:, None, :]).reshape(len(z), 16)))
    target = np.asarray(tr['y']).reshape(-1)
    y = np.mean([_label_model(u, z, target, (3, (2,), 'sparse', 2)), _label_model(u, z, target, (3, (2, 3), 'sparse', 1))], axis=0)
    state = _fit(trajectory, tr['traj'], n, 1e-6)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}

def _label_features(u, z):
    n = len(u)
    a = np.column_stack((np.ones(n), z))
    w = u * np.roll(z, 1, axis=1)
    balance = w.sum(axis=1, keepdims=True)
    base = np.column_stack((a, balance))
    bases = [base, np.column_stack((a, w)), np.column_stack((base, u)), np.column_stack((base, _poly(z)[:, 5:]))]
    mix = (u[:, :, None] * z[:, None, :]).reshape(n, 16)
    cross = [mix, w**2, w**3, np.abs(w)]
    uni, pairs, aggregate, cubic = [], [], [], []
    for q in (u, z, w):
        uni.extend([q, q**2, q**3, q**4, np.abs(q), q*np.abs(q)])
        for k in (.75, 1.5, 3.):
            uni.extend([np.sin(k*q), np.cos(k*q), np.tanh(k*q), q/(1+k*q*q), 1/(1+k*q*q)])
        for i in range(4):
            for j in range(i+1, 4):
                for sign in (-1, 1):
                    d = q[:, i] + sign*q[:, j]
                    pairs.extend([d**2, d**3, d**4, np.abs(d), np.sin(1.5*d), np.cos(1.5*d), 1/(1+d*d)])
        for i in range(4):
            for j in range(i, 4):
                for k in range(j, 4):
                    cubic.append(q[:, i]*q[:, j]*q[:, k])
        aggregate.extend([q.sum(1), (q*q).sum(1), (q**3).sum(1), (q**4).sum(1), (q*np.abs(q)).sum(1), np.abs(q).sum(1), np.prod(q, axis=1)])
    dirs = np.column_stack((balance, z @ [-1,-1,1,1], z @ [-1,-1,1,1] + .7*balance[:,0], z[:,0]-z[:,1], z[:,2]-z[:,3]))
    for k in (.5, 1., 2., 4.):
        aggregate.extend([np.sin(k*dirs), np.cos(k*dirs), np.tanh(k*dirs), dirs/(1+k*dirs**2), 1/(1+k*dirs**2)])
    aggregate.extend([dirs**2, dirs**3, np.abs(dirs)])
    cross.extend([np.column_stack([z[:,i]*np.roll(u,k,axis=1)[:,i]**p for k in range(4) for i in range(4)]) for p in (2,3,4)])
    cross.extend([np.column_stack([u[:,i]*np.roll(z,k,axis=1)[:,i]**p for k in range(4) for i in range(4)]) for p in (2,3)])
    cross.extend([z*balance, u*balance, w*balance])
    groups = [np.column_stack(q) for q in (uni,pairs,aggregate,cross,cubic)]
    return bases, groups


def _label_model(u, z, t, spec, fallback=None):
    ib, igr, mode, strength = spec
    if mode == 'legacy':
        return fallback
    bases, groups = _label_features(u, z)
    n = len(t)
    a = bases[ib]
    f = np.column_stack([groups[j] for j in igr]) if igr else np.empty((len(z),0))
    q, r = np.linalg.qr(a[:n], mode='reduced')
    basecoef = np.linalg.solve(r, q.T@t)
    basepred = a[n:]@basecoef
    if not f.shape[1]:
        return basepred
    proj = np.linalg.solve(r, q.T@f[:n])
    f = f-a@proj
    norm = np.linalg.norm(f[:n], axis=0)
    keep = norm > 1e-5
    f = f[:,keep]/norm[keep]
    g = f[:n]
    y = t-q@(q.T@t)
    if mode == 'ridge':
        return basepred+f[n:]@g.T@np.linalg.solve(g@g.T+strength*np.eye(n), y)
    active = []
    pred = basepred
    for _ in range(int(strength)):
        d = np.column_stack((a[:n], g[:,active]))
        q1 = np.linalg.qr(d, mode='reduced')[0]
        e = t-q1@(q1.T@t)
        gr = g-q1@(q1.T@g)
        h = np.maximum(1-(q1*q1).sum(1), 1e-8)
        grnorm = np.linalg.norm(gr, axis=0)
        v = gr / np.maximum(grnorm, 1e-8)
        e_loo = (e[:,None]-v*(v.T@e))/np.maximum(h[:,None]-v*v, 1e-8)
        loss = np.mean(np.abs(e_loo), axis=0)
        loss[grnorm<1e-5] = np.inf
        if np.all(~np.isfinite(loss)) or loss.min() >= .995*np.mean(np.abs(e/h)):
            break
        active.append(int(loss.argmin()))
        ds = np.column_stack((a, f[:,active]))
        pred = ds[n:] @ np.linalg.lstsq(ds[:n], t, rcond=None)[0]
    return pred
