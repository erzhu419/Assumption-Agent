import numpy as np
from itertools import combinations


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-9)[0]


def _cross(z):
    return np.column_stack([z[:, i] * z[:, j] for i, j in combinations(range(z.shape[1]), 2)])


def _calibrate(p, x):
    u, v = p['u'], p['v']
    w = np.roll(u, -1, axis=1)
    h = np.stack([np.ones_like(u), u, w, u*w, u**3], axis=2)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(6):
        c = np.stack([_ls(h[keep, j], v[keep, j]) for j in range(4)])
        e = np.linalg.norm(np.sum(h*c, axis=2)-v, axis=1)
        nxt = e <= max(0.003, 4*np.median(e))
        if nxt.sum() < 10 or np.array_equal(nxt, keep):
            break
        keep = nxt
    z = (x-c[:, 0])/c[:, 1]
    for _ in range(80):
        w = np.roll(z, -1, axis=1)
        err = c[:, 0]+c[:, 1]*z+c[:, 2]*w+c[:, 3]*z*w+c[:, 4]*z**3-x
        step = np.clip(err/np.maximum(c[:, 1]+c[:, 3]*w+3*c[:, 4]*z*z, 0.15), -0.5, 0.5)
        z -= step
        if np.max(abs(step)) < 1e-10:
            break
    return z, (x-c[:, 0])/c[:, 1]


def _features(study, x):
    a, diag = _calibrate(study['panel_a'], x)
    p = study['panel_b']
    b = _ls(np.c_[np.ones(len(p['u'])), p['u']], p['v'])
    p = study['panel_c']; u = p['u']
    cq = _ls(np.c_[np.ones(len(u)), _cross(u)], p['v1'])
    cc = _ls(np.c_[np.ones(len(u)), np.cos(2.4*u)], p['v2'])
    for z in (a, a@b[1:]+b[0], (a-b[0])@np.linalg.pinv(b[1:]), diag, x):
        q = _cross(z)
        p1 = np.c_[np.ones(len(z)), q]@cq
        cos = np.cos(2.4*z)
        p2 = np.c_[np.ones(len(z)), cos]@cc
        base = np.c_[z, z*z, q, z**3, z**4]
        yield base
        for k in (1., 1.6, 2.4, 3.2, 4.5):
            ss, co = np.sin(k*z), np.cos(k*z)
            yield np.c_[base, ss, co, np.sin(k*q), np.tanh(k*q), _cross(np.tanh(k*z)), p1, p2]
        pp = list(combinations(range(4), 2))
        a1 = z[:, [i for i,j in pp]]; a2 = z[:, [j for i,j in pp]]
        sums = np.c_[a1+a2, a1-a2, a1+2*a2, a1-2*a2, 2*a1+a2, 2*a1-a2]
        triples = np.column_stack([np.prod(z[:, j], axis=1) for j in combinations(range(4), 3)])
        ridge = np.c_[z, q, sums, p1, p2, p1+p2, p1-p2, triples]
        yield np.c_[base, triples, a1*q, a2*q, q/(1+0.5*(a1*a1+a2*a2)), *[f(k*ridge) for k in (0.8, 1.4, 2., 2.4, 3., 4., 5., 6., 8.) for f in (np.sin, np.cos)], np.tanh(ridge), ridge/(1+abs(ridge))]


def _paths(f, target, i, j):
    mu=f[i].mean(0); sd=f[i].std(0); mask=sd>1e-9
    l=(f[i][:,mask]-mu[mask])/sd[mask]; v=(f[j][:,mask]-mu[mask])/sd[mask]
    m=target[i].mean(0); h=target[i]-m
    g=l@l.T; vv=v@l.T; d, U=np.linalg.eigh(g)
    rhs=U.T@h
    out=[vv@U@(rhs/(np.maximum(d,0)[:,None]+r))+m for r in (0.02,0.2,2.,10.)]
    vals=[]
    for k in range(h.shape[1]):
        y=h[:,k]; rem=y.copy(); inds=[]; track=[]
        for count in range(1, 13):
            sc=abs(l.T@rem); sc[inds]=-1
            ix=int(np.argmax(sc)); inds.append(ix)
            w=_ls(l[:,inds],y); rem=y-l[:,inds]@w
            if count in (3,5,8,12):
                track.append(v[:,inds]@w+m[k])
        vals.append(track)
    return out+[np.column_stack([v[k] for v in vals]) for k in range(4)]


def run_node(inputs: dict) -> dict:
    study, sparse = inputs['__input__'], inputs['node_00']
    tr=study['train']; times=study['eval_times']; n=len(tr['x']); traj=tr['traj']
    _, _, vt=np.linalg.svd(traj, full_matrices=False)
    basis=vt[:2]
    target=np.c_[tr['y'], traj@basis.T]
    x=np.vstack([tr['x'], study['test']['x']])
    splits=[(np.setdiff1d(np.arange(n), j),j) for j in np.array_split(np.random.default_rng(112).permutation(n),5)]
    allscores=[]; preds=[]
    for f in _features(study,x):
        val=[]
        for i,j in splits:
            pp=np.asarray(_paths(f,target,i,j))
            e1=np.mean(abs(pp[:,:,0]-target[j,0]),axis=1)
            e2=np.mean(abs(pp[:,:,1:]@basis-traj[j]),axis=(1,2))
            val.append(np.c_[e1,e2])
        val=np.asarray(val)
        allscores.extend(val.mean(0)+0.2*val.max(0))
        preds.extend(_paths(f,target,np.arange(n),np.arange(n,len(x))))
    scores=np.asarray(allscores); preds=np.asarray(preds)
    outputs=[]
    for k, sl in [(0,slice(0,1)),(1,slice(1,3))]:
        ix=np.argsort(scores[:,k])[:8]
        delta=scores[ix,k]-scores[ix[0],k]
        w=np.exp(-delta/max(0.006,0.075*scores[ix[0],k])); w/=w.sum()
        outputs.append(np.sum(w[:,None,None]*preds[ix,:,sl],axis=0))
    pred=np.c_[outputs[0],outputs[1]@basis]
    prior=np.column_stack([sparse['y_hat'],sparse['state_pred']])
    weight=np.r_[0.5,np.full(len(times),0.65)]
    return {'panel_pred': (pred-(1-weight)*prior)/weight}
