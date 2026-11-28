import numpy as np
from itertools import combinations
from scipy.optimize import least_squares
from scipy.linalg import solve


def _fit(x,y):
    return np.linalg.lstsq(x,y,rcond=1e-10)[0]


def _coordinates(study,x):
    a=study['panel_a']; u=np.asarray(a['u']); v=np.asarray(a['v'])
    i,j=np.triu_indices(len(u),1)
    du=u[i]-u[j]; dv=v[i]-v[j]
    g=np.nanmedian(np.divide(dv,du,out=np.full_like(du,np.nan),where=abs(du)>1e-9),axis=0)
    o=np.median(v-u*g,axis=0)
    for k in range(x.shape[1]):
        r=abs(v[:,k]-u[:,k]*g[k]-o[k]); keep=r<max(.002,5*np.median(r))
        if keep.sum()>=3:
            o[k],g[k]=_fit(np.c_[np.ones(keep.sum()),u[keep,k]],v[keep,k])
    return (x-o)/g


def _panel_features(study, z):
    def affine(x):
        return np.column_stack((np.ones(len(x)), x))

    p = study['panel_c']
    u, v1, v2 = (np.asarray(p[k]) for k in ('u', 'v1', 'v2'))

    def soft(x, k):
        return x/(1+k*np.abs(x))

    def d1(x, k):
        return affine(np.column_stack((x, soft(x, k))))

    def r1(k):
        d = d1(u, k)
        return (d@_fit(d, v1)-v1).ravel()

    k = least_squares(r1, np.full(4, .83), bounds=(.25, 2.5), max_nfev=40).x
    soft_z = soft(z, k)
    sat = d1(z, k)@_fit(d1(u, k), v1)

    def r2(k):
        d = affine(np.cos(k*u))
        return (d@_fit(d, v2)-v2).ravel()

    freq = least_squares(r2, [2.4], bounds=(1.4, 3.4), max_nfev=30).x
    cos_z = np.cos(freq*z)
    cos_mix = affine(cos_z)@_fit(affine(np.cos(freq*u)), v2)
    return z, soft_z, sat, cos_z, cos_mix


def _readout(channels, y):
    from sklearn.linear_model import lars_path
    import warnings
    z, soft, sat, cos_z, cos_mix = channels
    n = len(y)
    base = np.column_stack(channels)
    i, j = np.triu_indices(base.shape[1])
    # Prioritize products of the matched panel channels, while allowing the data
    # to select additional interactions from the complete quadratic dictionary.
    core = np.column_stack((base, sat*cos_mix, sat[:, 0]*cos_mix[:, 1]))
    full = np.column_stack((core, base[:, i]*base[:, j]))
    mean = full[:n].mean(0)
    scale = np.maximum(full[:n].std(0), 1e-10)
    f = (full-mean)/scale
    scale_prior = np.ones(f.shape[1])
    scale_prior[:core.shape[1]] = 2.
    scale_prior[base.shape[1]:core.shape[1]] = 3.
    f *= scale_prior
    m = y.mean(0)
    ridge = f[:n]@f[:n].T + .07*np.eye(n)
    broad = f[n:]@f[:n].T@solve(ridge, y-m, assume_a='pos')+m
    preds = []
    for k in range(y.shape[1]):
        target = y[:, k]-m[k]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, _, path = lars_path(f[:n], target, method='lasso', max_iter=135)
        models = {}
        for v in path.T:
            ix = np.flatnonzero(np.abs(v)>1e-9)
            dof = len(ix)
            if not 4 <= dof <= min(27, n//2):
                continue
            g = f[:n, ix]
            cov = g.T@g + .007*np.eye(dof)
            coef = solve(cov, g.T@target, assume_a='pos')
            resid = target-g@coef
            lever = np.einsum('ij,ji->i', g, solve(cov, g.T, assume_a='pos')) + 1/n
            loo = np.mean(np.abs(resid)/np.maximum(1-lever, .05))
            score = loo*(1+.012*dof)
            old = models.get(tuple(ix))
            if old is None or score < old[0]:
                models[tuple(ix)] = (score, f[n:, ix]@coef+m[k])
        if models:
            best = sorted(models.values(), key=lambda pair:pair[0])[:6]
            scores = np.array([v[0] for v in best]); weight = (scores[0]/np.maximum(scores, 1e-12))**8
            preds.append(.96*np.average([v[1] for v in best],axis=0,weights=weight)+.04*broad[:,k])
        else:
            preds.append(broad[:, k])
    return np.column_stack(preds)


def _predict(study):
    tr = study['train']; z = _coordinates(study, np.vstack((tr['x'], study['test']['x'])))
    t = np.asarray(study['eval_times']); traj = np.asarray(tr['traj'])
    def basis(r):
        return -np.expm1(-t[:, None]*r)
    def profile(r):
        q = basis(r)
        return (q@_fit(q, traj.T)-traj.T).ravel()
    q = basis(least_squares(profile, [.4125, 1.81], bounds=([.05, .9], [.89, 5.])).x)
    target = np.column_stack((tr['y'], _fit(q, traj.T).T))
    pred = _readout(_panel_features(study, z), target)
    return np.column_stack((pred[:, 0], pred[:, 1:]@q.T))


def run_node(inputs: dict) -> dict:
    study=inputs['__input__']; offset=inputs['node_00']
    base=np.column_stack((offset['y_hat'],offset['state_pred']))
    pred=_predict(study)
    return {'panel_pred':base+(pred-base)/np.r_[.5,np.full(base.shape[1]-1,.65)]}
